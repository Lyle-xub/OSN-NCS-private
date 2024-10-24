from telnetlib import Telnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from .optical_unit import *
from lightridge import layers
import matplotlib.pyplot as plt

from .slm_func import *
from .has_func import *
from .cmos_func import *
from .dmd_func import *
from interface.imageInterface import *
from .utils import generate_square_coordinates
from .optical_unit import dorefa_w
from .event_func import *
from scipy.ndimage import zoom
from PIL import Image
import multiprocessing
from multiprocessing import Process, Lock, Manager
from multiprocessing import shared_memory
from multiprocessing.sharedctypes import Value, Array
from viztracer import VizTracer

import matplotlib.colors as color
import scipy.io as sio

import scienceplots

from matplotlib.colors import to_rgba

plt.style.use(["science", "nature"])

parula_list = sio.loadmat("parula.mat")["parula"]
parula = color.ListedColormap(parula_list, "parula")


class DDNN(nn.Module):
    def __init__(
        self,
        whole_dim,
        phase_dim,
        pixel_size,
        focal_length,
        wave_lambda,
        intensity_mode=False,
        scalar=None,
        cfg=None,
        layer_num=6,
    ):
        super(DDNN, self).__init__()

        # Use a dictionary for repetitive settings or properties
        optical_properties = {
            "whole_dim": whole_dim,
            "phase_dim": phase_dim,
            "pixel_size": pixel_size,
            "focal_length": focal_length,
            "wave_lambda": wave_lambda,
        }

        canvas_size = (phase_dim, phase_dim)
        square_size = cfg.reconstruction.square_size
        pattern = [3, 4, 3]
        square_coordinates = generate_square_coordinates(
            canvas_size, square_size, pattern
        )

        ordered_coordinates = [
            *square_coordinates[0:3],  # First row
            *square_coordinates[3:7],  # Second row
            *square_coordinates[7:10],  # Third row
        ]
        pad = (whole_dim - phase_dim) // 2

        # Setting detector configurations
        cfg.reconstruction.dety = [coord[0] for coord in ordered_coordinates]
        cfg.reconstruction.detx = [coord[1] for coord in ordered_coordinates]

        # Conditionally setup the model based on configuration

        self.prop1 = AngSpecProp(
            whole_dim=whole_dim,
            pixel_size=pixel_size,
            focal_length=focal_length,
            wave_lambda=wave_lambda,
        )

        self.prop = AngSpecProp(
            whole_dim=whole_dim,
            pixel_size=pixel_size,
            focal_length=focal_length,
            wave_lambda=wave_lambda,
        )

        self.scalar = (
            torch.tensor(1.0)
            if scalar is None
            else torch.tensor(scalar, dtype=torch.float32)
        )
        self.w_scalar = nn.Parameter(self.scalar)
        self.cfg = cfg

        # Set up the detector
        det_x_loc = [coord[1] + pad for coord in ordered_coordinates]
        det_y_loc = [coord[0] + pad for coord in ordered_coordinates]
        self.detector = layers.Detector(
            det_x_loc,
            det_y_loc,
            size=whole_dim,
            det_size=square_size,
            mode="mean",
            intensity_mode=intensity_mode,
        )
        self.layer_num = 6
        for i in range(1, self.layer_num + 1):
            setattr(self, f"phase{i}", PhaseMask(whole_dim, phase_dim))

        self.input = Incoherent_Int2Complex()
        for i in range(1, self.layer_num + 1):
            setattr(self, f"dmd{i}", DMD(whole_dim, phase_dim))

        self.alpha = nn.Parameter(torch.tensor(16.0), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor(1.00), requires_grad=False)

        self.dmd1 = DMD(whole_dim, phase_dim)

    def forward(self, input_field, physical_img=None):
        x = self.input(dorefa_a(input_field, 1))
        # x = getattr(self, 'phase1')(x1)
        # x = self.dmd1(self.prop(x))

        for i in range(1, self.layer_num + 1):  # Continue for phase2 and phase3
            phase_layer = getattr(self, f"phase{i}")
            # dmd = getattr(self, f"dmd{i}")
            dmd = self.dmd1
            x = phase_layer(x)
            if i == self.layer_num:
                x = self.prop(x)
                break
            x = dmd(self.prop(x))
        # out = self.detector(self.w_scalar.cuda() * x)
        out = self.w_scalar.cuda() * x
        return out

    def physical_forward_one_layer_cmos(self, x, phase, expose_time=0.04):

        DMD = run_a_image(x.squeeze())  # DMD update
        if isinstance(phase, torch.Tensor):
            phase = phase.cpu().detach().numpy()
            unique_values = np.sort(np.unique(phase))
            value_to_int = {v: i for i, v in enumerate(unique_values)}
            phase = np.vectorize(value_to_int.get)(phase)

        write_one_image_event_has(phase)  # slm update
        frame, orig_data = capture_one_image(expose_time=expose_time)  # cmos capture
        time.sleep(1)
        DMD.Halt()
        DMD.FreeSeq()
        DMD.Free()

        return torch.tensor(frame).unsqueeze(0), orig_data

    def physical_forward_one_layer_event(self, x, phase):

        e = multiprocessing.Event()
        if isinstance(phase, torch.Tensor):
            phase = phase.cpu().detach().numpy()
            unique_values = np.sort(np.unique(phase))
            value_to_int = {v: i for i, v in enumerate(unique_values)}
            phase = np.vectorize(value_to_int.get)(phase)

        p2 = Process(target=run_a_image_event, args=(x.squeeze(), e))
        p2.start()

        p3 = Process(target=write_one_image_event_has, args=(phase, e))
        p3.start()

        device = init_device()
        set_camera_params(device)

        events_iterator = EventsIterator.from_device(device, delta_t=2000)
        e.set()
        height, width = events_iterator.get_size()

        img = np.zeros((height, width), dtype=np.uint8)
        img_bgr = np.zeros((height, width), dtype=np.uint8)

        for evs in events_iterator:
            # frame_gen.process_events(evs)

            begin_tick = time.time()
            img = events_to_diff_image(evs, sensor_size=(height, width))
            img_bgr[img < 0] = 255
            img_bgr[img > 0] = 0

            image = img_bgr[0:720, 280:1000]
            print("event", time.time() - begin_tick)

            cv2.imwrite("img.png", img_bgr)
            logger.info("image process")
            break
        e.clear()

        p2.join()
        p3.join()

        time.sleep(1)

        return torch.tensor(image).unsqueeze(0)

    def plot_phases_and_output(
        self,
        input_field,
        physical_img,
        show=True,
        num=1,
    ):
        x = self.input(dorefa_a(input_field, 1))
        self._plot_output(x, "Output0.9", show)

        for i in range(1, self.layer_num + 1):
            phase_layer = getattr(self, f"phase{i}")
            # dmd = getattr(self, f"dmd{i}")
            dmd = self.dmd1
            plot_phase_name = f"Phase{i}"
            plot_output_name = f"Output{i}"

            x = phase_layer(x)
            self._plot_phase(phase_layer.w_p, plot_phase_name, show)

            if i == self.layer_num:
                break

            x = self.prop(x)
            self._plot_output(x, plot_output_name, show)
            x = dmd(x)
            self._plot_output(x, f"{plot_output_name}.1", show)

        x = self.prop(x)
        x = dmd(x)
        self._plot_output(x, f"Output{i}.1", show)

        out = self.detector(self.w_scalar.cuda() * x)
        print(out)

    @staticmethod
    def map_values_to_int(array):
        unique_values = np.sort(np.unique(array))
        return {v: i for i, v in enumerate(unique_values)}

    def save_phase_image(self, phase, title):
        file_path = f"{self.cfg.reconstruction.log_dir}/{title}.bmp"
        cv2.imwrite(file_path, phase)
        return file_path

    @staticmethod
    def create_padded_image(image_path, size=(1272, 1024), color="black"):
        with Image.open(image_path) as img:
            original_size = img.size
            padded_img = Image.new("L", size, color=color)
            offset = (
                (size[0] - original_size[0]) // 2,
                (size[1] - original_size[1]) // 2,
            )
            padded_img.paste(img, offset)
            return padded_img

    def _plot_phase(self, phase_data, title, show):
        phase = dorefa_w(phase_data, 8).cpu().detach().numpy()
        # print(np.unique(phase))
        mapped_arr = np.vectorize(DDNN.map_values_to_int(phase).get)(phase)
        phase = mapped_arr.squeeze(0)

        plt.imshow(phase, cmap="Spectral")
        plt.title(title)
        if show:
            plt.colorbar()
            plt.show()
            flattened_weights = phase.flatten()
            plt.hist(flattened_weights, bins=30, color="blue", edgecolor="black")
            plt.title("Distribution of Weights")
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency")
            plt.show()

        phase_img_path = self.save_phase_image(phase, title)
        phase_img = self.create_padded_image(phase_img_path)
        phase_img.save(f"{self.cfg.reconstruction.log_dir}/padded_{title}.bmp")
        phase_img.transpose(Image.FLIP_TOP_BOTTOM).save(
            f"{self.cfg.reconstruction.log_dir}/padded_{title}.bmp"
        )

        cropped_image1 = crop_center(
            Image.open(f"{self.cfg.reconstruction.log_dir}/padded_{title}.bmp")
        )
        cropped_image2 = crop_center(
            Image.open(
                "D:\project\control\img/reconstruction/BlazedGrating_Period2.bmp"
            )
        )
        superimposed_image = superimpose_images(cropped_image1, cropped_image2)
        superimposed_image_pil = Image.fromarray(superimposed_image)
        padded_image = pad_image(superimposed_image_pil)
        padded_image.save(
            f"{self.cfg.reconstruction.log_dir}/superimposed_phase_{title}.bmp"
        )

    @staticmethod
    def pad_and_rotate(image, size=(1600, 2560), angle=-48):
        pad_x = int((size[0] - image.shape[0]) / 2)
        pad_y = int((size[1] - image.shape[1]) / 2)
        padded_image = np.pad(
            image, ((pad_x, pad_x), (pad_y, pad_y)), "constant", constant_values=0
        )
        return rotate(padded_image, angle, reshape=False)

    def _plot_output(self, output, title, show):
        output = F.pad(output, (-150, -150, -150, -150), "constant", 0)
        output_tmp2 = (output.abs()).squeeze().abs().cpu().detach().numpy()
        print(output.shape)
        output_data = (
            F.interpolate(
                output.abs().unsqueeze(0) * 255,
                size=(
                    int(output.abs().shape[1] * 1.649),
                    int(output.abs().shape[1] * 1.649),
                ),
                mode="nearest",
            )
            .squeeze()
            .cpu()
            .detach()
            .numpy()
        )

        # output_data = (output.abs() * 255).detach().squeeze().cpu().numpy().astype(np.uint8)

        if ".1" in title:
            plt.imshow(output_data / 255, cmap=parula)
        elif "4" in title:
            plt.imshow((output_tmp2))
        else:
            plt.imshow((output_tmp2))

        # no clorbar
        # no axis
        plt.axis("off")
        # transparent background
        # plt.gca().set_axis_off()
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())

        # plt.title(title)
        plt.savefig(
            f"{self.cfg.reconstruction.log_dir}/{title}.png",
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )
        if show:
            # plt.colorbar()
            plt.show()

        rotated_image = DDNN.pad_and_rotate(output_data)
        if ".1" in title or ".9" in title:
            cv2.imwrite(
                f"{self.cfg.reconstruction.log_dir}/{title}_load.png", rotated_image
            )
