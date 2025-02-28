from telnetlib import Telnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

from lightridge import layers
import matplotlib.pyplot as plt

from scipy.ndimage import rotate
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
import cv2

from matplotlib.colors import to_rgba
from system.utils import crop_center, superimpose_images, pad_image
import system.net as net
from system.optical_unit import *

plt.style.use(["science", "nature"])
parula_list = sio.loadmat("D:\project\control\parula.mat")["parula"]
parula = color.ListedColormap(parula_list, "parula")


class BaseModel(nn.Module):

    def __init__(self, params=None):
        super().__init__()
        self.params = params
        if params is not None:
            self.initialize(params)  # Initialize if params provided
            self.checkpoint_dir = params.checkpoint_dir
        else:
            self.checkpoint_dir = 'ck'

    def initialize(self, params):
        self.params = params
        self.layer_num = params.core["network"]["layer_num"]
        self.wd = params.core["network"]["whole_dim"]
        self.pd = params.core["network"]["phase_dim"]

        for i in range(1, self.layer_num + 1):
            setattr(self, f"phase{i}", PhaseMask(self.wd, self.pd))
            setattr(
                self,
                f"unet{i}",
                net.ComplexUNet(
                    (self.wd, self.wd),
                    kernel_size=7,
                    bn_flag=False,
                    CB_layers=[3, 3, 3],
                    FM_num=[8, 16, 32],
                ),
            )

    @staticmethod
    def map_values_to_int(array):
        unique_values = np.sort(np.unique(array))
        return {v: i for i, v in enumerate(unique_values)}

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

    def save_phase_image(self, phase, title):
        file_path = f"{self.checkpoint_dir}/{title}.bmp"
        cv2.imwrite(file_path, phase)
        return file_path

    def _plot_phase(self, phase_data, title, show):
        phase = dorefa_w(phase_data, 8).cpu().detach().numpy()
        mapped_arr = np.vectorize(self.map_values_to_int(phase).get)(phase)
        phase = mapped_arr.squeeze(0).astype(np.uint8)
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

        # print(np.unique(phase))
        phase_img_path = self.save_phase_image(phase, title)
        phase_img = self.create_padded_image(phase_img_path)
        phase_img.save(f"{self.checkpoint_dir}/padded_{title}.bmp")
        phase_img.transpose(Image.FLIP_TOP_BOTTOM).save(f"{self.checkpoint_dir}/padded_{title}.bmp")
        # phase_img.transpose(Image.FLIP_LEFT_RIGHT).save(f"{self.checkpoint_dir}/padded_{title}.bmp")

        cropped_image1 = crop_center(Image.open(f"{self.checkpoint_dir}/padded_{title}.bmp"),
                                     size=self.params.core["network"]["phase_dim"])
        cropped_image2 = crop_center(Image.open("BlazedGrating_Period3.bmp"),
                                     size=self.params.core["network"]["phase_dim"])
        superimposed_image = superimpose_images(cropped_image1, cropped_image2)
        superimposed_image_pil = Image.fromarray(superimposed_image)
        padded_image = pad_image(superimposed_image_pil)
        padded_image.save(f"{self.checkpoint_dir}/superimposed_phase_{title}.bmp")
        plt.close()

    @staticmethod
    def pad_and_rotate(image, size=(1600, 2560), angle=-45):
        pad_x = int((size[0] - image.shape[0]) / 2)
        pad_y = int((size[1] - image.shape[1]) / 2)
        padded_image = np.pad(image, ((pad_x, pad_x), (pad_y, pad_y)), "constant", constant_values=0)
        return rotate(padded_image, angle, reshape=False)

    def _plot_output(self, output, title, show):
        pad = (self.wd - self.pd) // 2
        output = F.pad(output, (-pad, -pad, -pad, -pad), "constant", 0)
        output_tmp2 = (output.abs()).squeeze().abs().cpu().detach().numpy()
        output_data = (F.interpolate(
            output.abs().unsqueeze(0) * 255,
            size=(
                int(output.abs().shape[1] * 1.575),
                int(output.abs().shape[1] * 1.575),
            ),
            mode="nearest",
        ).squeeze().cpu().detach().numpy())

        # output_data = (output.abs() * 255).detach().squeeze().cpu().numpy().astype(np.uint8)
        plt.figure()
        if ".1" in title:
            plt.imshow(output_data / 255, cmap=parula)
        elif "4" in title:
            plt.imshow((output_tmp2))
        else:
            plt.imshow((output_tmp2))

        plt.axis("off")
        # transparent background
        # plt.gca().set_axis_off()
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())

        # plt.title(title)
        plt.savefig(
            f"{self.checkpoint_dir}/{title}.png",
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )
        if show:
            plt.show()

        rotated_image = DDNN.pad_and_rotate(output_data)
        if ".1" in title or ".9" in title:
            cv2.imwrite(f"{self.checkpoint_dir}/{title}_load.png", rotated_image)
        plt.close()

    def plot_phases_and_output(
        self,
        input_field,
        show=True,
    ):
        x = self.input(dorefa_a(input_field, 1))
        self._plot_output(x, "Output0.9", show)
        dmd = getattr(self, "dmd1", 1)
        for i in range(1, self.layer_num + 1):
            phase_layer = getattr(self, f"phase{i}")
            plot_phase_name = f"Phase{i}"
            plot_output_name = f"Output{i}"
            x = phase_layer(x)
            self._plot_phase(phase_layer.w_p, plot_phase_name, show=show)
            if i == self.layer_num:
                break
            else:
                x = self.prop(x)
                self._plot_output(x, plot_output_name, show)
                x = dmd(x)
                self._plot_output(x, f"{plot_output_name}.1", show)
        x = self.prop(x)
        x = dmd(x)
        self._plot_output(x, f"Output{i}.1", show)
        out = self.detector(self.w_scalar.cuda() * x)
        print(out)


class DDNN(BaseModel):

    def __init__(
        self,
        params,
        whole_dim,
        phase_dim,
        pixel_size,
        focal_length,
        wave_lambda,
        layer_num=4,
        intensity_mode=False,
        scalar=None,
    ):
        super(DDNN, self).__init__(params)

        self.prop = AngSpecProp(
            whole_dim=whole_dim,
            pixel_size=pixel_size,
            focal_length=focal_length,
            wave_lambda=wave_lambda,
        )
        self.scalar = (torch.tensor(1.0) if scalar is None else torch.tensor(scalar, dtype=torch.float32))
        self.w_scalar = nn.Parameter(self.scalar)
        self.detector = layers.Detector(
            params.det_x_loc,
            params.det_y_loc,
            size=whole_dim,
            det_size=params.core["data"]["detectors"]["square_size"],
            mode="mean",
            intensity_mode=intensity_mode,
        )
        self.layer_num = layer_num
        self.input = Incoherent_Int2Complex()
        self.alpha = nn.Parameter(torch.tensor(16.0), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor(1.00), requires_grad=False)
        if params.camera == "cmos":
            self.dmd1 = QuantSigmoid()
        else:
            self.dmd1 = DMD(whole_dim, phase_dim)
        self.params = params

    def forward(self, input_field):
        x = self.input(dorefa_a(input_field, 1))
        dmd = self.dmd1
        for i in range(1, self.layer_num + 1):  # Continue for phase2 and phase3
            phase_layer = getattr(self, f"phase{i}")

            x = phase_layer(x)
            if i == self.layer_num:
                x = self.prop(x)
                break
            x = dmd(self.prop(x))
        out = self.w_scalar.cuda() * x
        return out


class s4nn(BaseModel):

    def __init__(self, params=None):
        super(s4nn, self).__init__(params)
        # if params == None:
        self.phase1 = PhaseMask(400, 250)
        self.prop = AngSpecProp(whole_dim=400, pixel_size=12.5e-6, focal_length=0.3, wave_lambda=520e-9)
        self.dmd = DMD(400, 250)
        self.input = Incoherent_Int2Complex()
        self.detector = Detector()
        self.w = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_field):
        x = self.input(input_field)
        x = self.phase1(x)
        x = self.prop(x)
        x = self.dmd(x)
        # plt.imshow(x.abs().cpu().detach().numpy().squeeze())
        # plt.show()

        x = self.detector(self.w * x)
        # print(x)
        return x
