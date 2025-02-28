from DANTE import optical_unit
import torch
import torch.nn as nn
from DANTE.optical_unit import *
from lightridge import layers

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import net2
import time
import monai
import os
import scipy.io as sio
import matplotlib.colors as color

parula_list = sio.loadmat("parula.mat")["parula"]
parula = color.ListedColormap(parula_list, "parula")
alpha = 0.4
beta = 1


def write_txt(dir, data_save):
    if isinstance(data_save, str):
        print(data_save)
    with open(dir, "a") as data:
        data.write(str(data_save) + "\n")


def correct(energe, label):
    corr = (energe.argmax(dim=-1) == label.argmax(dim=-1)).sum().item()
    corr /= energe.size(0)
    return corr


def pad_label(label, whole_dim, phase_dim, detx=None, dety=None, size=30):

    batch_size = label.shape[0]

    padded_labels = torch.zeros(batch_size, phase_dim, phase_dim, device=label.device)

    for i in range(batch_size):
        _, index = torch.max(label[i], dim=0)
        x_start = max(0, min(detx[int(index.cpu().numpy())], phase_dim - size))
        y_start = max(0, min(dety[int(index.cpu().numpy())], phase_dim - size))
        padded_labels[i, x_start:x_start + size, y_start:y_start + size] = 1.5

    padded_labels = F.pad(
        padded_labels,
        (
            (whole_dim - phase_dim) // 2,
            (whole_dim - phase_dim) // 2,
            (whole_dim - phase_dim) // 2,
            (whole_dim - phase_dim) // 2,
        ),
        "constant",
        0,
    )

    return padded_labels.to(label.device)


def save_image(img, save_dir, norm=False, Gray=False):
    imgPath = os.path.join(save_dir)
    # torchvision.utils.save_image(img, imgPath)
    # 改写：torchvision.utils.save_image
    grid = torchvision.utils.make_grid(img, nrow=5, padding=2, pad_value=255, normalize=False, scale_each=False)
    if norm:
        # print(grid.max(), grid.min())
        ndarr = ((grid * 255 + 0.5).clamp_(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    else:
        ndarr = grid.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    # im.show()
    if Gray:
        im.convert("L").save(imgPath)  # Gray = 0.29900 * R + 0.58700 * G + 0.11400 * B
    else:
        im.save(imgPath)


def generate_square_coordinates(canvas_size, square_size, pattern):
    coordinates = []
    y_offset = (canvas_size[1] - (len(pattern) * square_size)) // (len(pattern) + 1)
    current_y = y_offset

    for row in pattern:
        x_offset = (canvas_size[0] - (row * square_size)) // (row + 1)
        current_x = x_offset

        for _ in range(row):
            coordinates.append((current_x, current_y))
            current_x += square_size + x_offset

        current_y += square_size + y_offset

    return coordinates


def generate_square_coordinates_2(canvas_size, square_size, pattern):
    coordinates = []
    y_offset = (canvas_size[1] - (len(pattern) * square_size)) // (len(pattern) + 1)
    current_y = y_offset

    for row in pattern:
        if row == 0:
            # Skip this row if there are no squares to draw
            current_y += square_size + y_offset
            continue

        x_offset = (canvas_size[0] - (row * square_size)) // (row + 1)
        current_x = x_offset

        for _ in range(row):
            coordinates.append((current_x, current_y))
            current_x += square_size + x_offset

        current_y += square_size + y_offset

    return coordinates


phase_dim = 400
whole_dim = 700

canvas_size = (phase_dim, phase_dim)
square_size = 40
pattern = [3, 4, 3]
square_coordinates = generate_square_coordinates_2(canvas_size, square_size, pattern)
ordered_coordinates = [
    *square_coordinates[0:3],  # First row
    *square_coordinates[3:7],  # Second row
    *square_coordinates[7:10],  # Third row
]

pad = (whole_dim - phase_dim) // 2

# Adding padding to the coordinates
det_x = [coord[1] for coord in ordered_coordinates]
det_y = [coord[0] for coord in ordered_coordinates]
det_y_loc = [coord[0] + pad for coord in ordered_coordinates]
det_x_loc = [coord[1] + pad for coord in ordered_coordinates]


class DDNN(nn.Module):

    def __init__(
        self,
        whole_dim,
        phase_dim,
        pixel_size,
        focal_length,
        wave_lambda,
        scalar=None,
        prop_error=None,
        phase_error=None,
        num_phases=5,
        cfg=None,
    ):
        super(DDNN, self).__init__()
        self.num_phases = num_phases
        self.prop = AngSpecProp(whole_dim, pixel_size, focal_length, wave_lambda, prop_error)
        self.cfg = cfg

        for i in range(1, num_phases + 1):
            setattr(self, f"phase{i}", PhaseMask(whole_dim, phase_dim, error=phase_error))
            # (
            #     setattr(self, f"DMD{i}", DMD(whole_dim, phase_dim))
            #     if i < num_phases
            #     else None
            # )
            setattr(
                self,
                f"unet{i}",
                net2.ComplexUNet(
                    (whole_dim, whole_dim),
                    kernel_size=3,
                    bn_flag=False,
                    CB_layers=[3, 3, 3],
                    FM_num=[4, 8, 16],
                ),
            )
        self.dmd1 = DMD(whole_dim, phase_dim)

        self.scalar = (torch.randn(1, dtype=torch.float32) if scalar is None else torch.tensor(scalar,
                                                                                               dtype=torch.float32))
        self.w_scalar = nn.Parameter(self.scalar)
        self.detector = layers.Detector(
            det_x_loc,
            det_y_loc,
            size=whole_dim,
            det_size=square_size,
            mode="mean",
            intensity_mode=False,
        )
        self.input = Incoherent_Int2Complex()

    def forward(self, input_field, cn_weight=1.0):
        self.in_outs_sim = []
        x = self.input(dorefa_a(input_field, 1))
        for i in range(1, self.num_phases + 1):
            self.in_outs_sim.append(x)
            phase = getattr(self, f"phase{i}")
            x = phase(x)
            x = self.prop(x)
            if self.cfg.train == "bat":
                x = (x + getattr(self, f"unet{i}")((x + getattr(self, f"at_mask_intensity_phy{i}")) / 2) * cn_weight)
                # x = x + getattr(self, f"unet{i}")(x) * cn_weight
            setattr(self, f"at_mask{i}", x)

            if i < self.num_phases:
                # x = getattr(self, f"DMD{i}")(x)
                x = self.dmd1(x)
                setattr(self, f"at_mask_intensity{i}", x.abs())
                pass
            else:
                self.at_sensor = x

                x = x.abs()
                self.at_sensor_intensity = x
                x = self.w_scalar * self.at_sensor_intensity
        return x

    def physical_forward(self, x):
        self.in_outs_phy = []
        with torch.no_grad():
            x = self.input(dorefa_a(x, 1))
            for i in range(1, self.num_phases + 1):
                self.in_outs_phy.append(x)
                phase = getattr(self, f"phase{i}")
                x = phase.physical_forward(x)
                x = self.prop(x)
                setattr(self, f"at_mask_phy{i}", x)

                if i < self.num_phases:
                    # x = getattr(self, f"DMD{i}")(x)
                    x = self.dmd1(x)
                    setattr(self, f"at_mask_intensity_phy{i}", x.abs())
                    pass
                else:
                    setattr(self, f"at_mask_intensity_phy{i}", x.abs())
                    self.at_sensor_phy = x
                    x = x.abs()
                    self.at_sensor_intensity_phy = x
                    x = self.w_scalar * self.at_sensor_intensity_phy
            return x

    def physical_forward_for_train(self, input_field_phy, input_field_sim, iter_num=1):

        x_sim = getattr(self, f"phase{iter_num}")(input_field_phy)
        x_sim = self.prop(x_sim)

        # if self.cfg.phy == "new":
        x_sim = x_sim + getattr(self, f"unet{iter_num}")(
            (x_sim + getattr(self, f"at_mask_intensity_phy{iter_num}")) / 2)
        # x = x + getattr(self, f"unet{iter_num}")(x)

        x_sim = self.dmd1(x_sim) if iter_num < self.num_phases else x_sim

        with torch.no_grad():
            x_phy = getattr(self, f"phase{iter_num}").physical_forward(input_field_phy)
            x_phy = self.prop(x_phy)
            x_phy = self.dmd1(x_phy) if iter_num < self.num_phases else x_phy

        return x_sim, x_phy

    def plot_sim(self, input_field, cn_weight=1.0):
        x = self.input(dorefa_a(input_field, 1))
        for i in range(1, self.num_phases + 1):
            phase = getattr(self, f"phase{i}")
            x = phase(x)
            x = self.prop(x)
            if self.cfg.train == "bat":
                x = (x + getattr(self, f"unet{i}")((x + getattr(self, f"at_mask_intensity_phy{i}")) / 2) * cn_weight)
                # x = x + getattr(self, f"unet{i}")(x) * cn_weight

            # x = getattr(self, f"DMD{i}")(x) if i < self.num_phases else x
            plt.figure()
            plt.imshow(
                x.abs().cpu().detach().numpy().reshape(whole_dim, whole_dim),
                cmap=parula,
            )
            plt.colorbar()
            plt.savefig(f"x_sim{i}_{cn_weight}.png")
            plt.close()
            x = self.dmd1(x) if i < self.num_phases else x
            plt.figure()
            plt.imshow(
                x.abs().cpu().detach().numpy().reshape(whole_dim, whole_dim),
                cmap=parula,
            )
            plt.colorbar()
            plt.savefig(f"x_sim{i}_dmd_{cn_weight}.png")
            plt.close()
        output_amp = self.detector(self.w_scalar * x.abs())
        return output_amp

    def plot_phy(self, input_field):
        x = self.input(dorefa_a(input_field, 1))
        for i in range(1, self.num_phases + 1):
            phase = getattr(self, f"phase{i}")
            x = phase.physical_forward(x)
            x = self.prop(x)
            # plot

            # x = getattr(self, f"DMD{i}")(x) if i < self.num_phases else x
            plt.figure()
            plt.imshow(
                x.abs().cpu().detach().numpy().reshape(whole_dim, whole_dim),
                cmap=parula,
            )
            plt.colorbar()
            plt.savefig(f"x_{i}_phy.png")
            plt.close()
            x = self.dmd1(x) if i < self.num_phases else x
            plt.figure()
            plt.imshow(
                x.abs().cpu().detach().numpy().reshape(whole_dim, whole_dim),
                cmap=parula,
            )
            plt.colorbar()
            plt.savefig(f"x_{i}_phy_dmd.png")
            plt.close()
        output_amp = self.detector(self.w_scalar * x.abs())
        return output_amp

    def phy_replace_sim(self):
        # state fusion
        if self.cfg.fusion == "new":
            with torch.no_grad():
                angle = torch.angle(self.at_sensor)
                angle2 = torch.angle(self.at_sensor_phy)
                amp = self.at_sensor_phy
                amp1 = self.at_sensor_intensity.cuda()

                modulus = (1 - alpha) * torch.abs(amp) + alpha * amp1

                new_data = modulus * torch.exp(1j * angle * beta)
                # new_data = modulus
                self.at_sensor.data.copy_(new_data.data)
                self.at_sensor_intensity.data.copy_(self.at_sensor_intensity_phy.data)
                # self.at_sensor_intensity.data.copy_(new_data.data)

                for i in range(1, self.num_phases):
                    angle = torch.angle(getattr(self, f"at_mask{i}"))
                    angle2 = torch.angle(getattr(self, f"at_mask_phy{i}"))
                    amp = getattr(self, f"at_mask_phy{i}")
                    # new_data = torch.abs(amp) * torch.exp(1j * angle)

                    amp1 = getattr(self, f"at_mask{i}")

                    modulus = (1 - alpha) * torch.abs(amp) + alpha * amp1
                    new_data = modulus * torch.exp(1j * angle * beta)

                    getattr(self, f"at_mask{i}").data.copy_(amp.data)
                    getattr(self, f"at_mask_intensity{i}").data.copy_(getattr(self, f"at_mask_intensity_phy{i}").data)
                    # getattr(self, f"at_mask_intensity{i}").data.copy_(new_data.data)
        elif self.cfg.fusion == "old":
            with torch.no_grad():
                angle = torch.angle(self.at_sensor)
                amp = self.at_sensor_phy
                new_data = torch.abs(amp) * torch.exp(1j * angle)

                self.at_sensor.data.copy_(new_data.data)
                self.at_sensor_intensity.data.copy_(self.at_sensor_intensity_phy.data)

                for i in range(1, self.num_phases):
                    angle = torch.angle(getattr(self, f"at_mask{i}"))
                    amp = getattr(self, f"at_mask{i}")
                    new_data = torch.abs(amp) * torch.exp(1j * angle)

                    getattr(self, f"at_mask{i}").data.copy_(new_data.data)
                    getattr(self, f"at_mask_intensity{i}").data.copy_(getattr(self, f"at_mask_intensity_phy{i}").data)
