import lightridge
import lightridge.layers as layers
import lightridge.utils as utils
import lightridge.data as dataset
from lightridge.get_h import _field_Fresnel

import os
import csv
import h5py
from time import time
import random
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle

device = "cuda:0"

import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR

from PIL import Image

from platform import python_version

print("Python version", python_version())
print("Pytorch - version", torch.__version__)
print("Pytorch - cuDNN version :", torch.backends.cudnn.version())

import math
from torch.fft import fftshift, fft2, ifft2, ifftshift
from torch.autograd import Function

# from utils import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import LinearSegmentedColormap


import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler
import time

import h5py
from PIL import Image
import os


# Config
batch_size = 64
device = "cuda:0"
epochs = 50
lr = 0.01
ssize = 300
pad_size = 400

# switching different path_1~4
# b-hc-hw
path_1 = "/home/ubuntu/zf/KTH/hdf5_files/b-hc_dev.npy"
path_2 = "/home/ubuntu/zf/KTH/hdf5_files/b-hc_output.npy"
path_3 = "/home/ubuntu/zf/model/b-hc_S4NN.pth"
path_4 = r"/home/ubuntu/zf/KTH/hdf5_files/b-hc-hw_dev.hdf5"

# path_1 = "/home/ubuntu/zf/KTH/hdf5_files/b-hw_dev.npy"
# path_2 = "/home/ubuntu/zf/KTH/hdf5_files/b-hw_output.npy"
# path_3 = "/home/ubuntu/zf/model/b-hw_S4NN.pth"
# path_4 = r"/home/ubuntu/zf/KTH/hdf5_files/b-hc-hw_dev.hdf5"

# path_1 = "/home/ubuntu/zf/KTH/hdf5_files/hc-hw_dev.npy"
# path_2 = "/home/ubuntu/zf/KTH/hdf5_files/hc-hw_output.npy"
# path_3 = "/home/ubuntu/zf/model/hc-hw_S4NN.pth"
# path_4 = r"/home/ubuntu/zf/KTH/hdf5_files/b-hc-hw_dev.hdf5"

# j-r-w
# path_1 = 'home/ubuntu/zf/KTH/hdf5_files/j-r_dev.npy'
# path_2 = 'home/ubuntu/zf/KTH/hdf5_files/j-r_output.npy'
# path_3 = 'home/ubuntu/zf/model/j-r_S4NN.pth'
# path_4 = r'home/ubuntu/zf/KTH/hdf5_files/j-r-w_dev.hdf5'

# path_1 = 'home/ubuntu/zf/KTH/hdf5_files/j-w_dev.npy'
# path_2 = 'home/ubuntu/zf/KTH/hdf5_files/j-w_output.npy'
# path_3 = 'home/ubuntu/zf/model/j-w_S4NN.pth'
# path_4 = r'home/ubuntu/zf/KTH/hdf5_files/j-r-w_dev.hdf5'

# path_1 = 'home/ubuntu/zf/KTH/hdf5_files/r-w_dev.npy'
# path_2 = 'home/ubuntu/zf/KTH/hdf5_files/r-w_output.npy'
# path_3 = 'home/ubuntu/zf/model/r-w_S4NN.pth'
# path_4 = r'home/ubuntu/zf/KTH/hdf5_files/j-r-w_dev.hdf5'


class Lens(nn.Module):
    def __init__(self, whole_dim, pixel_size, focal_length, wave_lambda):
        super(Lens, self).__init__()
        # basic parameters
        temp = np.arange(
            (-np.ceil((whole_dim - 1) / 2)), np.floor((whole_dim - 1) / 2) + 0.5
        )
        x = temp * pixel_size
        xx, yy = np.meshgrid(x, x)
        lens_function = np.exp(
            -1j * math.pi / wave_lambda / focal_length * (xx**2 + yy**2)
        )
        self.lens_function = torch.tensor(lens_function, dtype=torch.complex64).to(
            device
        )

    def forward(self, input_field):
        out = torch.mul(input_field, self.lens_function)
        return out


class AngSpecProp(nn.Module):
    def __init__(self, whole_dim, pixel_size, focal_length, wave_lambda):
        super(AngSpecProp, self).__init__()
        k = 2 * math.pi / wave_lambda  # optical wavevector
        df1 = 1 / (whole_dim * pixel_size)
        f = (
            np.arange(
                (-np.ceil((whole_dim - 1) / 2)), np.floor((whole_dim - 1) / 2) + 0.5
            )
            * df1
        )
        fxx, fyy = np.meshgrid(f, f)
        fsq = fxx**2 + fyy**2

        self.Q2 = torch.tensor(
            np.exp(-1j * (math.pi**2) * 2 * focal_length / k * fsq),
            dtype=torch.complex64,
        ).to(device)
        self.pixel_size = pixel_size
        self.df1 = df1

    def ft2(self, g, delta):
        return fftshift(fft2(ifftshift(g))) * (delta**2)

    def ift2(self, G, delta_f):
        N = G.shape[1]
        return ifftshift(ifft2(fftshift(G))) * ((N * delta_f) ** 2)

    def forward(self, input_field):
        # compute the propagated field
        # print(f'Ang in type: {type(input_field)}')
        Uout = self.ift2(self.Q2 * self.ft2(input_field, self.pixel_size), self.df1)
        # print(f'Ang in type: {type(input_field)}, Output shape: {Uout.shape}')
        return Uout


class ScaleSigner(Function):
    """take a real value x, output sign(x)*E(|x|)"""

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input) * torch.mean(torch.abs(input))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def scale_sign(input):
    return ScaleSigner.apply(input)


class Quantizer(Function):
    @staticmethod
    def forward(ctx, input, nbit):
        scale = 2**nbit - 1
        return torch.round(input * scale) / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def quantize(input, nbit):
    return Quantizer.apply(input, nbit)


def dorefa_w(w, nbit_w):
    if nbit_w == 1:
        w = scale_sign(w)
    else:
        #       weight = weight / 2 / max_w + 0.5
        #   weight_q = max_w * (2 * self.uniform_q(weight) - 1)
        w = torch.tanh(w)
        max_w = torch.max(torch.abs(w)).detach()
        w = w / 2 / max_w + 0.5
        w = 1.999 * quantize(w, nbit_w) - 1
    return w


def dorefa_a(input, nbit_a):
    # print(torch.clamp(0.1 * input, 0, 1))
    return quantize(torch.clamp(input, 0, 1), nbit_a)


print(dorefa_w(torch.tensor([0.1, 0.2, 0.3, 0, -1]), 1))


class DMD(nn.Module):
    def __init__(self, whole_dim, phase_dim):
        super().__init__()
        self.whole_dim = whole_dim
        self.phase_dim = phase_dim
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(10.0), requires_grad=False)
        self.trans = Incoherent_Int2Complex()
        self.sensor = Sensor()
        self.mask = self.create_mask(whole_dim, phase_dim)

    def create_mask(self, whole_dim, phase_dim):
        pad_size = (whole_dim - phase_dim) // 2
        mask = torch.zeros((whole_dim, whole_dim))
        mask[pad_size : pad_size + phase_dim, pad_size : pad_size + phase_dim] = 1
        return mask

    def forward(self, x, insitu=False):
        # print(x.shape)
        if not insitu:
            modulus_squared = self.sensor(x)
            # print(modulus_squared.max(), modulus_squared.min())
            # modulus_squared = self.conv(modulus_squared.unsqueeze(1)).squeeze(1)
            modulus_squared = dorefa_a(modulus_squared, 8)
        else:
            # x = x **2
            # x = torch.tanh(x)
            modulus_squared = x
        # print(modulus_squared.device)
        # modulus_squared = self.ln(modulus_squared)
        mask = self.mask.to(x.device)
        modulus_squared = modulus_squared * mask
        # print(modulus_squared.max(), modulus_squared.min())

        I_th = torch.mean(modulus_squared, dim=(-2, -1), keepdim=True)
        # print(I_th.max(), I_th.min())
        x = torch.sigmoid(self.beta * (modulus_squared - self.alpha * I_th))
        # print(self.beta, self.alpha)
        # print(x.max(), x.min())
        y = dorefa_a(x, 1)
        # print(y.max(), y.min())

        x = self.trans(y)

        x_real = x.real * mask
        x_imag = x.imag * mask
        x = torch.complex(x_real, x_imag)

        return x


class PhaseMask(nn.Module):
    def __init__(self, whole_dim, phase_dim, phase=None):
        super(PhaseMask, self).__init__()
        self.whole_dim = whole_dim
        phase = (
            torch.randn(1, phase_dim, phase_dim, dtype=torch.float32)
            if phase is None
            else torch.tensor(phase, dtype=torch.float32)
        )
        self.w_p = nn.Parameter(phase)
        pad_size = (whole_dim - phase_dim) // 2
        self.paddings = (pad_size, pad_size, pad_size, pad_size)
        self.init_weights()

    # kaiming init
    def init_weights(self):
        nn.init.kaiming_uniform_(self.w_p, a=math.sqrt(5))
        # torch.nn.init.normal_(self.w_p, mean=0.5, std=1)
        # nn.init.kaiming_normal_(self.w_p, a=math.sqrt(5))

    def forward(self, input_field):
        # with torch.no_grad():
        #     new_w_p = torch.where(self.w_p.data > 0, torch.tensor(0.0, device=self.w_p.device), self.w_p)
        #     self.w_p = torch.nn.Parameter(new_w_p)
        # mask_phase = dorefa_w(self.w_p,8) * math.pi * 1.999
        mask_phase = (dorefa_w(self.w_p, 8)) * math.pi
        # print('[info]',mask_phase.max())
        # print('[info]',mask_phase.min())
        # mask_phase = self.w_p *  math.pi * 1.999
        mask_whole = F.pad(
            torch.complex(torch.cos(mask_phase), torch.sin(mask_phase)), self.paddings
        ).to(device)
        output_field = torch.mul(input_field, mask_whole)
        # print(f'Phase Output type: {type(output_field)}, Output shape: {output_field.shape}')
        return output_field


class NonLinear_Int2Phase_for_DMD(nn.Module):
    def __init__(self):
        super(NonLinear_Int2Phase_for_DMD, self).__init__()

    def forward(self, input_field):
        phase = input_field * 1.999 * math.pi
        print(phase)
        phase = torch.complex(torch.cos(phase), torch.sin(phase)).to(device)
        return phase


class NonLinear_Int2Phase(nn.Module):
    def __init__(self):
        super(NonLinear_Int2Phase, self).__init__()

    def forward(self, input_field):
        phase = input_field * 1.999 * math.pi
        phase = torch.complex(torch.cos(phase), torch.sin(phase)).to(device)
        return phase


class Incoherent_Int2Complex(nn.Module):
    def __init__(self):
        super(Incoherent_Int2Complex, self).__init__()

    def forward(self, input_field):
        x = torch.complex(
            input_field, torch.zeros(input_field.shape, device=input_field.device)
        ).to(device)
        return x


class Sensor(nn.Module):
    def __init__(self):
        super(Sensor, self).__init__()

    def forward(self, input_field):
        x = torch.square(torch.real(input_field)) + torch.square(
            torch.imag(input_field)
        )
        return torch.tanh(x)


# write a test to sensor
def test_sensor():
    sensor = Sensor()
    x = torch.randn(1, 100, 100)
    x = dorefa_w(x, 8)
    print(torch.unique(x))


class HDF5Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        """
        Args:
            file_path (string): Path to the HDF5 file with images and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.file_path = file_path
        self.file = h5py.File(self.file_path, "r")
        self.images = self.file["images"]
        self.labels = self.file["labels"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Access the image and label from the HDF5 dataset
        image = self.images[idx]
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.int64)

        # Convert the numpy array to a PIL Image
        image = Image.fromarray(image.astype("uint8")).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_counts(self):
        # Calculate the number of instances of each class
        _, counts = np.unique(self.labels[:], return_counts=True)
        return counts

    def close(self):
        self.file.close()


class Detector(torch.nn.Module):

    def __init__(
        self,
        det_size_x=125,
        det_size_y=250,
        size=400,
        activation=torch.nn.Softmax(dim=-1),
        intensity_mode=False,
        mode="mean",
    ):
        super(Detector, self).__init__()
        self.size = size
        self.det_size = det_size_x
        self.activation = activation
        self.intensity_mode = intensity_mode
        self.mode = mode

        # Calculating the coordinates for the top and bottom detectors
        center_x = (size - det_size_y) // 2
        pad = 75
        top_y = 0 + pad
        bottom_y = size - det_size_x - pad

        self.x_loc = [center_x, center_x]  # Both detectors are horizontally centered
        self.y_loc = [top_y, bottom_y]  # One at the top, one at the bottom

    def forward(self, x):
        if self.intensity_mode:
            x = x.abs() ** 2  # intensity mode
        else:
            x = x.abs()

        detectors = []

        if self.mode == "mean":
            for i in range(2):  # Only two detectors
                region = x[
                    :,
                    self.x_loc[i] : self.x_loc[i] + self.det_size,
                    self.y_loc[i] : self.y_loc[i] + self.det_size,
                ]
                detectors.append(region.mean(dim=(1, 2)).unsqueeze(-1))

        elif self.mode == "sum":
            for i in range(2):  # Only two detectors
                region = x[
                    :,
                    self.x_loc[i] : self.x_loc[i] + self.det_size,
                    self.y_loc[i] : self.y_loc[i] + self.det_size,
                ]
                detectors.append(region.sum(dim=(1, 2)).unsqueeze(-1))

        detectors = (
            torch.cat(detectors, dim=-1)
            if detectors
            else torch.zeros(x.size(0), 0, device=x.device)
        )

        if self.activation is None:
            return detectors
        else:
            return self.activation(detectors)


def visualize_detector_positions(size, det_size_x, det_size_y, x_loc, y_loc):
    fig, ax = plt.subplots(1)
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)

    # Draw the detector regions as rectangles
    for i in range(len(x_loc)):
        rect = patches.Rectangle(
            (x_loc[i], y_loc[i] + 10),
            det_size_x,
            det_size_y,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        # Label the detectors
        ax.text(
            x_loc[i] + det_size_x / 2,
            y_loc[i] + det_size_y / 2,
            f"Detector {i+1}",
            horizontalalignment="center",
            verticalalignment="center",
            color="white",
            fontsize=12,
        )

    ax.set_aspect("equal", "box")
    ax.set_title("Detector Positions")
    ax.invert_yaxis()  # Invert Y axis to match image coordinate system
    plt.gca().set_facecolor("black")
    plt.show()


class donn(nn.Module):
    def __init__(self):
        super(donn, self).__init__()
        self.phase1 = PhaseMask(400, 250)
        self.phase2 = PhaseMask(400, 250)
        self.prop = AngSpecProp(
            whole_dim=400, pixel_size=12.5e-6, focal_length=0.3, wave_lambda=532e-9
        )
        self.dmd = DMD(400, 250)
        self.input = Incoherent_Int2Complex()
        self.detector = Detector()
        self.w = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_field):
        x = self.input(input_field)
        # x = self.prop(x)
        x = self.phase1(x)
        x = self.prop(x)

        # Enabled when distinguishing jogging, running and walking
        # x = self.dmd(x)
        # x = self.prop(x)
        # x = self.phase2(x)
        # x = self.prop(x)

        x = self.detector(self.w * x)
        # print(f'Output type: {type(x)}, Output shape: {x.shape}, Output: {x}')
        return x


def eval_model(model, val_dataloader, epoch):
    criterion = torch.nn.MSELoss(reduction="sum").to(device)
    model.eval()
    val_labels_all = []
    val_outputs_all = []
    with torch.no_grad():
        data = np.load(path_1, allow_pickle=True)
        video_names = data[:, 6]
        labels = data[:, 4].astype(int)
        unique_video_names, indices = np.unique(video_names, return_index=True)

        video_name = data[:, 6]
        video_order = data[:, 5]
        video_label = data[:, 4].astype(int)
        unique_video_names, indices = np.unique(video_names, return_index=True)
        unique_labels = labels[indices]
        one_hot_labels = np.zeros((unique_labels.size, 2), dtype=int)
        one_hot_labels[np.arange(unique_labels.size), unique_labels] = 1

        def initialize_video_array(video_names):
            dtype = [("name", "U50"), ("vector", "i4", (2,))]
            video_array = np.array(
                [(name, np.zeros(2, dtype="i4")) for name in video_names], dtype=dtype
            )
            return video_array

        video_array = initialize_video_array(unique_video_names)

        for val_data_batch in val_dataloader:
            val_images, val_labels = val_data_batch
            val_images = val_images.to(device)
            val_outputs = model(val_images.squeeze(1))
            val_outputs_all.extend(torch.argmax(val_outputs, dim=1).cpu().numpy())

            val_labels_all.extend(val_labels.cpu().numpy())

            val_outputs_see = val_outputs.cpu().numpy().tolist()
            max_index = np.argmax(val_outputs_see)
            val_outputs_one = np.zeros_like(val_outputs_see)
            val_outputs_one.flat[max_index] = 1
            val_outputs_one = val_outputs_one.astype("int32")
            val_outputs_one = val_outputs_one.reshape(-1)

        print(val_outputs_all)
        np.save(path_2, val_outputs_all)


transform = transforms.Compose(
    [
        transforms.Resize((ssize, ssize)),
        transforms.Pad(
            (
                (pad_size - ssize) // 2,
                (pad_size - ssize) // 2,
                (pad_size - ssize) - (pad_size - ssize) // 2,
                (pad_size - ssize) - (pad_size - ssize) // 2,
            )
        ),
        transforms.ToTensor(),
    ]
)

transform1 = transforms.Compose(
    [
        transforms.Resize((ssize, ssize)),
        transforms.Pad(
            (
                (pad_size - ssize) // 2,
                (pad_size - ssize) // 2,
                (pad_size - ssize) - (pad_size - ssize) // 2,
                (pad_size - ssize) - (pad_size - ssize) // 2,
            )
        ),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
)


val_dataset = HDF5Dataset(file_path=path_4, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

print(val_dataset[0][0].shape)

if __name__ == "__main__":
    model = donn()
    state_dict = torch.load(path_3, map_location=torch.device("cuda:0"))
    model.load_state_dict(state_dict, strict=False)

    model.to(device)


eval_model(model, val_dataloader, epoch=0)
