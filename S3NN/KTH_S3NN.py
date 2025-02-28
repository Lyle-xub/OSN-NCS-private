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

# Config:
batch_size = 32
epochs = 50
lr = 0.01
device = "cuda"
ssize = 480
pad_size = 640
dec_num = 6
train_path = r"D:/project/home/ubuntu/zf/KTH/hdf5_files/train.hdf5"
dev_path = r"D:/project/home/ubuntu/zf/KTH/hdf5_files/dev.hdf5"
dev_npy_path = "D:/project/home/ubuntu/zf/KTH/hdf5_files/dev.npy"
detail_path = "D:/project/home/ubuntu/zf/train_data/KTH_S3NN.npz"
model_save_path = r"D:/project/home/ubuntu/zf/model/KTH_S3NN.pth"

k = 400 / 400  # Scale factor for detector layer
det_size = 50
det_size_shift = 70
det_size = int(det_size / k)
det_x_loc = [120, 120, 225, 225, 330, 330]
det_y_loc = [142, 307, 142, 307, 142, 307]
det_x_loc = [int((x + det_size_shift) / k) for x in det_x_loc]
det_y_loc = [int((y + det_size_shift) / k) for y in det_y_loc]


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


class donn(nn.Module):
    def __init__(self, det_x_loc, det_y_loc, det_size, size):
        super(donn, self).__init__()
        true_phase = 400
        k = 400 / true_phase
        print(k)
        self.phase1 = PhaseMask(640, 400)
        self.phase2 = PhaseMask(640, 400)
        self.phase3 = PhaseMask(640, 400)
        self.phase4 = PhaseMask(640, 400)
        self.phase5 = PhaseMask(640, 400)
        self.phase6 = PhaseMask(640, 400)
        # self.phase6 = PhaseMask(int(640/k),int(400/k))

        self.prop = AngSpecProp(
            whole_dim=640, pixel_size=12.5e-6, focal_length=0.3, wave_lambda=520e-9
        )
        self.prop1 = AngSpecProp(
            whole_dim=int(640 / k),
            pixel_size=12.5e-6,
            focal_length=0.3,
            wave_lambda=520e-9,
        )

        self.dmd = DMD(640, 400)
        self.input = Incoherent_Int2Complex()
        self.detector = layers.Detector(
            x_loc=det_x_loc, y_loc=det_y_loc, det_size=det_size, size=size
        )
        self.w = nn.Parameter(torch.tensor(1.0)).to(device)
        self.k = k

    def forward(self, input_field):
        x = self.input(input_field)

        x = self.phase1(x)
        x = self.prop(x)
        x = self.dmd(x)

        x = self.phase2(x)
        x = self.prop(x)
        x = self.dmd(x)

        x = self.phase3(x)
        x = self.prop(x)
        x = self.dmd(x)

        x = self.phase4(x)
        x = self.prop(x)
        x = self.dmd(x)

        x = self.phase5(x)
        x = self.prop(x)
        x = self.dmd(x)

        x = self.phase6(x)
        x = self.prop(x)

        x = self.detector(self.w * x)
        # print(f'Output type: {type(x)}, Output shape: {x.shape}, Output: {x}')
        return x


epochs_list = []
frame_accuracy_list = []
video_accuracy_list = []


def train(model, train_dataloader, val_dataloader, epochs, lr):
    criterion = torch.nn.MSELoss(reduction="sum").to(device)
    # criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    global record_frame_accuracy
    record_frame_accuracy = 0.0
    best_epoch = 0
    # record_video_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        train_running_loss = 0.0
        train_running_correct = 0
        total_train_samples = 0

        tk0 = tqdm(train_dataloader, ncols=150, total=int(len(train_dataloader)))
        for train_data_batch in tk0:
            train_images, train_labels = train_data_batch
            train_images = train_images.to(device)
            train_labels = train_labels.to(device)
            train_labels_one_hot = torch.nn.functional.one_hot(
                train_labels, dec_num
            ).float()

            optimizer.zero_grad()
            train_outputs = model(train_images.squeeze(1))
            train_loss = criterion(train_outputs, train_labels_one_hot)

            b = 0.4
            flood = (train_loss - b).abs() + b
            optimizer.zero_grad()
            flood.backward()
            optimizer.step()

            train_running_loss += train_loss.item()
            train_predictions = torch.argmax(train_outputs, dim=1)
            train_running_correct += (train_predictions == train_labels).sum().item()
            total_train_samples += train_labels.size(0)

        train_loss = train_running_loss / total_train_samples
        train_accuracy = train_running_correct / total_train_samples
        tk0.set_description_str("Epoch {}/{} : Training".format(epoch + 1, epochs))
        tk0.set_postfix(
            {
                "Train Loss": "{:.4f}".format(train_loss),
                "Train Accuracy": "{:.4f}".format(train_accuracy),
            }
        )

        # Explicitly print training loss and accuracy at the end of each epoch
        print(
            f"Epoch {epoch + 1}/{epochs} - Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}"
        )

        scheduler.step()

        val_loss, val_accuracy, val_confusion_matrix = eval_model(
            model, val_dataloader, epoch
        )

        # Print validation results
        print(
            "Epoch {}: Validation Loss: {:.4f}, Validation Accuracy: {:.4f}".format(
                epoch + 1, val_loss, val_accuracy
            )
        )
        if val_accuracy > record_frame_accuracy:
            record_frame_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            best_epoch = epoch + 1

        print(
            "The best accuracy is now {:.4f}% at epoch {}".format(
                record_frame_accuracy * 100, best_epoch
            )
        )

        # mapping
        plt.figure(figsize=(8, 6))
        sns.heatmap(val_confusion_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Frame confusion Matrix for Epoch {}".format(epoch + 1))
        # plt.show()


def eval_model(model, val_dataloader, epoch):
    criterion = torch.nn.MSELoss(reduction="sum").to(device)
    model.eval()
    val_labels_all = []
    val_outputs_all = []
    val_running_loss = 0.0
    total_samples = 0
    batch_index = 0
    # val_outputs_video_all = []

    # print(video_array_start)

    with torch.no_grad():

        # get true_labels of video
        data = np.load(dev_npy_path, allow_pickle=True)
        video_names = data[:, 6]
        labels = data[:, 4].astype(int)
        unique_video_names, indices = np.unique(video_names, return_index=True)
        video_name = data[:, 6]
        video_order = data[:, 5]
        video_label = data[:, 4].astype(int)
        unique_video_names, indices = np.unique(video_names, return_index=True)
        unique_labels = labels[indices]
        one_hot_labels = np.zeros((unique_labels.size, dec_num), dtype=int)
        one_hot_labels[np.arange(unique_labels.size), unique_labels] = 1

        def initialize_video_array(video_names):
            dtype = [("name", "U50"), ("vector", "i4", (dec_num,))]
            video_array = np.array(
                [(name, np.zeros(dec_num, dtype="i4")) for name in video_names],
                dtype=dtype,
            )
            return video_array

        video_array = initialize_video_array(unique_video_names)

        for val_data_batch in val_dataloader:
            val_images, val_labels = val_data_batch
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            val_labels_one_hot = torch.nn.functional.one_hot(
                val_labels, dec_num
            ).float()

            val_outputs = model(val_images.squeeze(1))
            val_loss = criterion(val_outputs, val_labels_one_hot)
            val_outputs_all.extend(torch.argmax(val_outputs, dim=1).cpu().numpy())

            val_labels_all.extend(val_labels.cpu().numpy())
            val_running_loss += val_loss.item()
            total_samples += val_labels.size(0)

            row = data[batch_index]
            person, activity, day, sequence_number, label, order, name = row
            val_outputs_see = val_outputs.cpu().numpy().tolist()
            max_index = np.argmax(val_outputs_see)
            val_outputs_one = np.zeros_like(val_outputs_see)
            val_outputs_one.flat[max_index] = 1
            val_outputs_one = val_outputs_one.astype("int32")
            val_outputs_one = val_outputs_one.reshape(-1)

            # matching the video name
            for video in video_array:
                if video["name"] == name:
                    video["vector"] += val_outputs_one
                    break

            batch_index += 1

    for video in video_array:
        if video["name"] == name:
            max_index = np.argmax(video["vector"])
            video["vector"][:] = 0
            video["vector"][max_index] = 1
            break

    predicted_labels = np.array([np.argmax(video["vector"]) for video in video_array])
    true_labels = np.array([np.argmax(label) for label in one_hot_labels])
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    accuracy = accuracy_score(true_labels, predicted_labels)
    print("video accuracy:", accuracy)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Video confusion Matrix for Epoch {}".format(epoch + 1))

    val_loss = val_running_loss / total_samples
    val_accuracy = (np.array(val_labels_all) == np.array(val_outputs_all)).mean()
    val_confusion_matrix = confusion_matrix(val_labels_all, val_outputs_all)

    epochs_list.append(epoch)
    frame_accuracy_list.append(val_accuracy)
    video_accuracy_list.append(accuracy)

    return val_loss, val_accuracy, val_confusion_matrix


# train
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
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    ]
)

train_dataset = HDF5Dataset(file_path=train_path, transform=transform1)
class_counts = train_dataset.get_class_counts()
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
labels = torch.tensor(train_dataset.labels[:], dtype=torch.int64)
sample_weights = class_weights[labels]
sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(sample_weights), replacement=True
)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

val_dataset = HDF5Dataset(file_path=dev_path, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)


print(val_dataset[0][0].shape)
model = donn(det_x_loc, det_y_loc, det_size=det_size, size=pad_size)
print(model)
model.to(device)

start_time = time.time()
train(model, train_dataloader, val_dataloader, epochs, lr)
end_time = time.time()
total_time = end_time - start_time
print(f"Training completed in: {total_time:.0f}s")
np.savez(
    detail_path,
    epochs=epochs_list,
    val_accuracy=frame_accuracy_list,
    accuracy=video_accuracy_list,
)
