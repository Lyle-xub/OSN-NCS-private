import imageio
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import logging
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from PIL import Image
from system.utils import *
from system.optical_unit import *
import numpy as np
import cv2


def pad_label(
    label,
    whole_dim,
    phase_dim,
):
    padded_labels = F.pad(
        label,
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


class cropped_loss(nn.Module):

    def __init__(self, loss_slice):
        super(cropped_loss, self).__init__()
        self.loss_slice = loss_slice

    def forward(self, output, target):
        # print(self.loss_slice)
        diff = (output - target)[:, self.loss_slice, self.loss_slice]
        return torch.mean(torch.abs(diff)**2)


class DDNN(nn.Module):

    def __init__(self, whole_dim, phase_dim, pixel_size, focal_length, wave_lambda):
        super(DDNN, self).__init__()

        self.prop = AngSpecProp(whole_dim, pixel_size, focal_length, wave_lambda)
        self.phase1 = PhaseMask(whole_dim, phase_dim)

        self.input = Incoherent_Int2Complex()

    def forward(self, input_field):
        # x =
        x = self.input(input_field)
        x = self.phase1(x)
        out = self.prop(x)

        return out


class Trainer:
    model = DDNN(250, 250, 12.5e-6, 30e-2, 520e-9)
    device = "cuda"
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_func = cropped_loss(slice(0, 250))

    def train(self,):
        import logging
        for epoch in range(100):
            self._train_epoch()
            torch.save(
                self.model.state_dict(),
                f"img.pth",
            )
        logging.info("Finished Training")

    def _train_epoch(self,):

        self.model.train()
        labels = cv2.imread("image.png").transpose(2, 0, 1)[0] / 255
        labels = cv2.resize(labels, (250, 250))
        pad_labels = pad_label(torch.tensor(labels).unsqueeze(0), 250, 250).to('cuda')
        for ii in range(100):
            data = torch.ones(1, 1, 250, 250).to(self.device)
            inputs = data[0].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs.squeeze(1))
            loss = self.loss_func(outputs, pad_labels)
            print(loss)

            loss.backward()
            self.optimizer.step()


def map_values_to_int(array, custom_order=None, start=0, step=1, max_value=202):
    unique_values = np.unique(array)
    if custom_order is None:
        custom_order = unique_values
    else:
        # 确保 custom_order 包含所有 unique_values
        if not set(unique_values).issubset(set(custom_order)):
            raise ValueError("custom_order must contain all unique values from the array.")

        # 只保留在unique_values中的元素
        custom_order = [val for val in custom_order if val in unique_values]

    mapping = {}
    current_value = start
    for value in custom_order:
        mapping[value] = current_value % (max_value + 1)
        current_value += step

    return mapping


labels = cv2.imread("test_target.jpg").transpose(2, 0, 1)[0] / 255
labels = cv2.resize(labels, (250, 250))
labels = torch.tensor(labels)
labels = dorefa_a(labels, 1)
plt.imshow(labels)
plt.colorbar()  # 显示颜色条
plt.show()

train = Trainer()
train.train()

x = torch.ones(1, 1, 250, 250)
model = DDNN(250, 250, 12.5e-6, 30e-2, 520e-9)
model.load_state_dict(torch.load("img.pth"))
output = model(x.squeeze(1))

phase = dorefa_w(model.phase1.w_p, 8).cpu().detach().numpy() * math.pi * 1.999
print(phase)

print(len(np.unique(phase)))

unique_values, indices = np.unique(phase, return_inverse=True)
mapped_arr_unique = indices.reshape(phase.shape)
phase = mapped_arr_unique.squeeze(0)

plt.imshow(phase, interpolation="nearest")
plt.colorbar()
plt.title("Weight Matrix Distribution")
plt.show()

flattened_weights = phase.flatten()
plt.hist(flattened_weights, bins=256, color="blue", edgecolor="black")
plt.title("Distribution of Weights")
plt.xlabel("Weight Value")
plt.ylabel("Frequency")
plt.show()

phase = np.flipud(phase)
cv2.imwrite("img.bmp", phase.astype(np.uint8))

from utils import crop_center, superimpose_images, pad_image

cropped_image1 = crop_center(Image.open("img.bmp"), 250)
# pad_image(cropped_image1).save("pad_img.bmp")
cropped_image2 = crop_center(Image.open("BlazedGrating_Period2.bmp"), 250)
superimposed_image = superimpose_images(cropped_image1, cropped_image2)
superimposed_image_pil = Image.fromarray(superimposed_image)
padded_image = pad_image(superimposed_image_pil)
padded_image.save("img250_30.bmp")
plt.imshow(output.detach().squeeze().abs().cpu().numpy())
plt.show()
