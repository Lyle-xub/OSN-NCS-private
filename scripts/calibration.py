import imageio
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import logging
import matplotlib.pyplot as plt

from function.optical_unit import *
from scipy.ndimage import zoom
from PIL import Image
from function.utils import generate_square_coordinates
import function.optical_network as on
import numpy as np
import cv2


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((250, 250)),
        # transforms.Pad([150, 150], fill=(0), padding_mode='constant')
    ]
)
val_dataset = torchvision.datasets.MNIST(
    "./data/", train=False, transform=transform, download=True
)

# save validation dataset
for i in range(10):
    val_img, val_label = val_dataset[i]
    val_img = val_img.squeeze().numpy()
    val_img = (val_img * 255).astype(np.uint8)
    cv2.imwrite(f"{i}.bmp", val_img)


def pad_label(label, whole_dim, phase_dim, detx=None, dety=None, size=30):

    batch_size = label.shape[0]

    padded_labels = torch.zeros(batch_size, phase_dim, phase_dim, device=label.device)

    for i in range(batch_size):
        _, index = torch.max(label[i], dim=0)
        x_start = max(0, min(detx[int(index.cpu().numpy())], phase_dim - size))
        y_start = max(0, min(dety[int(index.cpu().numpy())], phase_dim - size))
        padded_labels[i, x_start : x_start + size, y_start : y_start + size] = 1

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


class cropped_loss(nn.Module):
    def __init__(self, loss_slice):
        super(cropped_loss, self).__init__()
        self.loss_slice = loss_slice

    def forward(self, output, target):
        # print(self.loss_slice)
        diff = (output - target)[:, self.loss_slice, self.loss_slice]
        return torch.mean(torch.abs(diff) ** 2)


class DDNN(nn.Module):
    def __init__(self, whole_dim, phase_dim, pixel_size, focal_length, wave_lambda):
        super(DDNN, self).__init__()

        self.prop = AngSpecProp(whole_dim, pixel_size, focal_length, wave_lambda)
        # self.lens = Lens(whole_dim, pixel_size, focal_length=10e-2, wave_lambda=wave_lambda)
        # phase = cv2.imread('D:\project\control\\1951usaf_test_target.jpg').transpose(2,0,1)[0] / 255
        # phase = cv2.resize(phase, (phase_dim,phase_dim))
        self.phase1 = PhaseMask(whole_dim, phase_dim)
        self.input = Incoherent_Int2Complex()

    def forward(self, input_field):
        # x =
        x = self.input(input_field)
        # x = self.prop(x)
        x = self.phase1(x)

        out = self.prop(x)

        return out


class Trainer:
    model = DDNN(250, 250, 12.5e-6, 30e-2, 532e-9)
    device = "cuda"
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_func = cropped_loss(slice(0, 250))

    def train(
        self,
        callback=None,
        update_progress=None,
        update_val_accuracy=None,
        update_images=None,
    ):

        import logging

        best_test_acc = 0
        for epoch in range(100):
            self._train_epoch(epoch, callback)

            torch.save(
                self.model.state_dict(),
                f"img.pth",
            )
        logging.info("Finished Training")

    def _train_epoch(self, epoch, callback=None):
        from lightridge import layers

        self.model.train()
        running_loss = 0.0
        correct, total = 0, 0
        labels = cv2.imread("D:\project\control\image.png").transpose(2, 0, 1)[0] / 255
        labels = cv2.resize(labels, (250, 250))
        # labels = torch.sigmoid(torch.tensor(labels)).to(self.device)

        # labels = cv2.imread('D:\project\control\\images.png').transpose(2,0,1)[0] / 255

        # labels = cv2.resize(labels, (250,250))
        pad_labels = torch.tensor(labels).to(self.device)
        # pad_labels = dorefa_a(pad_labels, 1)
        # pad_labels = F.pad(pad_labels, (200,200,200,200), 'constant', 0)
        for ii in range(100):
            # using one-hot label
            # print(data[1].shape)
            data = torch.ones(1, 1, 250, 250).to(self.device)
            # labels = torch.randn(1,10).to(self.device)

            # labels = cv2.imread('D:\project\control\\1951usaf_test_target.jpg').transpose(2,0,1)[0] / 255

            # print(pad_labels.shape)
            # to tensor

            # pad_labels = pad_label(labels, 500,250,
            #                         [200]*10,[200]*10, 50)

            inputs = data[0].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs.squeeze(1))
            loss = self.loss_func(outputs, pad_labels)
            print(loss)

            loss.backward()
            self.optimizer.step()


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


labels = cv2.imread("D:\project\control\image.png").transpose(2, 0, 1)[0] / 255
labels = cv2.resize(labels, (250, 250))
# labels = sigmoid(torch.tensor(labels))
labels = torch.tensor(labels)
labels = dorefa_a(labels, 1)
plt.imshow(labels)
plt.colorbar()  # 显示颜色条
plt.show()

train = Trainer()
print(train.model.phase1.w_p.mean())
train.train()

x = torch.ones(1, 1, 250, 250)
model = DDNN(250, 250, 12.5e-6, 30e-2, 532e-9)
ck = torch.load("img.pth")
model.load_state_dict(ck)
model.eval()
output = model(x.squeeze(1))

# read
print(len(torch.unique(model.phase1.w_p)))
# with torch.no_grad():
#     new_w_p = torch.where(model.phase1.w_p.data > 0, torch.tensor(0.0, device=model.phase1.w_p.device), model.phase1.w_p)
#     model.phase1.w_p = torch.nn.Parameter(new_w_p)
# max = model.phase1.w_p.min()

# with torch.no_grad():
#     new_w_p = torch.where(model.phase1.w_p.data > 0, max, model.phase1.w_p)
#     model.phase1.w_p = torch.nn.Parameter(new_w_p)

phase = dorefa_w(model.phase1.w_p, 8).cpu().detach().numpy() * math.pi * 1.999
# phase1 = phase
# phase2 = phase
# phase1[phase<0] = 0
# phase[phase<0] = 0

# phase = cv2.imread('D:\project\control\Vortex100_CenterX512_CenterY512.bmp').transpose(2,0,1)[0]
print(phase)
# print(np.unique(phase))
print(len(np.unique(phase)))


# 映射到256
mapped_arr = np.vectorize(on.DDNN.map_values_to_int(phase).get)(phase)
print(mapped_arr)
phase = mapped_arr.squeeze(0)

# phase = cv2.imread('D:\project\SLM_phase_pattern\\test\\bigqq.bmp')
#

plt.imshow(phase, interpolation="nearest")
plt.colorbar()  # 显示颜色条
plt.title("Weight Matrix Distribution")
plt.show()

flattened_weights = phase.flatten()

# 绘制柱状图
plt.hist(flattened_weights, bins=256, color="blue", edgecolor="black")
plt.title("Distribution of Weights")
plt.xlabel("Weight Value")
plt.ylabel("Frequency")
plt.show()


phase = np.flipud(phase)
cv2.imwrite("img.bmp", phase.astype(np.uint8))


from interface.imageInterface import crop_center, superimpose_images, pad_image

cropped_image1 = crop_center(Image.open("img.bmp"), 250)
pad_image(cropped_image1).save("pad_img.bmp")
cropped_image2 = crop_center(
    Image.open("D:\project\control\img/reconstruction/BlazedGrating_Period2.bmp"), 250
)
superimposed_image = superimpose_images(cropped_image1, cropped_image2)
superimposed_image_pil = Image.fromarray(superimposed_image)
padded_image = pad_image(superimposed_image_pil)
padded_image.save("img.bmp")
plt.imshow(output.detach().squeeze().abs().cpu().numpy())
plt.show()
