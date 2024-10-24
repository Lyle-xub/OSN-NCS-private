import imageio
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import logging
import matplotlib.pyplot as plt
from .optical_network import DDNN

from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QDesktopServices, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QSize, QThread
from PyQt5.QtCore import pyqtSignal

from .optical_unit import dorefa_w, dorefa_a
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image
import cv2

import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import zoom
import time
import os
from tensorboardX import SummaryWriter
import copy


GlobalHydra.instance().clear()
initialize(config_path="../config", version_base="1.2")
cfg = compose(config_name="main")


def correct_(energe, label):
    corr = (energe.argmax(dim=-1) == label.argmax(dim=-1)).sum().item()
    corr /= energe.size(0)
    return corr


def get_physical_image(n):
    file_path1 = f"D:\project\control\img\infer-30-new\{n}_ori.png"
    physical_img = Image.open(file_path1)
    physical_img = np.array(physical_img).astype(np.uint8) / 255.0
    physical_img = torch.tensor(physical_img).unsqueeze(0).unsqueeze(0)
    roi = {"x": 370, "y": 85, "width": 530, "height": 530}
    roi_coords = (85, 615, 370, 900)
    y1, y2, x1, x2 = roi_coords
    roi = physical_img[:, :, y1:y2, x1:x2]
    physical_img = F.interpolate(roi, (400, 400), mode="nearest")
    # pad to 700*700
    physical_img = F.pad(physical_img, (150, 150, 150, 150), "constant", 0).cuda()
    return physical_img


def pad_label(label, whole_dim, phase_dim, detx=None, dety=None, size=30):

    batch_size = label.shape[0]

    padded_labels = torch.zeros(batch_size, phase_dim, phase_dim, device=label.device)

    for i in range(batch_size):
        _, index = torch.max(label[i], dim=0)
        x_start = max(0, min(detx[int(index.cpu().numpy())], phase_dim - size))
        y_start = max(0, min(dety[int(index.cpu().numpy())], phase_dim - size))
        padded_labels[i, x_start : x_start + size, y_start : y_start + size] = 1.5

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


class OpticalNetworkTrainer:
    def __init__(
        self,
        data_path="./data",
        checkpoint_path=None,
        learning_rate=0.01,
        batch_size=20,
        epochs=200,
        insitu_train=False,
    ):
        self.data_path = data_path
        self.checkpoint_path = checkpoint_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.subbatch_size = 1
        self.subepoch = 2
        self.epochs = epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.insitu_train = insitu_train

        self._prepare_log(insitu_train=insitu_train)
        self._prepare_data()
        self._initialize_model()
        self._setup_logging()

    def _prepare_log(self, insitu_train=False):
        self.datatime = time.strftime("%H-%M-%S", time.localtime())
        self.date = time.strftime("%m-%d", time.localtime())
        self.log_dir = f"log/{self.date}"
        if cfg.reconstruction.wl == 532e-9:
            self.log_dir = f"log/{self.date}/green/{self.datatime}"
        elif cfg.reconstruction.wl == 632e-9:
            self.log_dir = f"log/{self.date}/red/{self.datatime}"
        self.ck_dir = f"{self.log_dir}/checkpoint/{cfg.reconstruction.phase_dim}-{cfg.reconstruction.square_size}-{cfg.reconstruction.distance}"

        if not insitu_train:
            if not os.path.exists(self.log_dir) or not os.path.exists(self.ck_dir):
                os.makedirs(self.log_dir, exist_ok=True)
                os.makedirs(self.ck_dir, exist_ok=True)

        else:
            self.log_dir = f"log/{self.date}/insituTrain/{self.datatime}"
            self.ck_dir = f"log/{self.date}/insituTrain/{self.datatime}/checkpoint"
            if not os.path.exists(self.ck_dir):
                os.makedirs(self.ck_dir, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir)

        cfg.reconstruction.log_dir = self.log_dir
        cfg.reconstruction.ck_dir = self.ck_dir

        cfg.reconstruct_phase = [
            f"{self.log_dir}/Phase1.bmp",
            f"{self.log_dir}/Phase2.bmp",
            f"{self.log_dir}/Phase3.bmp",
        ]
        cfg.reconstruct_output = [
            f"{self.log_dir}/Output1.png",
            f"{self.log_dir}/Output2.png",
            f"{self.log_dir}/Output3.png",
        ]

        self.Image_Path = OmegaConf.to_container(
            cfg.reconstruct_phase + cfg.reconstruct_output, resolve=True
        )  # 转换为普通列表

    def _prepare_data(self):
        subset_size = 2000
        pad = (cfg.reconstruction.whole_dim - cfg.reconstruction.phase_dim) // 2
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.RandomRotation(10),
                transforms.Resize(
                    (cfg.reconstruction.phase_dim, cfg.reconstruction.phase_dim)
                ),
                transforms.Pad([pad, pad], fill=(0), padding_mode="constant"),
            ]
        )
        dev_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (cfg.reconstruction.phase_dim, cfg.reconstruction.phase_dim)
                ),
                transforms.Pad([pad, pad], fill=(0), padding_mode="constant"),
            ]
        )
        self.train_dataset = torchvision.datasets.MNIST(
            self.data_path, train=True, transform=train_transform, download=True
        )
        self.val_dataset = torchvision.datasets.MNIST(
            self.data_path, train=False, transform=dev_transform, download=True
        )
        self.trainloader = DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.testloader = DataLoader(
            dataset=self.val_dataset, batch_size=1, shuffle=False
        )

        train_indices = torch.randperm(len(self.train_dataset))[:subset_size]
        val_indices = torch.randperm(len(self.val_dataset))[:subset_size]

        train_dataset = Subset(self.train_dataset, train_indices)
        val_dataset = Subset(self.val_dataset, val_indices)

        # 创建数据加载器
        self.subtrainloader = DataLoader(
            dataset=train_dataset, batch_size=self.subbatch_size, shuffle=True
        )
        self.subtestloader = DataLoader(
            dataset=val_dataset, batch_size=self.subbatch_size, shuffle=False
        )

    def _initialize_model(self):
        self.model = DDNN(
            cfg.reconstruction.whole_dim,
            cfg.reconstruction.phase_dim,
            cfg.reconstruction.pixel_size,
            cfg.reconstruction.distance,
            cfg.reconstruction.wl,
            cfg=cfg,
        )
        self.model.to(self.device)
        # self.loss_func = nn.CrossEntropyLoss()
        loss_slice = slice(
            cfg.reconstruction.whole_dim // 2 - cfg.reconstruction.phase_dim // 2,
            cfg.reconstruction.whole_dim // 2 + cfg.reconstruction.phase_dim // 2,
        )
        self.loss_func = cropped_loss(loss_slice)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.2, patience=20
        )
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.5)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     self.optimizer, T_0=self.epochs, eta_min=0.0001
        # )

        if cfg.reconstruction.checkpoint_path:
            self.checkpoint_path = cfg.reconstruction.checkpoint_path
        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint, strict=False)

    def train(
        self,
        callback=None,
        update_progress=None,
        update_val_accuracy=None,
        update_images=None,
    ):

        self._setup_logging()
        best_test_acc = 0
        for epoch in range(self.epochs):
            _, loss = self._train_epoch(epoch, callback)
            self.scheduler.step(loss)
            if update_progress:
                update_progress.emit(epoch / self.epochs)
            test_acc = self._evaluate(epoch)
            if test_acc > 90 and self.model.dmd1.beta.data <= 200:
                for i in range(1, 2):
                    dmd = getattr(self.model, f"dmd{i}")
                    dmd.beta.data = dmd.beta.data + 3
                    print(f"beta{i}", dmd.beta.data)
            if update_val_accuracy:
                update_val_accuracy.emit(test_acc)
            if update_images:
                update_images.emit(self.Image_Path)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(
                    self.model.state_dict(),
                    f"{self.ck_dir}/{best_test_acc:.2f}_{epoch+1:03d}_{self.datatime}.pth",
                )
            logging.info(f"Best test accuracy so far: {best_test_acc:.2f}%")
        logging.info("Finished Training")

    def _train_epoch(self, epoch, callback=None):

        self.model.train()
        running_loss = 0.0
        correct = 0
        correct_sim, correct_phy, total = 0, 0, 0

        for i, data in enumerate(self.trainloader, 0):
            # using one-hot label
            # print(data[1].shape)
            labels = (
                torch.nn.functional.one_hot(data[1], num_classes=10)
                .float()
                .to(self.device)
            )

            pad_labels = pad_label(
                labels,
                cfg.reconstruction.whole_dim,
                cfg.reconstruction.phase_dim,
                cfg.reconstruction.detx,
                cfg.reconstruction.dety,
                cfg.reconstruction.square_size,
            )

            inputs = data[0].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs.squeeze(1))
            # turn the output to 1bit

            # print(outputs.shape, labels.shape)
            loss = self.loss_func(outputs, pad_labels)
            # loss = self.loss_func(outputs, labels)

            loss.backward()
            self.optimizer.step()

            outputs = self.model.detector(outputs)
            _, predicted = torch.max(outputs.data, 1)
            _, corrected = torch.max(labels.data, 1)

            total += labels.size(0)
            correct += (predicted == corrected).sum().item()

            running_loss += loss.item()

            # clamp the phase in 0-1
            # for layer in self.model.named_modules():
            #     if "phase" in layer:
            #         layer.w_p.data = torch.clamp(layer.w_p.data, 0, 1)

            if i % 25 == 24:
                logging.info(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 25:.4f}")
                running_loss = 0.0
                if callback:
                    callback(
                        i,
                        len(self.trainloader),
                        running_loss / total,
                        correct / total,
                    )
        train_acc_sim = 100 * correct / total
        # train_acc_phy = 100 * correct_phy / total
        logging.info(f"Epoch {epoch+1}: Training accuracy: {train_acc_sim:.2f}%  ")

        return train_acc_sim, running_loss / total

    def _evaluate(self, epoch):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i, data in enumerate(self.testloader):
                labels = data[1].to(self.device)
                # turn the label to 0 and 1
                labels = (
                    torch.nn.functional.one_hot(labels, num_classes=10)
                    .float()
                    .to(self.device)
                )
                # labels = torch.cat(
                #     (labels[:, 0].unsqueeze(1), labels[:, 1:].sum(1).unsqueeze(1)),
                #     dim=1,
                # )

                images = data[0].to(self.device)
                outputs = self.model(images.squeeze(1))
                outputs = self.model.detector(outputs)
                _, predicted = torch.max(outputs.data, 1)

                _, corrected = torch.max(labels.data, 1)
                total += labels.size(0)
                correct += (predicted == corrected).sum().item()

                # correct += (predicted == labels.reshape(predicted.shape)).sum().item()
        test_acc = 100 * correct / total
        logging.info(f"Epoch {epoch + 1}: Test accuracy: {test_acc:.2f}%")

        physical_img = get_physical_image(4)

        self.model.plot_phases_and_output(
            self.val_dataset[4][0].to(self.device), physical_img.squeeze(1), show=False
        )
        return test_acc

    def pysical_eval_cmos(self):
        eval_image_path = []
        val_running_counter = 0
        index = 0
        checkpoint = torch.load(
            "D:\project\control\log\\01-24\insituTrain\\17-07-58\checkpoint\\0.71_001_17-07-58.pth"
        )
        # checkpoint = torch.load(
        #     "D:\project\control\checkpoint\\01-10\87.61_002_20240110181939_400.pth"
        # )
        self.model.load_state_dict(checkpoint)

        for layer in self.model.named_modules():
            if "phase" in layer:
                index += 1
                phase = (dorefa_w(layer.w_p, 8)) * math.pi
                fig, ax = plt.subplots(1, 1)
                ax.imshow(
                    phase.cpu().detach().numpy(),
                )
                plt.savefig(
                    f"img/eval_phase_{index}.png", bbox_inches="tight", pad_inches=0
                )
                eval_image_path.append(f"img/eval_phase_{index}.png")
                plt.close()

        for images, labels in self.testloader:
            for x, labels in zip(images, labels):
                # save input image

                Image.fromarray((x.squeeze().numpy() * 255).astype(np.uint8), "L").save(
                    f"img/reconstruction/eval_input.bmp"
                )

                print("-" * 100)
                for ii, (name, layer) in enumerate(self.model.named_modules()):
                    if "phase" in name:
                        if name == "phase1":
                            # 放大到 600 * 600
                            # x = x * 255
                            expose_time = 0.1

                            x = F.interpolate(
                                x.abs().unsqueeze(0),
                                size=(1600, 1600),
                                mode="nearest",
                            ).squeeze(0)

                        x, _ = self.model.physical_forward_one_layer_cmos(
                            x, dorefa_w(layer.w_p, 8), expose_time=expose_time
                        )
                        expose_time = 0.1
                        cv2.imwrite(
                            f"img/reconstruction/orig_cmos_{name}.bmp",
                            x.squeeze().cpu().detach().numpy(),
                        )

                        INNER_PAD = 470
                        x = x.to(self.device)
                        x = torch.nn.functional.pad(
                            x,
                            pad=(
                                -int(INNER_PAD),
                                -int(INNER_PAD),
                                -int(INNER_PAD),
                                -int(INNER_PAD),
                            ),
                        ).float()

                        # pool 500 * 500 to 800 * 800
                        x = F.interpolate(
                            x.unsqueeze(0),
                            size=(800, 800),
                            mode="nearest",
                        ).squeeze(0)
                        print("cmos", torch.unique(x))

                        logging.info(f"layer {ii} {name} {x.size()}")

                        plt.imshow(
                            x.squeeze().cpu().detach().numpy(),
                            vmax=255,
                            vmin=0,
                        )
                        plt.colorbar()

                        plt.savefig(
                            f"img/reconstruction/eval_cmos_{name}.png",
                            bbox_inches="tight",
                            pad_inches=0,
                        )
                        plt.close()

                        modulus_squared = (x / 255).abs()  # ** 2
                        # print('[modulues]',torch.unique(modulus_squared))
                        # I_th = torch.mean(modulus_squared, dim=(-2, -1), keepdim=True)
                        # if name == "phase1":
                        #     x = torch.sigmoid(self.model.dmd1.beta * (modulus_squared  -
                        #     (self.model.dmd1.alpha)
                        #         * (I_th)
                        #     ))
                        # elif name == "phase2":
                        #     # x = torch.sigmoid(self.model.dmd2.beta * (modulus_squared  -
                        #     #     (self.model.dmd2.alpha)
                        #     #         * (I_th)
                        #     #     ))
                        #     x =  modulus_squared
                        # else:
                        #     break

                        # x = dorefa_a(x, 1)
                        # print(torch.unique(x))
                        # print(x)
                        x = modulus_squared

                x = self.model.detector(self.model.w_scalar * x)

    def calibration(self):
        # 创建一个200*200的全亮图像
        img = np.ones((1, 400, 400), dtype=np.uint8) * 255
        img2 = np.ones((1, 200, 200), dtype=np.uint8) * 255

        # 通过键盘打断
        while True:
            # 添加等待时间，超时自动进行
            key = input("input:")
            if key == "q":
                break
            for i in range(3):
                if i == 0:
                    x = img
                x, _ = self.model.physical_forward_one_layer_cmos(x, img2)
                x = x.to(self.device)
                x = torch.nn.functional.pad(
                    x, pad=(-int(774), -int(774), -int(774), -int(774))
                ).float()
                # pool 500 * 500 to 400 * 400
                x = F.interpolate(
                    x.unsqueeze(0),
                    size=(400, 400),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

                fig, ax = plt.subplots(1, 1)
                ax.imshow(
                    x.squeeze().cpu().detach().numpy(),
                    vmax=x.cpu().detach().numpy().max() * 1,
                    vmin=x.cpu().detach().numpy().min() * 1,
                )
                plt.savefig(
                    f"img/reconstruction/eval_cmos_{i}.png",
                    bbox_inches="tight",
                    pad_inches=0,
                )
                x = x.cpu().detach().numpy()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("log/training.log"), logging.StreamHandler()],
        )

    def Pysical_forward_one_step(self, x, label, n, step):
        index = 0

        label = (
            torch.nn.functional.one_hot(label, num_classes=10).float().to(self.device)
        )
        for ii, (name, layer) in enumerate(self.model.named_modules()):
            if "phase" in name:
                if name == "phase1":
                    # 放大到 600 * 600
                    x = x
                    expose_time = 0.02

                    x = F.interpolate(
                        x.abs().unsqueeze(0),
                        size=(1600, 1600),
                        mode="nearest",
                    ).squeeze(0)

                if index < n:
                    x, _ = self.model.physical_forward_one_layer_cmos(
                        x, dorefa_w(layer.w_p, 8), expose_time=expose_time
                    )
                    expose_time = 0.02
                    cv2.imwrite(
                        f"img/insituTrain/orig_cmos_{name}.bmp",
                        (x).squeeze().cpu().detach().numpy(),
                    )

                    self.writer.add_image(
                        "orig_cmos",
                        (x).squeeze().cpu().detach().numpy(),
                        step,
                        dataformats="HW",
                    )

                    INNER_PAD = 500
                    x = x.to(self.device)
                    x = torch.nn.functional.pad(
                        x,
                        pad=(
                            -int(INNER_PAD),
                            -int(INNER_PAD),
                            -int(INNER_PAD),
                            -int(INNER_PAD),
                        ),
                    ).float()

                    x = F.interpolate(
                        x.unsqueeze(0),
                        size=(800, 800),
                        mode="nearest",
                    ).squeeze(0)
                    cv2.imwrite(
                        f"img/insituTrain/train_cmos_{name}.bmp",
                        (x).squeeze().cpu().detach().numpy(),
                    )
                    self.writer.add_image(
                        "transform_cmos",
                        (x).squeeze().cpu().detach().numpy().astype(np.uint8),
                        step,
                        dataformats="HW",
                    )
                    index += 1
                    if index == 1 and n == 2:
                        x = F.interpolate(
                            x.unsqueeze(0),
                            size=(400, 400),
                            mode="nearest",
                        ).squeeze(0)
                        x = torch.nn.functional.pad(
                            x, pad=(200, 200, 200, 200), mode="constant", value=0
                        )
                        x = x / 255
                        x = self.model.dmd1(x, insitu=True)
                        cv2.imwrite(
                            f"img/insituTrain/dmd_phase2.bmp",
                            (x.abs() * 255).squeeze().cpu().detach().numpy(),
                        )
                        self.writer.add_image(
                            "dmd1",
                            (x.abs() * 255)
                            .squeeze()
                            .cpu()
                            .detach()
                            .numpy()
                            .astype(np.uint8),
                            step,
                            dataformats="HW",
                        )
                        x = x.abs()

                else:
                    index += 1
                    if index - n == 1:
                        x = F.interpolate(
                            x.unsqueeze(0),
                            size=(400, 400),
                            mode="nearest",
                        ).squeeze(0)
                        x = torch.nn.functional.pad(
                            x, pad=(200, 200, 200, 200), mode="constant", value=0
                        )
                        x = x / 255
                        # x = self.model.dmd2(x)
                    if index == 2:
                        print("layer2 turn to computer..")
                        if index - n == 1:
                            # x = self.model.input(x)
                            x = self.model.dmd1(x, insitu=True)
                            cv2.imwrite(
                                f"img/insituTrain/dmd_{name}.bmp",
                                (x.abs() * 255).squeeze().cpu().detach().numpy(),
                            )
                            self.writer.add_image(
                                "dmd1",
                                (x.abs() * 255)
                                .squeeze()
                                .cpu()
                                .detach()
                                .numpy()
                                .astype(np.uint8),
                                step,
                                dataformats="HW",
                            )
                        x = self.model.prop(x)
                        x = self.model.phase2(x)
                        x = self.model.prop(x)
                        x = self.model.dmd2(x)
                    elif index == 3:
                        print("layer3 turn to computer..")
                        if index - n == 1:
                            # x = self.model.input(x)
                            x = self.model.dmd2(x, insitu=True)
                            cv2.imwrite(
                                f"img/insituTrain/dmd_{name}.bmp",
                                (x.abs() * 255).squeeze().cpu().detach().numpy(),
                            )
                            self.writer.add_image(
                                "dmd2",
                                (x.abs() * 255)
                                .squeeze()
                                .cpu()
                                .detach()
                                .numpy()
                                .astype(np.uint8),
                                step,
                                dataformats="HW",
                            )
                        x = self.model.prop(x)
                        x = self.model.phase3(x)
                        x = self.model.prop(x)

        x = self.model.detector(self.model.w_scalar.cuda() * x)

        loss = self.loss_func(x, label.unsqueeze(0))
        return x, loss

    def pysical_train(self, n, thread, callback):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.size())

                # 解析层次号
                if "phase" in name:
                    layer_number = int(name.split(".")[0].split("phase")[1])
                    # 冻结 self.value 之前的层
                    if layer_number <= n:
                        param.requires_grad = False
                        print(f"[Frozen layer]: {name}")

        checkpoint = torch.load(
            "D:\project\control\log\\01-31\green\\20-53-39\checkpoint\\400-45-0.5\96.02_001_20-53-39.pth"
        )
        self.model.load_state_dict(checkpoint)
        self.model.train()
        train_running_loss = 0.0
        train_running_correct = 0
        total_images = 0
        print("training starts.")
        print("len of subtrainloader", len(self.subtrainloader))
        index = 0
        best_acc = 0

        for epoch in range(self.subepoch):

            for ii, (images, labels) in enumerate(self.subtrainloader):
                batch_outputs = []
                batch_losses = []

                for image, label in zip(images, labels):
                    if thread.is_running:
                        output, loss = self.Pysical_forward_one_step(
                            image, label, n, step=ii
                        )

                        batch_outputs.append(output)
                        batch_losses.append(loss)
                    else:
                        return None, None, None, None

                # 将所有损失相加，然后进行反向传播

                print(
                    f"alpha: {self.model.dmd1.alpha.data},  beta: {self.model.dmd1.beta.data}"
                )
                print(
                    f"alpha: {self.model.dmd2.alpha.data},  beta: {self.model.dmd2.beta.data}"
                )
                self.writer.add_scalar("dmd1_alpha", self.model.dmd1.alpha.data, ii)
                # self.writer.add_scalar('dmd1_beta', self.model.dmd1.beta.data, ii)
                self.writer.add_scalar("dmd2_alpha", self.model.dmd2.alpha.data, ii)
                self.optimizer.zero_grad()
                total_loss = torch.stack(batch_losses).sum()
                total_loss.backward()
                self.optimizer.step()

                # 计算准确率
                if len(batch_outputs) != 1:
                    batch_outputs = torch.cat(batch_outputs, dim=0).squeeze()
                else:
                    batch_outputs = batch_outputs[0]
                predicted = torch.argmax(batch_outputs, dim=1)
                correct = (predicted == labels.to(self.device)).float().sum()

                train_running_loss += total_loss.item()
                train_running_correct += correct.item()
                print("train_running_correct", train_running_correct)
                total_images += len(labels)
                print(total_images)
                index += 1
                if callback:
                    callback(
                        index,
                        len(self.trainloader),
                        train_running_loss / total_images,
                        train_running_correct / total_images,
                    )

                train_loss = train_running_loss / total_images
                train_accuracy = train_running_correct / total_images

                if ii % 25 == 24:
                    print("plot...")
                    self.model.plot_phases_and_output(
                        self.val_dataset[1][0].to(self.device), show=False
                    )
                    if best_acc < train_accuracy:
                        best_acc = train_accuracy
                        torch.save(
                            self.model.state_dict(),
                            f"{self.ck_dir}/{best_acc:.2f}_{epoch+1:03d}_{self.datatime}.pth",
                        )

            self.scheduler.step(train_loss)
        self.writer.close()

        return train_loss, train_accuracy

    def adaptive_training(self, thread):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.size())
                param.requires_grad = False
                if "phase" in name:
                    layer_number = int(name.split(".")[0].split("phase")[1])
                    if layer_number == 1:
                        param.requires_grad = True
                        print(f"[train layer]: {name}")

        checkpoint = torch.load(
            "D:\project\control\log\\01-31\green\\20-53-39\checkpoint\\400-45-0.5\96.02_001_20-53-39.pth"
        )

        self.model.load_state_dict(checkpoint)

        self.pysical_model = copy.deepcopy(self.model)
        self.pysical_optimizer = torch.optim.Adam(
            self.pysical_model.parameters(),
            lr=self.learning_rate,
        )

        self.pysical_model.train()
        train_running_loss = 0.0
        train_running_correct = 0
        total_images = 0
        print("training starts.")
        print("len of subtrainloader", len(self.subtrainloader))
        index = 0
        best_acc = 0

        for epoch in range(self.subepoch):

            for ii, (images, labels) in enumerate(self.subtrainloader):
                batch_outputs = []
                batch_losses = []

                for image, label in zip(images, labels):
                    x = F.interpolate(
                        image.abs().unsqueeze(0),
                        size=(1600, 1600),
                        mode="nearest",
                    ).squeeze(0)
                    x, _ = self.pysical_model.physical_forward_one_layer_cmos(
                        x, dorefa_w(self.pysical_model.phase1.w_p, 8)
                    )
                    INNER_PAD = 500
                    x = x.to(self.device)
                    x = torch.nn.functional.pad(
                        x,
                        pad=(
                            -int(INNER_PAD),
                            -int(INNER_PAD),
                            -int(INNER_PAD),
                            -int(INNER_PAD),
                        ),
                    ).float()
                    x = F.interpolate(
                        x.unsqueeze(0),
                        size=(400, 400),
                        mode="nearest",
                    ).squeeze(0)
                    x = F.pad(x, pad=(200, 200, 200, 200))
                    x = x / 255
                    # x = x **2
                    # x = torch.tanh(x)
                    label = dorefa_a(
                        torch.square(
                            self.model.prop(
                                self.model.phase1(self.model.input(dorefa_a(image, 1)))
                            ).abs()
                        ),
                        8,
                    )
                    loss = self.loss_func(x, label)
                    batch_outputs.append(x)
                    batch_losses.append(loss)
                self.writer.add_image(
                    "pysical_output",
                    (x.abs() * 255).squeeze().cpu().detach().numpy().astype(np.uint8),
                    ii,
                    dataformats="HW",
                )
                self.writer.add_image(
                    "label",
                    (label.abs() * 255)
                    .squeeze()
                    .cpu()
                    .detach()
                    .numpy()
                    .astype(np.uint8),
                    ii,
                    dataformats="HW",
                )

                self.pysical_optimizer.zero_grad()
                total_loss = torch.stack(batch_losses).sum()
                total_loss.backward()
                self.pysical_optimizer.step()
                self.writer.add_scalar("pysical_loss", total_loss, ii)

                # 将所有损失相加，然后进行反向传播

                print(f"phase1: {torch.mean((self.pysical_model.phase1.w_p))}")
                print(f"phase2: {torch.mean((self.pysical_model.phase2.w_p))}")

                # 计算准确率
                if len(batch_outputs) != 1:
                    batch_outputs = torch.cat(batch_outputs, dim=0).squeeze()
                else:
                    batch_outputs = batch_outputs[0]

                train_running_loss += total_loss.item()
                total_images += len(labels)
                print(ii)
                index += 1

                train_loss = train_running_loss / total_images

                if ii % 25 == 24:
                    print("plot...")
                    self.pysical_model.plot_phases_and_output(
                        self.val_dataset[1][0].to(self.device), show=False
                    )

                    torch.save(
                        self.model.state_dict(),
                        f"{self.ck_dir}/{epoch+1:03d}_{self.datatime}.pth",
                    )

                self.scheduler.step(train_loss)

        return train_loss


class OpticalTrainerWithEvent(OpticalNetworkTrainer):
    def __init__(
        self,
        data_path="./data",
        checkpoint_path=None,
        learning_rate=0.01,
        batch_size=20,
        epochs=200,
        insitu_train=False,
    ):
        self.data_path = data_path
        self.checkpoint_path = checkpoint_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.subbatch_size = 1
        self.subepoch = 2
        self.epochs = epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.insitu_train = insitu_train

        self._prepare_log(insitu_train=insitu_train)
        self._prepare_data()
        self._initialize_model()
        self._setup_logging()

    def _prepare_log(self, insitu_train=False):
        self.datatime = time.strftime("%H-%M-%S", time.localtime())
        self.date = time.strftime("%m-%d", time.localtime())
        self.log_dir = f"log/{self.date}"
        if cfg.reconstruction.wl == 532e-9:
            self.log_dir = f"log/{self.date}/green/{self.datatime}"
        elif cfg.reconstruction.wl == 632e-9:
            self.log_dir = f"log/{self.date}/red/{self.datatime}"
        self.ck_dir = f"{self.log_dir}/checkpoint/{cfg.reconstruction.phase_dim}-{cfg.reconstruction.square_size}-{cfg.reconstruction.distance}"

        if not insitu_train:
            if not os.path.exists(self.log_dir) or not os.path.exists(self.ck_dir):
                os.makedirs(self.log_dir, exist_ok=True)
                os.makedirs(self.ck_dir, exist_ok=True)

        else:
            self.log_dir = f"log/{self.date}/insituTrain/{self.datatime}"
            self.ck_dir = f"log/{self.date}/insituTrain/{self.datatime}/checkpoint"
            if not os.path.exists(self.ck_dir):
                os.makedirs(self.ck_dir, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir)

        cfg.reconstruction.log_dir = self.log_dir
        cfg.reconstruction.ck_dir = self.ck_dir

        cfg.reconstruct_phase = [
            f"{self.log_dir}/Phase1.bmp",
            f"{self.log_dir}/Phase2.bmp",
            f"{self.log_dir}/Phase3.bmp",
        ]
        cfg.reconstruct_output = [
            f"{self.log_dir}/Output1.png",
            f"{self.log_dir}/Output2.png",
            f"{self.log_dir}/Output3.png",
        ]

        self.Image_Path = OmegaConf.to_container(
            cfg.reconstruct_phase + cfg.reconstruct_output, resolve=True
        )  # 转换为普通列表

    def _prepare_data(self):
        subset_size = 2000
        pad = (cfg.reconstruction.whole_dim - cfg.reconstruction.phase_dim) // 2
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (cfg.reconstruction.phase_dim, cfg.reconstruction.phase_dim)
                ),
                transforms.Pad([pad, pad], fill=(0), padding_mode="constant"),
            ]
        )
        self.train_dataset = torchvision.datasets.MNIST(
            self.data_path, train=True, transform=transform, download=True
        )
        self.val_dataset = torchvision.datasets.MNIST(
            self.data_path, train=False, transform=transform, download=True
        )
        self.trainloader = DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.testloader = DataLoader(
            dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False
        )

        train_indices = torch.randperm(len(self.train_dataset))[:subset_size]
        val_indices = torch.randperm(len(self.val_dataset))[:subset_size]

        train_dataset = Subset(self.train_dataset, train_indices)
        val_dataset = Subset(self.val_dataset, val_indices)

        # 创建数据加载器
        self.subtrainloader = DataLoader(
            dataset=train_dataset, batch_size=self.subbatch_size, shuffle=True
        )
        self.subtestloader = DataLoader(
            dataset=val_dataset, batch_size=self.subbatch_size, shuffle=False
        )

    def _initialize_model(self):
        self.model = DDNN(
            cfg.reconstruction.whole_dim,
            cfg.reconstruction.phase_dim,
            cfg.reconstruction.pixel_size,
            cfg.reconstruction.distance,
            cfg.reconstruction.wl,
            cfg=cfg,
        )
        self.model.to(self.device)
        self.loss_func = nn.CrossEntropyLoss()
        loss_slice = slice(
            cfg.reconstruction.whole_dim // 2 - cfg.reconstruction.phase_dim // 2,
            cfg.reconstruction.whole_dim // 2 + cfg.reconstruction.phase_dim // 2,
        )
        # self.loss_func = cropped_loss(loss_slice)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.2, patience=20
        )
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)

        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint)

    def pysical_eval_event(self):
        eval_image_path = []
        val_running_counter = 0
        index = 0
        checkpoint = torch.load(
            "D:\project\control\log\\01-24\insituTrain\\17-07-58\checkpoint\\0.92_001_17-07-58.pth"
        )
        # checkpoint = torch.load(
        #     "D:\project\control\checkpoint\\01-10\87.61_002_20240110181939_400.pth"
        # )
        self.model.load_state_dict(checkpoint)

        for layer in self.model.named_modules():
            if "phase" in layer:
                index += 1
                phase = (dorefa_w(layer.w_p, 8)) * math.pi
                fig, ax = plt.subplots(1, 1)
                ax.imshow(
                    phase.cpu().detach().numpy(),
                )
                plt.savefig(
                    f"img/eval_phase_{index}.png", bbox_inches="tight", pad_inches=0
                )
                eval_image_path.append(f"img/eval_phase_{index}.png")
                plt.close()

        for images, labels in self.testloader:
            for x, labels in zip(images, labels):
                # save input image

                Image.fromarray((x.squeeze().numpy() * 255).astype(np.uint8), "L").save(
                    f"img/reconstruction/eval_input.bmp"
                )

                print("-" * 100)
                for ii, (name, layer) in enumerate(self.model.named_modules()):
                    if "phase" in name:
                        if name == "phase1":

                            x = F.interpolate(
                                x.abs().unsqueeze(0),
                                size=(1600, 1600),
                                mode="nearest",
                            ).squeeze(0)

                        x, _ = self.model.physical_forward_one_layer_event(
                            x, dorefa_w(layer.w_p, 8)
                        )
                        cv2.imwrite(
                            f"img/reconstruction/orig_evnt_{name}.bmp",
                            x.squeeze().cpu().detach().numpy(),
                        )

                        INNER_PAD = 470
                        x = x.to(self.device)
                        x = torch.nn.functional.pad(
                            x,
                            pad=(
                                -int(INNER_PAD),
                                -int(INNER_PAD),
                                -int(INNER_PAD),
                                -int(INNER_PAD),
                            ),
                        ).float()

                        # pool 500 * 500 to 800 * 800
                        x = F.interpolate(
                            x.unsqueeze(0),
                            size=(800, 800),
                            mode="nearest",
                        ).squeeze(0)
                        print("cmos", torch.unique(x))

                        logging.info(f"layer {ii} {name} {x.size()}")

                        plt.imshow(
                            x.squeeze().cpu().detach().numpy(),
                            vmax=255,
                            vmin=0,
                        )
                        plt.colorbar()

                        plt.savefig(
                            f"img/reconstruction/eval_event_{name}.png",
                            bbox_inches="tight",
                            pad_inches=0,
                        )
                        plt.close()

                        modulus_squared = (x / 255).abs()  # ** 2
                        x = modulus_squared

                x = self.model.detector(self.model.w_scalar * x)


class reconstruction_pysical_trainthread_with_cmos(QThread):
    """
    Add the training process to a thread
    Separate the training process from the main thread
    """

    update_progress = pyqtSignal(int)
    update_images = pyqtSignal(list)
    update_val_accuracy = pyqtSignal(float)
    errorOccurred = pyqtSignal(str)

    def __init__(
        self, value, train, calibration, insitu_train=False, error=False, callback=None
    ):
        QThread.__init__(self)
        self.value = value
        self.callback = callback
        self.is_running = True
        self.train = train
        self.calibration = calibration
        self.insitu_train = insitu_train

    def run(self):
        # try:
        if self.calibration:
            trainer = OpticalNetworkTrainer()
            trainer.calibration()
        else:
            if self.train:
                trainer = OpticalNetworkTrainer()
                trainer.train(
                    callback=self.callback,
                    update_progress=self.update_progress,
                    update_val_accuracy=self.update_val_accuracy,
                    update_images=self.update_images,
                )
            elif self.insitu_train:
                trainer = OpticalNetworkTrainer(insitu_train=True)
                trainer.pysical_train(
                    self.value,
                    self,
                    callback=self.callback,
                )
                # trainer.adaptive_training(self)
            else:
                trainer = OpticalNetworkTrainer()
                trainer.pysical_eval_cmos()

    # except Exception as e:
    #     self.errorOccurred.emit(str(e))
    # Optionally, log the error or handle it further here

    def stop(self):
        self.is_running = False
