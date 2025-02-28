from system.parameter import *
import torch
import torchvision
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from system.utils import *
import scipy.io as scio
from copy import deepcopy
from system.model import *
from torch.utils.data import DataLoader, Subset


class BaseAgent:
    logger = get_logger("log.txt")
    current_epoch = 0
    current_iteration = 0
    scheduler = None
    criterian = nn.MSELoss()
    optimizer = None
    relu = nn.ReLU(inplace=True)

    def __init__(self, params):
        self.params = params
        self.manual_seed = self.params.manual_seed
        print("seed: ", self.manual_seed)
        random.seed(self.manual_seed)
        np.random.seed(self.manual_seed)
        self.device = params.device
        if self.params.model_type == "s4nn":
            self.network = s4nn(params)
        else:
            self.network = DDNN(params, **params.core["network"])
        self.lr = self.params.lr
        self.evaluate = eval
        # self.update_settings()

    def load_checkpoint(self, filename):
        if filename is None:
            self.logger.info("do not load checkpoint")
        elif self.params.model_type == "s2nn" or self.params.model_type == "s3nn":
            checkpoint = torch.load(filename)
            donn_state = checkpoint["donn"]
            self.network.load_state_dict(donn_state, strict=False)
            self.current_epoch = checkpoint["epoch"]
            self.current_iteration = checkpoint["iteration"]
            self.lr = checkpoint["lr"]
            # if self.optimizer is not None and "optimizer" in checkpoint:
            #     self.optimizer.load_state_dict(checkpoint["optimizer"])
            # if self.scheduler is not None and "scheduler" in checkpoint:
            #     self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n".format(
                filename,
                checkpoint["epoch"],
                checkpoint["iteration"],
            ))
        else:
            checkpoint = torch.load(filename)
            _, _ = self.network.load_state_dict(checkpoint, strict=False)

    def run(self):
        self.train()

    def train(self):
        best_test_acc = 0
        for epoch in range(self.params.max_epoch):
            loss = self._train_epoch(epoch)
            self.scheduler.step()
            test_acc = self._evaluate(epoch)
            if test_acc > 90 and self.network.dmd1.beta.data <= 200:
                for i in range(1, 2):
                    dmd = getattr(self.network, f"dmd{i}")
                    dmd.beta.data = dmd.beta.data + 3
                    print(f"beta{i}", dmd.beta.data)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                self.current_epoch = epoch
                self.save_checkpoint()
            self.logger.info(f"Best test accuracy so far: {best_test_acc:.2f}%")
        self.logger.info("Finished Training")

    def _train_epoch(self, epoch):
        self.network.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(self.trainloader, 0):
            self.current_iteration = i
            labels = (torch.nn.functional.one_hot(data[1], num_classes=10).float().to(self.device))
            pad_labels = pad_label(
                labels,
                self.params.core["network"]["whole_dim"],
                self.params.core["network"]["phase_dim"],
                **self.params.core["data"]["detectors"],
            )
            inputs = data[0].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.network(inputs.squeeze(1))
            # print(outputs.shape, pad_labels.shape)
            # loss = self.loss_func(outputs, pad_labels)
            outputs_det = self.network.detector(outputs)
            loss = self.criterian(outputs_det, labels)
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(outputs_det.data, 1)
            _, corrected = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == corrected).sum().item()
            running_loss += loss.item()
            if i % 25 == 24:
                self.logger.info(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 25:.4f}")
                running_loss = 0.0

        train_acc = 100 * correct / total
        self.logger.info(f"Epoch {epoch+1}: Training accuracy: {train_acc:.2f}%")
        self.network.plot_phases_and_output(self.val_dataset[4][0].to(self.device), show=False)
        return running_loss / total

    def _evaluate(self, epoch):
        self.network.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i, data in enumerate(self.testloader):
                labels = data[1].to(self.device)
                # turn the label to 0 and 1
                labels = (torch.nn.functional.one_hot(labels, num_classes=10).float().to(self.device))
                images = data[0].to(self.device)
                outputs = self.network(images.squeeze(1))
                outputs = self.network.detector(outputs)
                _, predicted = torch.max(outputs.data, 1)
                _, corrected = torch.max(labels.data, 1)
                total += labels.size(0)
                correct += (predicted == corrected).sum().item()

        test_acc = 100 * correct / total
        self.logger.info(f"Epoch {epoch + 1}: Test accuracy: {test_acc:.2f}%")

        # self.network.plot_phases_and_output(
        #     self.val_dataset[4][0].to(self.device), None, show=False
        # )
        return test_acc

    def save_checkpoint(self):
        file_name = "epoch_%s.pth.tar" % str(self.current_epoch).zfill(5)

        state = {
            "epoch": self.current_epoch,
            "lr": self.lr,
            "iteration": self.current_iteration,
            "donn": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        torch.save(state, os.path.join(self.params.checkpoint_dir, file_name))


class SimAgent(BaseAgent):

    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.loss_func = cropped_loss(params.loss_slice)
        # self.loss_func = self.criterian
        self.network = DDNN(params, **params.core["network"])
        self._prepare_data()
        self._initialize_model()
        # self.update_settings()

    def _prepare_data(self):
        subset_size = 2000
        pad = (self.params.core["network"]["whole_dim"] - self.params.core["network"]["phase_dim"]) // 2
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (
                    self.params.core["network"]["phase_dim"],
                    self.params.core["network"]["phase_dim"],
                ),
                antialias=True,
            ),
            transforms.Pad([pad, pad], fill=(0), padding_mode="constant"),
        ])
        dev_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (
                    self.params.core["network"]["phase_dim"],
                    self.params.core["network"]["phase_dim"],
                ),
                antialias=True,
            ),
            transforms.Pad([pad, pad], fill=(0), padding_mode="constant"),
        ])
        self.train_dataset = torchvision.datasets.MNIST("data", train=True, transform=train_transform, download=True)
        self.val_dataset = torchvision.datasets.MNIST("data", train=False, transform=dev_transform, download=True)
        self.trainloader = DataLoader(dataset=self.train_dataset, batch_size=self.params.batch_size, shuffle=True)
        self.testloader = DataLoader(dataset=self.val_dataset, batch_size=1, shuffle=False)

        train_indices = torch.randperm(len(self.train_dataset))[:subset_size]
        val_indices = torch.randperm(len(self.val_dataset))[:subset_size]

        train_dataset = Subset(self.train_dataset, train_indices)
        val_dataset = Subset(self.val_dataset, val_indices)

        self.subtrainloader = DataLoader(dataset=train_dataset, batch_size=self.params.subbatch_size, shuffle=True)
        self.subtestloader = DataLoader(dataset=val_dataset, batch_size=self.params.subbatch_size, shuffle=False)

    def _initialize_model(self):
        self.network.to(self.device)
        print(self.params.lr)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.params.lr)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode="min", factor=0.2, patience=20)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                    T_max=self.params.max_epoch,
                                                                    eta_min=0.0001)


if __name__ == "__main__":
    params = SimParams()
    agent = SimAgent(params)
    agent.train()
    pass
