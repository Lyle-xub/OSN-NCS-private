import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
import random
from torch.nn.functional import normalize
# from donn import DDNN
# from easydict import EasyDict as edict


def padding(array, whole_dim):
    # pad square array
    array_size = array.shape[-1]
    pad_size1 = (whole_dim - array_size) // 2
    pad_size2 = whole_dim - array_size - pad_size1
    array_pad = F.pad(array, (pad_size1, pad_size2, pad_size1, pad_size2))
    return array_pad


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class cropped_loss(nn.Module):

    def __init__(self, whole_dim, phase_dim):
        super(cropped_loss, self).__init__()
        loss_slice = slice(
            whole_dim // 2 - phase_dim // 2,
            whole_dim // 2 + phase_dim // 2,
        )
        self.loss_slice = loss_slice

    def forward(self, output, target):
        # print(self.loss_slice)
        diff = (output - target)[:, self.loss_slice, self.loss_slice]
        return torch.mean(torch.abs(diff)**2)


def diff_loss(x, y):
    return torch.mean(torch.abs(x - y))


def create_unique_experiment_folder(phase_error, training_method):
    experiment_id = 1
    folder_created = False
    attempt = 0

    while not folder_created:
        folder_name = f"log/{experiment_id:03}_error_{phase_error}_{training_method}"
        folder_path = os.path.join(os.getcwd(), folder_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)  # 创建文件夹
            print(f"Folder created: {folder_path}")
            folder_created = True
        else:
            experiment_id += 1

    return folder_path, experiment_id


def complexor(inp: torch.Tensor):
    return torch.stack((inp.real, inp.imag), dim=0)


class CNormSq(nn.Module):

    def __init__(self, normed=True):
        super(CNormSq, self).__init__()
        self.normed = normed

    def forward(self, inputs):
        return normalize(complexor(inputs), dim=1) if self.normed else complexor(inputs)
