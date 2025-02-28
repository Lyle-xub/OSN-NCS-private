import torch
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import logging
from PIL import Image
import cv2
import torchvision
from torch.nn.functional import normalize
from torchvision import transforms
from scipy.ndimage import rotate


def repeat_and_pad_rotate(image, size=(1600, 2560), angle=-45, k=1.575):

    interval = int(20 * k)
    img_height, img_width = image.shape
    new_width = img_width * 3 + interval * 2
    repeated_image = np.zeros((img_height, new_width), dtype=image.dtype)

    for i in range(3):
        start_x = i * (img_width + interval)
        repeated_image[:, start_x:start_x + img_width] = image

    print(repeated_image.shape)
    pad_x = int((size[0] - repeated_image.shape[0]) / 2)
    pad_y = int((size[1] - repeated_image.shape[1]) / 2)
    print(pad_x, pad_y)

    padded_image = np.pad(
        repeated_image,
        ((pad_x, pad_x), (pad_y, pad_y)),
        "constant",
        constant_values=0,
    )

    return rotate(padded_image, angle, reshape=False)


def eval(output, label):
    top_n, top_i = output.topk(1)
    return (top_i[:, 0] == label).float().mean()


def get_logger(log_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if not logger.handlers:
        fh = logging.FileHandler(log_dir)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger


def pad_label(
    label,
    whole_dim,
    phase_dim,
    detx=None,
    dety=None,
    square_size=40,
):

    batch_size = label.shape[0]
    padded_labels = torch.zeros(batch_size, phase_dim, phase_dim, device=label.device)

    for i in range(batch_size):
        _, index = torch.max(label[i], dim=0)
        x_start = max(0, min(detx[int(index.cpu().numpy())], phase_dim - square_size))
        y_start = max(0, min(dety[int(index.cpu().numpy())], phase_dim - square_size))
        padded_labels[i, x_start:x_start + square_size, y_start:y_start + square_size] = 1

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
        return torch.mean(torch.abs(diff)**2)


def get_roi_from_file(path: str):
    """
    Helper function to read bias from a file
    """
    roi = {}
    try:
        roi_file = open(path, "r")
    except IOError:
        print("Cannot open roi file: " + path)
    else:
        for line in roi_file:
            # Skip lines starting with '%': comments
            if line.startswith("%"):
                continue

            split_line = line.split(" ")
            if len(split_line) == 4:
                roi["x"] = int(split_line[0])
                roi["y"] = int(split_line[1])
                roi["width"] = int(split_line[2])
                roi["height"] = int(split_line[3])

    return roi


def get_biases_from_file(path: str):
    """
    Helper function to read bias from a file
    """
    biases = {}
    try:
        biases_file = open(path, "r")
    except IOError:
        print("Cannot open bias file: " + path)
    else:
        for line in biases_file:
            if line.startswith("%"):
                continue

            split = line.split("%")
            biases[split[1].strip()] = int(split[0])
    return biases


def generate_square_coordinates(canvas_size, square_size, pattern):
    coordinates = []
    y_offset = (canvas_size[1] - (len(pattern) * square_size)) // (len(pattern) + 1)
    current_y = y_offset

    for row in pattern:
        if row == 0:
            current_y += square_size + y_offset
            continue
        x_offset = (canvas_size[0] - (row * square_size)) // (row + 1)
        current_x = x_offset

        for _ in range(row):
            coordinates.append((current_x, current_y))
            current_x += square_size + x_offset

        current_y += square_size + y_offset

    return coordinates


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return "{:.4f}({:.2f}) ".format(self.avg, self.val)


class AverageMeterList(object):

    def __init__(self, len_):
        self.len_ = len_
        self.AML = [AverageMeter() for _ in range(len_)]
        self.reset()

    def reset(self):
        for AM in self.AML:
            AM.reset()

    def update(self, val_list, n=1):
        for val, AM in zip(val_list, self.AML):
            AM.update(val, n)

    def avg(self):
        return [AM.avg for AM in self.AML]

    def __repr__(self):
        res = ""
        for AM in self.AML:
            res += AM.__repr__()
        return res


def diff_loss(x, y):
    return torch.mean(torch.abs(x - y))


def correct(energe, label):
    corr = (energe.argmax(dim=-1) == label.argmax(dim=-1)).sum().item()
    corr /= energe.size(0)
    return corr


def map_values_to_int(array, custom_order=None, start=0, step=1, max_value=202):
    unique_values = np.unique(array)
    if custom_order is None:
        custom_order = unique_values
    else:
        if not set(unique_values).issubset(set(custom_order)):
            raise ValueError("custom_order must contain all unique values from the array.")
        custom_order = [val for val in custom_order if val in unique_values]

    mapping = {}
    current_value = start
    for value in custom_order:
        mapping[value] = current_value % (max_value + 1)
        current_value += step

    unique_values, indices = np.unique(phase, return_inverse=True)
    mapped_arr = indices.reshape(phase.shape)
    print("Using np.unique (sorted order):\n", mapped_arr)
    print(mapped_arr)
    phase = mapped_arr.squeeze(0)

    return mapping


def contrast_exponential(image, factor):
    normalized_image = image.astype(np.float32) / 255.0
    adjusted_image = np.power(normalized_image, factor)
    adjusted_image = (adjusted_image * 255.0).astype(np.uint8)
    return adjusted_image


def contrast_logarithmic(image, factor):

    normalized_image = image.astype(np.float32) / 255.0 + 1e-5
    adjusted_image = factor * np.log(normalized_image)
    adjusted_image = cv2.normalize(adjusted_image, None, 0, 1, cv2.NORM_MINMAX)
    adjusted_image = (adjusted_image * 255.0).astype(np.uint8)

    return adjusted_image


def contrast_inverse_curve(img, alpha=1):
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_norm = hist / img.size
    cdf = hist_norm.cumsum()

    mapping = (255 * (cdf)**alpha).astype(np.uint8)
    output_img = cv2.LUT(img, mapping)

    return output_img


def create_checkpoint_directory(exp):
    # Define the base directory
    exp_id = 0
    checkpoint_dir_base = os.path.join("record", exp)

    while True:
        checkpoint_dir = os.path.join(checkpoint_dir_base, str(exp_id))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Created directory: {checkpoint_dir}")
            break
        exp_id += 1
    print("exp_id: ", exp_id)
    return checkpoint_dir, exp_id


def complexor(inp: torch.Tensor):
    return torch.stack((inp.real, inp.imag), dim=0)


class CNormSq(nn.Module):

    def __init__(self, normed=True):
        super(CNormSq, self).__init__()
        self.normed = normed

    def forward(self, inputs):
        return normalize(complexor(inputs), dim=1) if self.normed else complexor(inputs)


def save_image(img, save_dir, norm=True, Gray=False):
    imgPath = os.path.join(save_dir)
    grid = torchvision.utils.make_grid(img, nrow=4, padding=2, pad_value=255, normalize=norm, scale_each=False)
    if norm:
        # print(grid.max(), grid.min())
        ndarr = ((grid * 255 + 0.5).clamp_(0, 255).to(torch.uint8).permute(1, 2,
                                                                           0).detach().cpu().numpy().astype(np.uint8))
    else:
        ndarr = grid.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    # im.show()
    if Gray:
        im.convert("L").save(imgPath)  # Gray = 0.29900 * R + 0.58700 * G + 0.11400 * B
    else:
        im.save(imgPath)


def crop_center(img, size=400):
    width, height = img.size
    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2
    return img.crop((left, top, right, bottom))


def superimpose_images(image1, image2):
    img1 = np.array(image1)
    img2 = np.array(image2)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    superimpose = np.mod(img1 + img2, 255).astype(np.uint8)
    return superimpose


def pad_image(img, new_size=(1272, 1024)):
    padded_image = Image.new("L", (new_size[0], new_size[1]), color="black")
    offset = ((new_size[0] - img.width) // 2, (new_size[1] - img.height) // 2)
    padded_image.paste(img, offset)
    return padded_image


def events_to_diff_image_positive_only(events, sensor_size=(720, 1280), strict_coord=True):

    xs = events["x"]
    ys = events["y"]
    ps = events["p"]

    positive_events = ps == 1
    xs = xs[positive_events]
    ys = ys[positive_events]

    coords = np.stack((ys, xs))

    abs_coords = np.ravel_multi_index(coords, sensor_size)

    img = np.bincount(abs_coords, minlength=sensor_size[0] * sensor_size[1])

    img = img.reshape(sensor_size)

    return img


def events_to_diff_image(events, sensor_size=(720, 1280), strict_coord=True):
    """
    Place events into an image using numpy
    """
    xs = events["x"]
    ys = events["y"]
    ps = events["p"] * 2 - 1

    # Ensure all coordinates are valid (this assumption means no need for mask)
    coords = np.stack((ys, xs))

    # Convert 2D coordinates to 1D indices
    abs_coords = np.ravel_multi_index(coords, sensor_size)

    # Bin counts based on 1D indices and polarities
    img = np.bincount(abs_coords, weights=ps, minlength=sensor_size[0] * sensor_size[1])

    # Reshape the 1D array back to the 2D image shape
    img = img.reshape(sensor_size)

    return img
