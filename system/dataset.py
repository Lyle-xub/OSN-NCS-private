import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from os.path import join
import torchvision
import h5py
import os


class HDF5Dataset(Dataset):

    def __init__(self, file_path, transform=None):
        """
        Args:
            file_path (string): Path to the HDF5 file with images and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.file_path = file_path
        self.file = h5py.File(self.file_path, 'r')
        self.images = self.file['images']
        self.labels = self.file['labels']
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Access the image and label from the HDF5 dataset
        image = self.images[idx]
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.int64)

        # Convert the numpy array to a PIL Image
        image = Image.fromarray(image.astype('uint8')).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_counts(self):
        # Calculate the number of instances of each class
        _, counts = np.unique(self.labels[:], return_counts=True)
        return counts

    def close(self):
        self.file.close()


class S4NNFilePathManager:

    def __init__(self, dataset_type, base_dir="data/KTH"):
        self.base_dir = base_dir
        self.paths = {}
        self.dataset_type = dataset_type
        self._set_paths()

    def add_paths(self, prefix, train_hdf5, dev_hdf5, dev_npy, detail_npz, model_pth):
        self.paths[prefix] = {
            "train": os.path.join(self.base_dir, train_hdf5),
            "dev": os.path.join(self.base_dir, dev_hdf5),
            "dev_npy_path": os.path.join(self.base_dir, dev_npy),
            "detail_path": os.path.join(self.base_dir, detail_npz),
            "model_save_path": os.path.join(self.base_dir, model_pth),
        }

    def get_paths(self, prefix=None):
        if prefix:
            return self.paths.get(prefix)
        return self.paths.get(self.dataset_type)

    def _set_paths(self):
        if self.dataset_type == 'b-hc':
            self.add_paths(
                "b-hc",
                "b-hc_train.hdf5",
                "b-hc_dev.hdf5",
                "b-hc_dev.npy",
                "b-hc_01_random_S4NN.npz",
                "model/b-hc_01_random_S4NN.pth",
            )
        elif self.dataset_type == 'b-hw':
            self.add_paths(
                "b-hw",
                "KTH01_b-hw_random_train.hdf5",
                "b-hw_dev.hdf5",
                "b-hw_dev.npy",
                "b-hw_01_random_S4NN.npz",
                "model/b-hw_01_random_S4NN.pth",
            )
        elif self.dataset_type == 'hc-hw':
            self.add_paths(
                "hc-hw",
                "KTH01_hc-hw_random_train1.hdf5",
                "hc-hw_dev.hdf5",
                "hc-hw_dev.npy",
                "hc-hw_01_random_S4NN.npz",
                "model/hc-hw_01_random_S4NN.pth",
            )
        else:
            raise ValueError(f"Invalid dataset_type: {self.dataset_type}.  Must be 'b-hc', 'b-hw', or 'hc-hw'.")
