"""train and test dataset

author baiyu
"""

import os
import sys
import pickle
import glob
from PIL import Image

import os
import numpy
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class CIFAR100Train(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        # if transform is given, we transoform data using
        with open(os.path.join(path, "train"), "rb") as cifar100:
            self.data = pickle.load(cifar100, encoding="bytes")
        self.transform = transform

    def __len__(self):
        return len(self.data["fine_labels".encode()])

    def __getitem__(self, index):
        label = self.data["fine_labels".encode()][index]
        r = self.data["data".encode()][index, :1024].reshape(32, 32)
        g = self.data["data".encode()][index, 1024:2048].reshape(32, 32)
        b = self.data["data".encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image


class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        with open(os.path.join(path, "test"), "rb") as cifar100:
            self.data = pickle.load(cifar100, encoding="bytes")
        self.transform = transform

    def __len__(self):
        return len(self.data["data".encode()])

    def __getitem__(self, index):
        label = self.data["fine_labels".encode()][index]
        r = self.data["data".encode()][index, :1024].reshape(32, 32)
        g = self.data["data".encode()][index, 1024:2048].reshape(32, 32)
        b = self.data["data".encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image


class CustomImageDataset(Dataset):
    """Custom Dataset for single-channel images."""

    def __init__(self, path, transform=None):
        self.image_paths = glob.glob(os.path.join(path, "**/*.png"), recursive=True)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = read_image(image_path)  # This reads as a tensor with shape [C, H, W]

        # Ensure image is single-channel
        if image.size(0) != 1:
            raise ValueError(
                f"Expected single-channel image, got {image.size(0)} channels."
            )

        # Convert to float tensor
        image = image.float()

        # Extract label from the path
        label = int(os.path.basename(os.path.dirname(image_path)))

        if self.transform:
            image = self.transform(image)

        return image, label
