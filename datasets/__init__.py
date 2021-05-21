import os
import torch
import numbers
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10
from datasets.celeba import CelebA
from torch.utils.data import Subset
import numpy as np


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


def get_dataset(args, config):
    tran_transform = transforms.Compose([
        transforms.Resize(config.data.image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),])
    test_transform = transforms.Compose([transforms.Resize(config.data.image_size), transforms.ToTensor()])

    if config.data.dataset == "CIFAR10":
        dataset = CIFAR10(
            os.path.join(args.exp, "datasets", "cifar10"),
            train=True,
            download=True,
            transform=tran_transform,
        )
        test_dataset = CIFAR10(
            os.path.join(args.exp, "datasets", "cifar10_test"),
            train=False,
            download=True,
            transform=test_transform,
        )

    elif config.data.dataset == "CELEBA":
        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64
        dataset = CelebA(
            root=os.path.join(args.exp, "datasets", "celeba"),
            split="train",
            transform=transforms.Compose(
                [
                    Crop(x1, x2, y1, y2),
                    transforms.Resize(config.data.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
            download=True,
        )

        test_dataset = CelebA(
            root=os.path.join(args.exp, "datasets", "celeba"),
            split="test",
            transform=transforms.Compose(
                [
                    Crop(x1, x2, y1, y2),
                    transforms.Resize(config.data.image_size),
                    transforms.ToTensor(),
                ]
            ),
            download=True,
        )

    else:
        dataset, test_dataset = None, None

    return dataset, test_dataset

def data_transform(config, X, **kwargs):
    if config.data.rescaled:
        X = X - 0.5
    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]
    return X


def inverse_data_transform(config, X, **kwargs):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]
    if config.data.rescaled:
        X = X + 0.5

    return torch.clamp(X, 0.0, 1.0)
