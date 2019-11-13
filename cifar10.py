from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

import torch
from torchvision import datasets, transforms
from utils.config import FLAGS

def data_transforms():
    """get transform of dataset"""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2612)
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    test_transforms = val_transforms

    return train_transforms, val_transforms, test_transforms


def dataset(train_transforms, val_transforms, test_transforms):
    if not FLAGS.test_only:
        train_set = datasets.CIFAR10(FLAGS.dataset_dir, train=True, transform=train_transforms)
    else:
        train_set = None
    val_set = datasets.CIFAR10(FLAGS.dataset_dir, train=False, transform=val_transforms)
    test_set = None

    return train_set, val_set, test_set


def data_loader(train_set, val_set, test_set):
    if not FLAGS.test_only:
        train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=FLAGS.batch_size, shuffle=True,
                pin_memory=True, num_workers=FLAGS.data_loader_workers,
                drop_last=getattr(FLAGS, 'drop_last', False))
    else:
        train_loader = None
    val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=FLAGS.batch_size, shuffle=False,
            pin_memory=True, num_workers=FLAGS.data_loader_workers,
            drop_last=getattr(FLAGS, 'drop_last', False))
    test_loader = val_loader

    if train_loader is not None:
        FLAGS.data_size_train = len(train_loader.dataset)
    if val_loader is not None:
        FLAGS.data_size_val = len(val_loader.dataset)
    if test_loader is not None:
        FLAGS.data_size_test = len(test_loader.dataset)

    return train_loader, val_loader, test_loader
