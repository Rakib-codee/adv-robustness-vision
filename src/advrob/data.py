"""CIFAR-10 loaders. Images stay in [0, 1]; normalization lives INSIDE the model
(see models.py) so attacks and epsilon are always defined in pixel space."""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10(root="data", batch_size=128, num_workers=2, image_size=32):
    train_tf = [transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip()]
    test_tf = []
    # Resizing for ViTs is done inside the model, so image_size stays 32 here.
    train_set = datasets.CIFAR10(root, train=True, download=True,
                                 transform=transforms.Compose(train_tf + [transforms.ToTensor()]))
    test_set = datasets.CIFAR10(root, train=False, download=True,
                                transform=transforms.Compose(test_tf + [transforms.ToTensor()]))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
