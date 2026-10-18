"""Models take inputs in [0, 1] and normalize internally."""
import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class Normalized(nn.Module):
    def __init__(self, net, mean, std, resize_to=None):
        super().__init__()
        self.net = net
        self.resize_to = resize_to
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        if self.resize_to is not None:
            x = F.interpolate(x, size=self.resize_to, mode="bilinear", align_corners=False)
        return self.net((x - self.mean) / self.std)


def resnet18_cifar(num_classes=10):
    """ResNet-18 with a 3x3 stem and no max-pool (standard for 32x32 images)."""
    from torchvision.models import resnet18
    net = resnet18(num_classes=num_classes)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    return Normalized(net, CIFAR_MEAN, CIFAR_STD)


def deit_tiny(num_classes=10, pretrained=True):
    """ImageNet-pretrained DeiT-Tiny; 32x32 inputs are upsampled to 224 inside the model."""
    import timm
    net = timm.create_model("deit_tiny_patch16_224", pretrained=pretrained,
                            num_classes=num_classes)
    return Normalized(net, IMNET_MEAN, IMNET_STD, resize_to=224)


def build_model(name: str, **kw):
    if name == "resnet18":
        return resnet18_cifar(**kw)
    if name == "deit_tiny":
        return deit_tiny(**kw)
    raise ValueError(f"unknown model: {name}")
