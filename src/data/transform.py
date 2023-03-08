# Adapted from: https://github.com/mboudiaf/TIM/blob/master/src/datasets/transform.py
import torch
import torchvision.transforms as transforms
from PIL import ImageEnhance

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

TRANSFORM_TYPE_DICT = dict(
    Brightness=ImageEnhance.Brightness,
    Contrast=ImageEnhance.Contrast,
    Sharpness=ImageEnhance.Sharpness,
    Color=ImageEnhance.Color
)


class ImageJitter:
    def __init__(self, transform_dict):
        self.transforms = [(TRANSFORM_TYPE_DICT[k], transform_dict[k]) for k in transform_dict]

    def __call__(self, img):
        out = img
        rand_tensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(rand_tensor[i]*2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out


def without_augment(size=84, enlarge=False):
    if enlarge:
        resize = int(size*256./224.)
    else:
        resize = size
    return transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                NORMALIZE,
            ])


def with_augment(size=84, disable_random_resize=False, jitter=False):
    if disable_random_resize:
        return transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            NORMALIZE,
        ])
    else:
        if jitter:
            return transforms.Compose([
                transforms.RandomResizedCrop(size),
                ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                NORMALIZE,
            ])
        else:
            return transforms.Compose([
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                NORMALIZE,
            ])