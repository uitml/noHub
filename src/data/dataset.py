import numpy as np
import torch as th
from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder

import config
from data import transform


def get_transform(size, aug=False):
    if aug:
        tfm = transform.with_augment(size, disable_random_resize=False, jitter=True)
    else:
        tfm = transform.without_augment(size, enlarge=True)
    return tfm


def get_dataset(name, split, aug=False):
    if name == "mini":
        tfm = get_transform(size=84, aug=aug)
        dataset = ImageFolder(config.DATA_DIR / "mini_imagenet" / split, transform=tfm)

    elif name == "tiered":
        tfm = get_transform(size=84, aug=aug)
        dataset = ImageFolder(config.DATA_DIR / "tiered_imagenet" / split, transform=tfm)

    elif name == 'cub':
        tfm = transform.without_augment(84, enlarge=True)
        dataset = ImageFolder(config.DATA_DIR / "cub" / split, transform=tfm)

    elif name == "debug":
        npz = np.load(config.DATA_DIR / "debug" / f"{split}.npz")
        data = th.from_numpy(npz["data"])
        labels = th.from_numpy(npz["labels"])
        dataset = TensorDataset(data, labels)

    else:
        raise RuntimeError(f"Unknown dataset name: '{name}'")

    return dataset
