import logging
import torch as th

import config
from models.arch.ResNet import *
from models.arch.DenseNet import *
from models.arch.Conv4 import Conv4
from models.arch.MobileNet import MobileNet
from models.arch.WideResNet import wideres
from models.arch.debug_model import debug
from models.arch.s2m2.wide_resnet import wrn_s2m2


logger = logging.getLogger(__name__)


MODELS = {
    "resnet10": resnet10,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "densenet121": densenet121,
    "densenet169": densenet169,
    "densenet201": densenet201,
    "densenet161": densenet161,
    "conv4": Conv4,
    "mobilenet": MobileNet,
    "wideres": wideres,
    "debug": debug,
    "wrn_s2m2": wrn_s2m2,
}


def get_model(arch, checkpoint_file, dataset_name):
    if dataset_name == "debug":
        n_classes = 50
    elif dataset_name == "mini":
        n_classes = 64
    elif dataset_name == "tiered":
        n_classes = 351
    elif dataset_name == "cub":
        n_classes = 100
    else:
        raise RuntimeError(f"Unknown dataset name: '{dataset_name}'.")

    model = MODELS[arch](num_classes=n_classes, remove_linear=False)
    model = th.nn.DataParallel(model)
    model = model.to(config.DEVICE)

    if arch != "debug" and checkpoint_file is not None:
        # Load weights
        checkpoint_path = config.MODELS_DIR / checkpoint_file
        logger.info(f"Loading weights from file: '{str(checkpoint_path)}'")
        checkpoint = th.load(checkpoint_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["state_dict"])

    return model
