import os
import logging
import torch as th
from tqdm.auto import tqdm

import config

logger = logging.getLogger(__name__)


def _features_file_path(cache_dir, split):
    return config.FEATURES_DIR / cache_dir / f"{split}_features.pt"


def _labels_file_path(cache_dir, split):
    return config.FEATURES_DIR / cache_dir / f"{split}_labels.pt"


@th.no_grad()
def extract_features(model, loader):
    model.eval()
    logger.info(f"Extracting features.")
    features, labels = [], []
    for inputs, lab in tqdm(loader):
        inputs = inputs.to(config.DEVICE)
        lab = lab.to(config.DEVICE)
        feat, _ = model(inputs, feature=True)

        features.append(feat)
        labels.append(lab)

    features = th.cat(features, dim=0)
    labels = th.cat(labels, dim=0)
    return features, labels


def load_features(cache_dir, split):
    features = th.load(_features_file_path(cache_dir, split), map_location=config.DEVICE)
    labels = th.load(_labels_file_path(cache_dir, split), map_location=config.DEVICE)
    logger.info(f"Successfully loaded cached features from '{config.FEATURES_DIR / cache_dir}'. "
                f"{features.shape=}, {labels.shape=}")
    return features, labels


def save_features(features, labels, cache_dir, split):
    os.makedirs(config.FEATURES_DIR / cache_dir, exist_ok=True)
    th.save(features, _features_file_path(cache_dir, split))
    th.save(labels, _labels_file_path(cache_dir, split))
    logger.info(f"Successfully saved features to '{cache_dir}'. {features.shape=}, {labels.shape=}")


def get_features(model, loader, cache_dir, use_cached, split):
    if use_cached:
        assert cache_dir is not None, "Cannot have cache_dir=None when using cached features."
        features, labels = load_features(cache_dir, split)
    else:
        features, labels = extract_features(model, loader)
        if cache_dir is not None:
            save_features(features, labels, cache_dir, split)

    # Ensure that features are float32.
    features = features.float()

    return features, labels
