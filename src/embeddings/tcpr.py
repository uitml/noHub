import logging
import torch as th
from torch import nn

import config

logger = logging.getLogger(__name__)


def get_cos_sin(x, y):
    cos_val = (x * y).sum(-1, keepdim=True) / th.norm(x, 2, -1, keepdim=True) / th.norm(y, 2, -1, keepdim=True)
    sin_val = (1 - cos_val * cos_val).sqrt()
    return cos_val, sin_val


def norm(x):
    return nn.functional.normalize(x, p=2, dim=-1)


@config.record_defaults(prefix="tcpr")
class TCPR(nn.Module):
    def __init__(self, k=15000):
        super(TCPR, self).__init__()

        self.k = k

    def get_base_approximation(self, base_features, support_features):
        # Make base features broadcastable with support_features
        base_features = base_features[None, :, :]

        base_features = norm(base_features)
        support_mean = th.mean(support_features, dim=1, keepdim=True)

        if (n_base := base_features.size(1)) < self.k:
            logger.warning(f"TCPR.k ({self.k}) > number of base features ({n_base}). "
                           f"Using k = 0.1 * number of base features.")
            k = int(0.1 * n_base)
        else:
            k = self.k

        similar = norm(support_mean) @ norm(base_features).transpose(1, 2)
        sim_cos, pred = similar[:, 0].topk(k, dim=1, largest=True, sorted=True)

        sim_weight_num = th.pow(sim_cos, 0.5)
        sim_weight = sim_weight_num / sim_weight_num.sum(dim=1, keepdim=True)

        approximation = th.sum(sim_weight[:, :, None] * base_features[0, pred, :], dim=1)[:, None, :]
        approximation = norm(approximation)

        return approximation

    def forward(self, features, base_features, support_labels):
        features = norm(features)

        support_features = features[:, :support_labels.size(1)]
        base_approx = self.get_base_approximation(base_features, support_features)

        cos_val, sin_val = get_cos_sin(features, base_approx)
        features = norm(features - cos_val * base_approx)
        return features
