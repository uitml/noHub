import torch as th
from torch import nn


def initial_weights(support_features, support_labels, normalize=False):
    n_ways = len(th.unique(support_labels))
    one_hot = th.eye(n_ways, device=support_features.device)[support_labels][:, :, :, None]
    weights = (support_features[:, :, None, :] * one_hot).sum(dim=1) / one_hot.sum(dim=1)
    # Should have size (n_episodes, n_ways, dim)
    assert weights.size() == (support_features.size(0), n_ways, support_features.size(2))
    if normalize:
        weights = nn.functional.normalize(weights, dim=2, p=2)
    return weights


class SimpleShot(nn.Module):
    def __init__(self, support_features, support_labels):
        super(SimpleShot, self).__init__()
        self.register_buffer("weights", initial_weights(support_features, support_labels))

    def forward(self, query_features):
        # Nearest neighbor classification.
        dists = th.cdist(self.weights, query_features, p=2)
        predictions = th.argmin(dists, dim=1)
        assert predictions.size() == (query_features.size(0), query_features.size(1))
        return predictions
