import logging
import torch as th
from torch import nn

import config
from classifiers.simpleshot import initial_weights

logger = logging.getLogger(__name__)


@config.record_defaults(prefix="siamese")
class SIAMESE(nn.Module):
    def __init__(self, support_features, support_labels, n_ways, n_shots, n_queries, lam=10, alpha=0.2, epsilon=1e-3,
                 optimal_transport_max_iter=1000, n_iter=20):
        super(SIAMESE, self).__init__()
        self.n_episodes = support_features.size(0)
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = n_queries

        self.lam = lam
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.optimal_transport_max_iter = optimal_transport_max_iter

        self.register_buffer(
            name="prototypes", tensor=initial_weights(support_features, support_labels, normalize=False)
        )

    def update_from_estimate(self, estimate):
        self.prototypes += self.alpha * (estimate - self.prototypes)

    def compute_optimal_transport(self, dist, support_labels, r, c):
        n_runs, n, m = dist.shape
        n_support = support_labels.size(1)

        p = th.exp(- self.lam * dist)
        p /= p.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)

        u = th.zeros(n_runs, n, device=config.DEVICE)
        iters = 1
        # normalize this matrix
        while th.max(th.abs(u - p.sum(2))) > self.epsilon:
            u = p.sum(2)
            p *= (r / u).view((n_runs, -1, 1))
            p[:, :n_support].fill_(0)
            p[:, :n_support].scatter_(2, support_labels.unsqueeze(2), 1)
            p *= (c / p.sum(1)).view((n_runs, 1, -1))
            p[:, :n_support].fill_(0)
            p[:, :n_support].scatter_(2, support_labels.unsqueeze(2), 1)
            if iters >= self.optimal_transport_max_iter:
                logger.warning(f"Convergence warning: SIAMESE.compute_optimal_transport did not converge in "
                               f"{self.optimal_transport_max_iter} iterations.")
                break
            iters += 1
        return p

    def get_probs(self, features, support_labels):
        # compute squared dist to centroids [n_runs][n_samples][n_ways]
        dist = (features.unsqueeze(2) - self.prototypes.unsqueeze(1)).norm(dim=3).pow(2)

        r = th.ones(self.n_episodes, self.n_ways * (self.n_shots + self.n_queries), device=config.DEVICE)
        c = th.ones(self.n_episodes, self.n_ways, device=config.DEVICE) * (self.n_queries + self.n_shots)

        p_xj = self.compute_optimal_transport(dist, support_labels, r, c)
        return p_xj

    def estimate_from_mask(self, mask, features, dis=False):
        if dis:
            mask = th.zeros_like(mask, device=config.DEVICE).scatter_(2, mask.argmax(dim=-1).unsqueeze(-1), 1.)
        emus = mask.permute(0, 2, 1).matmul(features) / (self.n_queries + self.n_shots)
        return emus

    def perform_epoch(self, features, support_labels):
        probs = self.get_probs(features, support_labels)
        prototype_estimates = self.estimate_from_mask(probs, features)

        # update centroids
        self.update_from_estimate(prototype_estimates)
        return probs

    def fit_predict(self, support_features, support_labels, query_features):
        features = th.cat((support_features, query_features), dim=1)

        probs = None
        for it in range(self.n_iter):
            probs = self.perform_epoch(features, support_labels)

        pred = probs.argmax(dim=2)[:, (self.n_ways * self.n_shots):]
        assert pred.size() == (self.n_episodes, self.n_ways * self.n_queries)
        return pred