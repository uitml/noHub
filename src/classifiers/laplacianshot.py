import logging
import torch as th
from torch import nn
import numpy as np

import config
import helpers
from classifiers.simpleshot import initial_weights
from wandb_utils import wandb_logger

logger = logging.getLogger(__name__)


@config.record_defaults(prefix="laplacianshot")
class LaplacianShot(nn.Module):
    def __init__(self, lam=0.7, knn=2, max_iter=50, rel_tol=1e-6, rectify_prototypes=True, symmetrize_affinity=True,
                 normalize_prototypes=True):
        super(LaplacianShot, self).__init__()

        self.lam = th.tensor(lam, device=config.DEVICE)
        self.knn = knn
        self.max_iter = max_iter
        self.rel_tol = th.tensor(rel_tol, device=config.DEVICE)
        self.eps = th.tensor(1e-20, device=config.DEVICE)
        self.rectify_prototypes = rectify_prototypes
        self.symmetrize_affinity = symmetrize_affinity
        self.normalize_prototypes = normalize_prototypes

    @th.no_grad()
    def knn_affinity(self, inputs, knn):
        n_episodes, n_samples, _ = inputs.size()
        dist = th.cdist(inputs, inputs)

        knn_idx = dist.topk(knn + 1, largest=False, sorted=True, dim=2)[1][:, :, 1:].ravel()
        episode_idx = th.arange(n_episodes, device=config.DEVICE)[:, None, None].expand(-1, n_samples, knn).ravel()
        sample_idx = th.arange(n_samples, device=config.DEVICE)[None, :, None].expand(n_episodes, -1, knn).ravel()

        aff = th.zeros((n_episodes, n_samples, n_samples), device=config.DEVICE)
        aff[episode_idx, sample_idx, knn_idx] = 1

        if self.symmetrize_affinity:
            aff = (aff + aff.transpose(1, 2)) / 2

        return aff

    @th.no_grad()
    def entropy_energy(self, y, a, b):
        temp = (a * y) - (self.lam * b * y)
        e = (y * th.log(th.maximum(y, self.eps)) + temp).sum(dim=(1, 2))
        return e

    @th.no_grad()
    def rectify_prototypes_and_queries(self, prototypes, support_features, query_features):
        if not self.rectify_prototypes:
            return prototypes, query_features

        # Check and impute nan values
        prototypes = self.check_nans(prototypes, "prototypes")
        support_features = self.check_nans(support_features, "support_features")
        query_features = self.check_nans(query_features, "query_features")

        # Shift query samples
        delta = support_features.mean(dim=1, keepdim=True) - query_features.mean(dim=1, keepdim=True)
        new_query_features = query_features + delta

        features = th.cat((support_features, new_query_features), dim=1)

        # Rectify prototypes
        logits = self.cosine_similarity(features, prototypes)

        pred = th.argmax(logits, dim=2)
        weights = nn.functional.softmax(10 * logits, dim=2)
        mask = th.eye(weights.size(2), device=config.DEVICE)[pred]
        weights *= mask
        new_prototypes = (weights[:, :, :, None] * features[:, :, None, :]).mean(1)

        if (proto_nan := th.isnan(new_prototypes)).any():
            raise RuntimeError(f"Got {helpers.npy(proto_nan.long().sum())} NaN-values in rectified prototypes.")

        if self.normalize_prototypes:
            new_prototypes = nn.functional.normalize(new_prototypes, p=2, dim=2)

        return new_prototypes, new_query_features

    @staticmethod
    def custom_softmax(inputs):
        # Custom softmax implementation based on `normalize` in
        # https://github.com/imtiazziko/LaplacianShot/blob/5de261226f6904379cff4cc5a1d183e309934809/src/lshot_update.py
        max_col = inputs.max(dim=2, keepdims=True).values
        inputs = inputs - max_col
        return nn.functional.softmax(inputs, dim=2)

    @staticmethod
    def cosine_similarity(x1, x2):
        x1_ = nn.functional.normalize(x1, dim=2, p=2)
        x2_ = nn.functional.normalize(x2, dim=2, p=2)
        cos_sim = (x1_[:, :, None, :] * x2_[:, None, :, :]).sum(dim=3)
        return cos_sim

    @staticmethod
    def check_nans(x, name, fill_value=0):
        if (nan_mask := th.isnan(x)).any():
            x = x.clone()
            x[nan_mask] = 0
            num_nans = helpers.npy(nan_mask.long().sum())
            tot = np.prod(x.size())
            logger.warning(f"Found {num_nans} of {tot} NaN values in {name}. Imputing with {fill_value}.")
        return x

    @th.no_grad()
    def fit_predict(self, support_features, support_labels, query_features, log_interval=2, log_wandb=True,
                    global_step=0):
        # Get prototypes
        prototypes = initial_weights(support_features, support_labels, normalize=self.normalize_prototypes)
        # Prototype rectification
        prototypes, query_features = self.rectify_prototypes_and_queries(prototypes, support_features, query_features)
        # Compute affinity
        kernel = self.knn_affinity(query_features, knn=self.knn)

        # Initialize
        neg_a = -1 * (th.cdist(query_features, prototypes) ** 2)
        y = self.custom_softmax(neg_a)
        prev_e = th.full((query_features.size(0),), th.inf, device=config.DEVICE)

        for i in range(self.max_iter):
            b = kernel @ y
            y = self.custom_softmax(neg_a + self.lam * b)
            e = self.entropy_energy(y, -1 * neg_a, b)

            if (i > 0) and (th.abs(e - prev_e) <= (self.rel_tol * th.abs(prev_e))).all():
                # Converged for all episodes
                logger.debug(f"LaplacianShot converged in {i} iterations. Mean final entropy energy: "
                             f"{helpers.npy(e).mean()}.")
                break
            else:
                prev_e = e.clone()

            if i % log_interval == 0:
                # Log to WandB and console
                _e = helpers.npy(e).mean()
                logger.debug(f"LaplacianShot: iter = {i} - Mean entropy energy = {_e}")
                if log_wandb:
                    wandb_logger.accumulate({"loss.laplacianshot": _e}, global_step=global_step, local_step=i,
                                            max_local_steps=self.max_iter)

            if th.isnan(y).any():
                raise RuntimeError(f"Got {th.isnan(y).long().sum()} NaN values in y.")

        else:
            logger.warning(f"ConvergenceWarning: LaplacianShot did not converge in {self.max_iter} iterations.")

        query_pred = y.argmax(dim=2)
        return query_pred


if __name__ == '__main__':
    import numpy as np
    from sklearn.datasets import make_blobs

    from data.episode_sampler import EpisodeSampler

    logging.basicConfig(level=logging.DEBUG)

    data, labels = make_blobs(n_samples=1000, centers=10, cluster_std=1.0)
    ep = EpisodeSampler(th.from_numpy(data.astype(np.float32)), th.from_numpy(labels),
                        n_episodes=1, n_ways=5, n_shots=5, n_queries=15)
    for (sf, sl), (qf, ql) in ep:
        ls = LaplacianShot(max_iter=10)
        sf = nn.functional.normalize(sf, dim=2)
        qf = nn.functional.normalize(qf, dim=2)
        qp = ls.fit_predict(sf, sl, qf)
        acc = (ql.numpy() == qp.numpy()).mean()
        print("Acc:", acc)
