import logging
import torch as th
from torch import nn

import config

logger = logging.getLogger(__name__)


@config.record_defaults(prefix="ease")
class EASE(nn.Module):
    def __init__(self, k=40, lam=100, low_rank=5, l2_normalize_outputs=True):
        super(EASE, self).__init__()

        self.k = k
        self.lam = lam
        self.low_rank = low_rank
        self.l2_normalize_outputs = l2_normalize_outputs

    @staticmethod
    def _l2_normalize(features, p=2):
        norms = features.norm(dim=p, keepdim=True)
        return features / norms

    @staticmethod
    def _qr_reduction(features):
        features = th.linalg.qr(features.permute(0, 2, 1)).R
        features = features.permute(0, 2, 1)
        return features

    def forward(self, features):
        features = self._l2_normalize(features)
        features = self._qr_reduction(features)  # to speed up svd,not necessary for the performance
        u, _, _ = th.svd(features)
        u = u[:, :, :self.low_rank]

        W = th.abs(u.matmul(th.transpose(u, dim0=2, dim1=1)))
        for i in range(W.shape[0]):
            W[i, :, :].squeeze().fill_diagonal_(0)
        isqrt_diag = 1. / th.sqrt(1e-4 + th.sum(W, dim=-1, keepdim=True))
        W = W * isqrt_diag * th.transpose(isqrt_diag, dim0=2, dim1=1)

        W1 = th.ones_like(W, device=config.DEVICE)
        for i in range(W1.shape[0]):
            W1[i, :, :].squeeze().fill_diagonal_(0)
        isqrt_diag = 1. / th.sqrt(1e-4 + th.sum(W1, dim=-1, keepdim=True))
        W1 = W1 * isqrt_diag * th.transpose(isqrt_diag, dim0=2, dim1=1)

        lapReg1 = th.transpose(features, dim0=2, dim1=1).matmul(self.lam * W1 - W).matmul(features)
        e, v1 = th.linalg.eigh(lapReg1)

        normed = features.matmul(v1[:, :, :self.k])
        if self.l2_normalize_outputs:
            normed = self._l2_normalize(normed)
        return normed
