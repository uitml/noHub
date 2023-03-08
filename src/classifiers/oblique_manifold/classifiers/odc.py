import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from classifiers.oblique_manifold.manifolds import ManifoldParameter


class ODC(nn.Module):

    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.manifold = kwargs['manifold']
        # self.scale_factor = cfg.scale_factor
        self.scale_factor = cfg.scale_factor
        self.anchor = ManifoldParameter(data=kwargs['anchor'],
                                        manifold=self.manifold, requires_grad=True)

        self.proto = ManifoldParameter(data=kwargs['proto'],
                                       manifold=self.manifold, requires_grad=True)

        self.log_weight = self.ft(torch.arange(0, self.cfg.T + 1, dtype=torch.float)).to(device=config.DEVICE)
        pass

    def ft(self, t):
        if self.cfg.T == 0:
            r = t / (1 + self.cfg.T)
        else:
            r = t / self.cfg.T
        return ((1 - r * (2 * r - 1) ** 2) + 0.1) / 1.1 * 0.9

    def forward(self, x):

        # tangent space
        tan = self.manifold.logmap(x.unsqueeze(0), self.anchor.unsqueeze(1))
        proto = self.manifold.logmap(self.proto.unsqueeze(0), self.anchor.unsqueeze(1))

        tan = tan.view(*tan.size()[:2], -1)
        proto = proto.view(*proto.size()[:2], -1)

        sup = tan[:, :self.cfg.num_support, ...]
        que = tan[:, self.cfg.num_support:, ...]

        # y_train = [0 for _ in range(self.cfg.n_way)] + [1 for _ in range(self.cfg.T + 1)] + \
        #           [2 + _ % self.cfg.n_way for _ in range(self.cfg.num_query)]
        # trans = umap.UMAP().fit(data.reshape(data.shape[0], -1).detach().cpu().numpy())
        # plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], c=y_train, cmap='Spectral', s=8)

        dist = torch.cat([
            self.cal_dist(sup, proto),
            self.cal_dist(que, proto),
        ], dim=1) / (self.anchor.size()[1] * self.anchor.size()[2] / self.cfg.out_dim)

        logits = -dist * self.scale_factor
        if self.cfg.T == 0:
            logits = torch.cat([logits[:, :self.cfg.num_support, ...], logits[:, self.cfg.num_support:, ...]],
                               1)
        else:
            logits = torch.cat([self.log_weight.unsqueeze(-1).unsqueeze(-1) * logits[:, :self.cfg.num_support, ...],
                                (1 - self.log_weight).unsqueeze(-1).unsqueeze(-1) * logits[:, self.cfg.num_support:,
                                                                                    ...]],
                               1)
        logits = logits.sum(0) / self.log_weight.sum(0)
        return logits

    def cal_dist(self, x, w):
        """
        return Euclidean distance
        :param x: n * p
        :param w: m * p
        :return: n * m
        """
        # x = F.normalize(x, p=2, dim=-1)
        # w = F.normalize(w, p=2, dim=-1)

        if len(x.size()) == 2:
            return 1 / 2 * (w ** 2).sum(-1).unsqueeze(0) + 1 / 2 * (x ** 2).sum(-1).unsqueeze(1) - x.mm(
                w.transpose(-1, -2))
        else:
            return 1 / 2 * (w ** 2).sum(-1).unsqueeze(1) + 1 / 2 * (x ** 2).sum(-1).unsqueeze(2) - x.bmm(
                w.transpose(-1, -2))


class ODCPretrain(nn.Module):

    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.manifold = kwargs['manifold']
        self.scale_factor = 1
        self.anchor = ManifoldParameter(data=kwargs['anchor'],
                                        manifold=self.manifold, requires_grad=True)

        self.proto = ManifoldParameter(data=kwargs['proto'],
                                       manifold=self.manifold, requires_grad=True)

        self.log_weight = self.ft(torch.arange(0, self.cfg.T + 1, dtype=torch.float)).to(device=config.DEVICE)

    def ft(self, t):
        if self.cfg.T == 0:
            r = t / (1 + self.cfg.T)
        else:
            r = t / self.cfg.T
        return 1 - r * (2 * r - 1) ** 2

    def forward(self, x):

        # tangent space
        tan = self.manifold.logmap(x.unsqueeze(0), self.anchor.unsqueeze(1))
        proto = self.manifold.logmap(self.proto.unsqueeze(0), self.anchor.unsqueeze(1))

        tan = tan.view(*tan.size()[:2], -1)
        proto = proto.view(*proto.size()[:2], -1)

        sup = tan[:, :self.cfg.num_support, ...]
        que = tan[:, self.cfg.num_support:, ...]

        # y_train = [0 for _ in range(self.cfg.n_way)] + [1 for _ in range(self.cfg.T + 1)] + \
        #           [2 + _ % self.cfg.n_way for _ in range(self.cfg.num_query)]
        # trans = umap.UMAP().fit(data.reshape(data.shape[0], -1).detach().cpu().numpy())
        # plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], c=y_train, cmap='Spectral', s=8)

        dist = torch.cat([
            self.cal_dist(sup, proto),
            self.cal_dist(que, proto),
        ], dim=1) / (self.anchor.size()[1] * self.anchor.size()[2] / self.cfg.out_dim)

        logits = -dist * self.scale_factor

        logits = torch.cat([logits[:, :self.cfg.num_support, ...], logits[:, self.cfg.num_support:, ...]],
                           1)

        logits = logits.sum(0) / self.log_weight.sum(0)
        return logits

    def cal_dist(self, x, w):
        """
        return Euclidean distance
        :param x: n * p
        :param w: m * p
        :return: n * m
        """
        # ablation study for normalize
        x = F.normalize(x, p=2, dim=-1)
        w = F.normalize(w, p=2, dim=-1)

        if len(x.size()) == 2:
            return 1 / 2 * (w ** 2).sum(-1).unsqueeze(0) + 1 / 2 * (x ** 2).sum(-1).unsqueeze(1) - x.mm(
                w.transpose(-1, -2))
        else:
            return 1 / 2 * (w ** 2).sum(-1).unsqueeze(1) + 1 / 2 * (x ** 2).sum(-1).unsqueeze(2) - x.bmm(
                w.transpose(-1, -2))
