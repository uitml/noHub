import torch as th
from torch import nn
from argparse import Namespace

import config
from classifiers.simpleshot import initial_weights
from .manifolds import Oblique
from .classifiers import ODC
from .optimizers import RiemannianAdam
from .utils import setup, get_meta_data, clean_up, compute_loss_acc, set_seed, reduce_tensor, smooth_one_hot


@config.record_defaults(prefix="om")
class ObliqueManifoldClassifier(nn.Module):
    def __init__(self, n_ways, n_shots, t=14, lr_weights=0.1, lr_anchors=0.1, train_meta_epochs=40,
                 loss_weights=(1, 10, 1), scale_factor=15):
        super(ObliqueManifoldClassifier, self).__init__()

        self.n_ways = n_ways
        self.n_shots = n_shots
        self.manifold = Oblique()

        self.t = t
        self.lr_weights = lr_weights
        self.lr_anchors = lr_anchors
        self.train_meta_epochs = train_meta_epochs
        self.loss_weights = loss_weights
        self.scale_factor = scale_factor

    def forward(self):
        pass

    def _get_odc_cfg(self, n_support, out_dim):
        return Namespace(
            num_support=n_support,
            scale_factor=self.scale_factor,
            T=self.t,
            out_dim=out_dim,
        )

    def _get_anchor(self, support_features, query_features):
        n_support, n_query = support_features.size(0), query_features.size(0)
        anchor = [support_features.mean(dim=0, keepdim=True)]
        query_sum = query_features.sum(0, keepdim=True)
        support_sum = support_features.sum(0, keepdim=True)
        for t in range(1, self.t + 1):
            anchor.append(((self.t - t) * query_sum + t * support_sum) / ((self.t - t) * n_support + t * n_query))

        if len(anchor) > 1:
            anchor = th.cat(anchor, dim=0)
        else:
            anchor = anchor[0]

        return anchor

    def _fit_predict_episode(self, support_features, support_labels, query_features):
        n_support, out_dim = support_features.size()
        features = th.cat((support_features, query_features), dim=0)

        # Support prototypes
        sup_prototype = initial_weights(support_features[None, ...], support_labels[None, ...], normalize=False)[0]
        # Anchors
        anchor = self._get_anchor(support_features, query_features)
        # Add extra axis to pretend we did RSMA
        features, sup_prototype, anchor = features[:, None, :], sup_prototype[:, None, :], anchor[:, None, :]

        projection = self.manifold.proj(features)  # project to manifold
        sup_prototype = self.manifold.proj(sup_prototype)  # project to manifold
        anchor = self.manifold.proj(anchor)

        que_proj = projection[n_support:, ...]  # (num_query, out_dim, out_dim )
        sup_proj = projection[:n_support, ...]  # (num_support, out_dim, out_dim)

        sup_fea = sup_proj.clone().detach()
        que_fea = que_proj.clone().detach()

        # init weights and layers
        cfg = self._get_odc_cfg(n_support=n_support, out_dim=support_features.size(1))
        fc = ODC(cfg, manifold=self.manifold, anchor=anchor, proto=sup_prototype).to(device=config.DEVICE)

        optimizer = RiemannianAdam(
            [{'params': fc.proto, 'lr': self.lr_weights}, {'params': fc.anchor, 'lr': self.lr_anchors}],
        )
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(0.1 * self.train_meta_epochs), eta_min=1e-9)

        for _ in range(0, self.train_meta_epochs):
            fc.train()

            logits = fc(th.cat([sup_fea, que_fea], 0))

            sup_logits = logits[:n_support, ...] - 0 * smooth_one_hot(support_labels, classes=self.n_ways, smoothing=0)
            que_logits = logits[n_support:, ...]

            # compute loss and acc
            sup_loss, sup_acc = compute_loss_acc(sup_logits, support_labels, num_class=self.n_ways, smoothing=0)

            if cfg.T == 0:
                loss = self.loss_weights[0] * sup_loss
            else:
                que_probs = que_logits.softmax(-1)
                que_cond_ent = -(que_probs * th.log(que_probs + 1e-12)).sum(-1).mean(0)
                que_ent = -(que_probs.mean(0) * th.log(que_probs.mean(0))).sum(-1)
                loss = self.loss_weights[0] * sup_loss \
                    - (self.loss_weights[1] * que_ent - self.loss_weights[2] * que_cond_ent)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        query_pred = que_logits.argmax(dim=1).detach()
        return query_pred

    def fit_predict(self, support_features, support_labels, query_features):
        query_predictions = []
        for sf, sl, qf in zip(support_features, support_labels, query_features):
            qp = self._fit_predict_episode(sf, sl, qf)
            query_predictions.append(qp)
        return th.stack(query_predictions, dim=0)
