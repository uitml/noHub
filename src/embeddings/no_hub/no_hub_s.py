import torch as th
from torch import nn

import config
from .no_hub_base import NoHubBase
from .loss import NoHubAlignLoss, NoHubUniformLoss, NoHubSUniformLoss
from .util import x2p


@config.record_defaults(prefix="nohubs", ignore=["pca_weights"])
class NoHubS(NoHubBase):
    def __init__(self, inputs, support_labels, *, init="pca", out_dims=400, initial_dims=None,
                 kappa=0.5, perplexity=40.0, re_norm=True, eps=1e-12, p_sim="vmf", p_rel_tol=1e-2, p_abs_tol=None,
                 p_betas=(None, None), pca_mode="base", pca_weights=None, loss_weights=(0.3,),
                 learning_rate=1e-1, n_iter=150, different_label_exaggeration=5, exaggeration_mode="linear",
                 modify_align_loss=True, modify_uniform_loss=True,
                 ):

        # Pre-compute masks for same/different labelled supports.
        n_samples = inputs.size(1)
        n_episodes, n_support = support_labels.size()
        same_label = support_labels[:, :, None] == support_labels[:, None, :]

        self.mask_same_label = th.full((n_episodes, n_samples, n_samples), False, device=config.DEVICE).bool()
        self.mask_same_label[:, :n_support, :n_support] = same_label

        self.mask_different_label = th.full((n_episodes, n_samples, n_samples), False, device=config.DEVICE).bool()
        self.mask_different_label[:, :n_support, :n_support] = th.logical_not(same_label)

        if len(loss_weights) == 1:
            # Assume weights are (loss_weights[0], 1 - loss_weights[0])
            loss_weights = (loss_weights[0], 1 - loss_weights[0])

        self.different_label_exaggeration = different_label_exaggeration
        self.exaggeration_mode = exaggeration_mode
        self.modify_align_loss = modify_align_loss
        self.modify_uniform_loss = modify_uniform_loss

        super(NoHubS, self).__init__(
            inputs=inputs, init=init, out_dims=out_dims, initial_dims=initial_dims, kappa=kappa, perplexity=perplexity,
            re_norm=re_norm, eps=eps, p_sim=p_sim, p_rel_tol=p_rel_tol, p_abs_tol=p_abs_tol, p_betas=p_betas,
            pca_mode=pca_mode, pca_weights=pca_weights, loss_weights=loss_weights, learning_rate=learning_rate,
            n_iter=n_iter
        )

    def _cosine_dist_with_labels(self):
        # Cosine distances
        inputs_normed = nn.functional.normalize(self.inputs, p=2, dim=2)
        dist = -1 * inputs_normed @ inputs_normed.transpose(1, 2)
        # Make same label supports as close as possible (ignoring self-similarities)
        dist[self.mask_same_label] = -1
        # Make different label supports as far away as possible
        dist[self.mask_different_label] = 1
        return dist

    def set_p(self):
        if not self.modify_align_loss:
            super(NoHubS, self).set_p()
            return

        # Modify the p-matrix such that:
        #   1. Supports from the same class should be maximally similar
        #   2. Supports from different classes should be maximally dissimilar.
        if self.p_sim == "vmf":
            dist = self._cosine_dist_with_labels()
        else:
            raise RuntimeError(f"Unsupported p_sim={self.p_sim}.")

        self.p = x2p(dist, perplexity=self.perplexity, sim="precomputed", rel_tol=self.p_rel_tol,
                     abs_tol=self.p_abs_tol, eps=self.eps, betas=self.p_betas)

    def init_losses(self):
        if self.modify_uniform_loss:
            uniformity_loss = NoHubSUniformLoss(
                kappa=self.kappa, different_label_exaggeration=self.different_label_exaggeration,
                exaggeration_mode=self.exaggeration_mode
            )
        else:
            uniformity_loss = NoHubUniformLoss(kappa=self.kappa)

        return [
            NoHubAlignLoss(kappa=self.kappa),
            uniformity_loss
        ]
