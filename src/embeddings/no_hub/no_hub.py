import logging

import config
from .no_hub_base import NoHubBase
from .loss import NoHubAlignLoss, NoHubUniformLoss

logger = logging.getLogger(__name__)


@config.record_defaults(prefix="nohub", ignore=["pca_weights"])
class NoHub(NoHubBase):
    def __init__(self, inputs, *, init="pca", out_dims=400, initial_dims=None, kappa=0.5, perplexity=45.0, re_norm=True,
                 eps=1e-12, p_sim="vmf", p_rel_tol=1e-2, p_abs_tol=None, p_betas=(None, None), pca_mode="base",
                 pca_weights=None, learning_rate=1e-1, n_iter=50, loss_weights=(0.2,)):

        if len(loss_weights) == 1:
            loss_weights = (loss_weights[0], 1 - loss_weights[0])

        super(NoHub, self).__init__(
            inputs, init=init, out_dims=out_dims, initial_dims=initial_dims, kappa=kappa, perplexity=perplexity,
            re_norm=re_norm, eps=eps, p_sim=p_sim, p_rel_tol=p_rel_tol, p_abs_tol=p_abs_tol, p_betas=p_betas,
            pca_mode=pca_mode, pca_weights=pca_weights, loss_weights=loss_weights, learning_rate=learning_rate,
            n_iter=n_iter
        )

    def init_losses(self):
        return [
            NoHubAlignLoss(kappa=self.kappa),
            NoHubUniformLoss(kappa=self.kappa)
        ]
