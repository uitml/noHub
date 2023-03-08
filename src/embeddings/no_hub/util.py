import logging
import torch as th
from torch import nn

import config
import helpers

logger = logging.getLogger(__name__)


@th.no_grad()
def ent_beta(dist, beta):
    beta = beta[:, :, None]
    p = th.exp(-dist * beta)
    p_sum = p.sum(dim=2, keepdim=True)
    ent = th.log(p_sum) + beta * th.sum(dist * p, dim=2, keepdim=True) / p_sum
    p = p / p_sum
    return ent.squeeze(), p


@th.no_grad()
def x2p(x, perplexity, sim="rbf", betas=(None, None), max_iter=20, rel_tol=1e-2, abs_tol=None, eps=1e-12,
        return_dist_and_betas=False):
    n_episodes, n_samples, _ = x.shape
    if sim == "rbf":
        dist = th.cdist(x, x)
        max_dist = 1e12
    elif sim == "vmf":
        x_norm = nn.functional.normalize(x, dim=2, p=2)
        dist = - (x_norm @ x_norm.transpose(1, 2))
        max_dist = 1
    elif sim == "precomputed":
        dist = x
        max_dist = dist.max()
    else:
        raise RuntimeError(f"Unknown similarity function in x2p: '{sim}'")

    rang = th.arange(n_samples)
    dist[:, rang, rang] = max_dist

    log_u = th.log(th.tensor(perplexity, device=config.DEVICE))
    # Compute the tolerance. Either relative to log_u, or absolute.
    if rel_tol is not None:
        if abs_tol is not None:
            logger.warning(f"Argument 'abs_tol' in x2p is ignored when 'rel_tol' is not None.")
        tol = rel_tol * log_u
    else:
        assert abs_tol is not None, "Either 'rel_tol' or 'abs_tol' must be not None in x2p."
        tol = th.tensor(abs_tol, device=config.DEVICE)

    # Initialize variables
    beta = th.ones(n_episodes, n_samples, device=config.DEVICE)
    beta_min, beta_max = betas
    beta_min = th.full_like(beta, (beta_min if beta_min is not None else th.nan), device=config.DEVICE)
    beta_max = th.full_like(beta, (beta_max if beta_max is not None else th.nan), device=config.DEVICE)
    gt_mask = th.full_like(beta, True)
    lt_mask = th.full_like(beta, False)
    ent, p = ent_beta(dist, beta)

    for it in range(max_iter):
        ent_diff = ent - log_u
        gt_mask = ent_diff > tol
        lt_mask = ent_diff < -1 * tol

        if (not th.any(gt_mask)) and (not th.any(lt_mask)):
            logger.debug(f"Binary search converged in {it} iterations.")
            break

        if th.any(gt_mask):
            beta_min = th.where(gt_mask, beta, beta_min)
            beta = th.where(
                gt_mask,
                th.where(th.isnan(beta_max), beta * 2, (beta + beta_max) / 2),
                beta
            )

        if th.any(lt_mask):
            beta_max = th.where(lt_mask, beta, beta_max)
            beta = th.where(
                lt_mask,
                th.where(th.isnan(beta_min), beta / 2, (beta + beta_min) / 2),
                beta
            )

        ent, p = ent_beta(dist, beta)

    else:
        n_not_converged = int(helpers.npy(gt_mask.float().sum() + lt_mask.float().sum()))
        logger.warning(f"Convergence warning: x2p did not converge for {n_not_converged}/"
                       f"{beta.size(0) * beta.size(1)} samples.")

    if return_dist_and_betas:
        return dist, beta

    p = (p + p.transpose(1, 2))
    p = p / p.sum(dim=(1, 2), keepdims=True)
    p = th.maximum(p, eps)
    return p