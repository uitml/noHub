import logging
from abc import ABC, abstractmethod
import torch as th
from torch import nn

import config
import helpers
from wandb_utils import wandb_logger
from pca import get_pca_weights
from .util import x2p


logger = logging.getLogger(__name__)


class NoHubBase(ABC, nn.Module):
    def __init__(self, inputs, *, init="pca", out_dims=32, initial_dims=None, kappa=2.0, perplexity=30.0, re_norm=True,
                 eps=1e-12, p_sim="vmf", p_rel_tol=1e-2, p_abs_tol=None, p_betas=(None, None), pca_mode="base",
                 pca_weights=None, loss_weights=None, learning_rate=1e-1, n_iter=50):
        super(NoHubBase, self).__init__()

        self.re_norm = re_norm
        self.eps = th.tensor(eps, device=config.DEVICE)
        self.kappa = kappa
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.perplexity = perplexity
        self.p_sim = p_sim
        self.p_rel_tol = p_rel_tol
        self.p_abs_tol = p_abs_tol
        self.eps = th.tensor(eps, device=config.DEVICE)
        self.p_betas = p_betas

        # Passing out_dims=None causes embeddings to have same dimensionality as inputs
        self.out_dims = out_dims if out_dims is not None else inputs.size(2)

        assert inputs.ndim == 3, f"NoHub expected inputs tensor with shape (episodes, samples, feature). " \
                                 f"Got: '{inputs.size()}.'"
        self.embedding_size = (inputs.size(0), inputs.size(1), self.out_dims)

        # Determine the pca-weights (transformation) based on the given pca_mode
        self.set_pca_weights(pca_mode=pca_mode, pca_weights=pca_weights, inputs=inputs)
        # Initialize embeddings
        self.init_embeddings(init=init, out_dims=self.out_dims, inputs=inputs)
        # Run PCA to pre-process inputs?
        self.preprocess_inputs(initial_dims=initial_dims, inputs=inputs)
        # Compute P-values
        self.set_p()

        # Initialize losses
        self.losses = self.init_losses()
        # Loss weights
        self.set_loss_weights(loss_weights, len(self.losses))

    def set_pca_weights(self, pca_mode, pca_weights, inputs):
        if pca_mode == "base":
            assert pca_weights is not None, "pca_mode='base' requires pca_weights to be not None."
            self.pca_weights = pca_weights
        elif pca_mode == "episode":
            if pca_weights is not None:
                logger.warning(f"Argument 'pca_weights' is ignored when pca_mode='episode'.")
            self.pca_weights = get_pca_weights(inputs)

    def init_embeddings(self, init, out_dims, inputs):
        if init == "random":
            initial_embeddings = th.randn(size=self.embedding_size, device=config.DEVICE)
        elif init == "pca":
            initial_embeddings = inputs @ self.pca_weights[:, :, :out_dims]
        else:
            raise RuntimeError(f"Unknown init strategy for NoHub: '{init}'.")

        initial_embeddings = nn.functional.normalize(initial_embeddings, dim=-1, p=2)
        self.register_parameter(name="embeddings", param=nn.Parameter(initial_embeddings, requires_grad=True))

    def preprocess_inputs(self, initial_dims, inputs):
        if initial_dims is None:
            self.inputs = inputs
        else:
            self.inputs = inputs @ self.pca_weights[:, :, :initial_dims]

    def set_loss_weights(self, loss_weights, n_loss_terms):
        if loss_weights is None:
            self.loss_weights = n_loss_terms * [1 / n_loss_terms]
        else:
            assert len(loss_weights) == n_loss_terms, f"Expected loss weights to have same length as losses. " \
                                                      f"Got {len(loss_weights)} != {n_loss_terms}."
            self.loss_weights = loss_weights

    def set_p(self):
        self.p = x2p(self.inputs, perplexity=self.perplexity, sim=self.p_sim, rel_tol=self.p_rel_tol,
                     abs_tol=self.p_abs_tol, eps=self.eps, betas=self.p_betas)

    @abstractmethod
    def init_losses(self):
        # Should return a list of loss modules.
        pass

    @th.no_grad()
    def update_embeddings(self, new_embeddings):
        self.embeddings.copy_(new_embeddings)

    def forward(self):
        return self.embeddings

    def loss(self):
        return sum([weight * loss(self) for weight, loss in zip(self.loss_weights, self.losses)])

    def train_step(self, optimizer):
        optimizer.zero_grad()
        embeddings = self()
        loss = self.loss()
        loss.backward()
        optimizer.step()

        if self.re_norm:
            normed_embeddings = nn.functional.normalize(embeddings, dim=2, p=2)
            self.update_embeddings(normed_embeddings)

        return loss.detach()


def train_no_hub(no_hub, log_interval=10, profiler=None, global_step=0, log_wandb=True):
    opt = th.optim.Adam(params=no_hub.parameters(), lr=no_hub.learning_rate, betas=(0.9, 0.999))
    losses = th.zeros(no_hub.n_iter, device=config.DEVICE)

    for i in range(no_hub.n_iter):
        loss = no_hub.train_step(optimizer=opt)
        losses[i] = loss

        if i % log_interval == 0:
            # Log to WandB and console
            _loss = helpers.npy(loss)
            logger.debug(f"NoHub-iter = {i} - Loss = {_loss}")
            if log_wandb:
                wandb_logger.accumulate({"loss.NoHub": _loss}, global_step=global_step, local_step=i,
                                        max_local_steps=no_hub.n_iter)

        if profiler is not None:
            profiler.step()

    if th.any(th.isnan(losses[-1])):
        logger.warning(f"NoHub resulted in nan loss.")
    else:
        logger.debug(f"NoHub final loss = {losses[-1]}")

    no_hub.eval()
    embeddings = no_hub()
    return embeddings.detach(), losses

