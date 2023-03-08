"""Adapted from: https://github.com/mboudiaf/TIM"""
import logging
from abc import ABC, abstractmethod
import torch
from torch import nn

import config
import helpers
from classifiers.tim.utils import get_one_hot
from classifiers.simpleshot import initial_weights
from wandb_utils import wandb_logger


logger = logging.getLogger(__name__)


class TIM(ABC, nn.Module):
    def __init__(self, support, query, y_s, temp, loss_weights, n_iter):
        super(TIM, self).__init__()
        
        self.temp = temp
        self.loss_weights = list(loss_weights)
        self.n_iter = n_iter

        # Initialize weights
        weights = initial_weights(support_features=support, support_labels=y_s)
        self.register_parameter(name="weights", param=nn.Parameter(weights, requires_grad=True))

        if self.loss_weights[0] is None:
            self.compute_lambda(support, query, y_s)

    def get_logits(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """
        n_tasks = samples.size(0)
        logits = self.temp * (samples.matmul(self.weights.transpose(1, 2))
                              - 1 / 2 * (self.weights**2).sum(2).view(n_tasks, 1, -1)
                              - 1 / 2 * (samples**2).sum(2).view(n_tasks, -1, 1))
        return logits

    def get_preds(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, s_shot, feature_dim]

        returns :
            preds : torch.Tensor of shape [n_task, shot]
        """
        logits = self.get_logits(samples)
        preds = logits.argmax(2)
        return preds

    def compute_lambda(self, support, query, y_s):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]

        updates :
            self.loss_weights[0] : Scalar
        """
        self.N_s, self.N_q = support.size(1), query.size(1)
        self.num_classes = torch.unique(y_s).size(0)
        self.loss_weights[0] = (1 + self.loss_weights[2]) * self.N_s / self.N_q

    @abstractmethod
    def run_adaptation(self, support, query, y_s):
        """
        Corresponds to the baseline (no transductive inference = SimpleShot)
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        pass

    def forward(self, inputs):
        logits = self.get_logits(inputs)
        return logits.argmax(dim=2)


@config.record_defaults(prefix="tim")
class TIMGD(TIM):
    def __init__(self, support, query, y_s, temp=15, loss_weights=(0.1, 1.0, 0.1), n_iter=1000, lr=1e-4):
        super().__init__(support=support, query=query, y_s=y_s, temp=temp, loss_weights=loss_weights, n_iter=n_iter)
        self.lr = lr

    def run_adaptation(self, support, query, y_s, log_interval=20, global_step=0, log_wandb=True):
        """
        Corresponds to the TIM-GD inference
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        y_s_one_hot = get_one_hot(y_s)
        for i in range(self.n_iter):
            logits_s = self.get_logits(support)
            logits_q = self.get_logits(query)

            ce = - (y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1).sum(0)
            q_probs = logits_q.softmax(2)
            q_cond_ent = - (q_probs * torch.log(q_probs + 1e-12)).sum(2).mean(1).sum(0)
            q_ent = - (q_probs.mean(1) * torch.log(q_probs.mean(1))).sum(1).sum(0)
            loss = self.loss_weights[0] * ce - (self.loss_weights[1] * q_ent - self.loss_weights[2] * q_cond_ent)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % log_interval == 0:
                # Log to WandB and console
                _loss = helpers.npy(loss).mean()
                logger.debug(f"TIM: iter = {i} - Mean loss = {_loss}")
                if log_wandb:
                    wandb_logger.accumulate({"loss.tim": _loss}, global_step=global_step, local_step=i,
                                            max_local_steps=self.n_iter)
