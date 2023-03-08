import logging
import torch as th

import config
import helpers
from classifiers.tim.tim import TIM
from classifiers.tim.utils import get_one_hot
from wandb_utils import wandb_logger


logger = logging.getLogger(__name__)


@config.record_defaults(prefix="alpha_tim")
class AlphaTIM(TIM):
    def __init__(self, support, query, y_s, shot, temp=15, loss_weights=(1.0, 1.0, 1.0), n_iter=1000, lr=1e-4,
                 entropies=("shannon", "alpha", "alpha"), alpha_values=(5.0, 5.0, 5.0), use_tuned_alpha_values=True):
        super().__init__(support, query, y_s, temp, loss_weights, n_iter)

        self.shot = shot
        self.temp = temp
        self.loss_weights = loss_weights
        self.n_iter = n_iter
        self.lr = lr
        self.entropies = entropies
        self.alpha_values = alpha_values
        self.use_tuned_alpha_values = use_tuned_alpha_values

        if self.use_tuned_alpha_values or self.alpha_values is None:
            self.get_alpha_values(shot)

    def get_alpha_values(self, shot):
        if shot == 1:
            self.alpha_values = [2.0, 2.0, 2.0]
        elif shot >= 5:
            self.alpha_values = [7.0, 7.0, 7.0]

    def run_adaptation(self, support, query, y_s, log_interval=20, global_step=0, log_wandb=True):
        """
        Corresponds to the ALPHA-TIM inference
        inputs:
            support : th.Tensor of shape [n_task, s_shot, feature_dim]
            query : th.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : th.Tensor of shape [n_task, s_shot]
            y_q : th.Tensor of shape [n_task, q_shot]
        updates :
            self.weights : th.Tensor of shape [n_task, num_class, feature_dim]
        """
        logger.debug("Executing ALPHA-TIM adaptation over {} iterations on {} shot tasks ...".format(self.n_iter,
                                                                                                     self.shot))
        self.weights.requires_grad_()
        optimizer = th.optim.Adam([self.weights], lr=self.lr)
        y_s_one_hot = get_one_hot(y_s)
        # self.model.train()

        for i in range(self.n_iter):
            logits_s = self.get_logits(support)
            logits_q = self.get_logits(query)

            q_probs = logits_q.softmax(2)

            # Cross entropy type
            if self.entropies[0] == 'shannon':
                ce = - (y_s_one_hot * th.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1).sum(0)
            elif self.entropies[0] == 'alpha':
                ce = th.pow(y_s_one_hot, self.alpha_values[0]) * th.pow(logits_s.softmax(2) + 1e-12, 1 - self.alpha_values[0])
                ce = ((1 - ce.sum(2))/(self.alpha_values[0] - 1)).mean(1).sum(0)
            else:
                raise ValueError("Entropies must be in ['shannon', 'alpha']")

            # Marginal entropy type
            if self.entropies[1] == 'shannon':
                q_ent = - (q_probs.mean(1) * th.log(q_probs.mean(1))).sum(1).sum(0)
            elif self.entropies[1] == 'alpha':
                q_ent = ((1 - (th.pow(q_probs.mean(1), self.alpha_values[1])).sum(1)) / (self.alpha_values[1] - 1)).sum(0)
            else:
                raise ValueError("Entropies must be in ['shannon', 'alpha']")

            # Conditional entropy type
            if self.entropies[2] == 'shannon':
                q_cond_ent = - (q_probs * th.log(q_probs + 1e-12)).sum(2).mean(1).sum(0)
            elif self.entropies[2] == 'alpha':
                q_cond_ent = ((1 - (th.pow(q_probs + 1e-12, self.alpha_values[2])).sum(2)) / (self.alpha_values[2] - 1)).mean(1).sum(0)
            else:
                raise ValueError("Entropies must be in ['shannon', 'alpha']")

            # Loss
            loss = self.loss_weights[0] * ce - (self.loss_weights[1] * q_ent - self.loss_weights[2] * q_cond_ent)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % log_interval == 0:
                # Log to WandB and console
                _loss = helpers.npy(loss).mean()
                logger.debug(f"AlphaTIM: iter = {i} - Mean loss = {_loss}")
                if log_wandb:
                    wandb_logger.accumulate({"loss.alpha_tim": _loss}, global_step=global_step, local_step=i,
                                            max_local_steps=self.n_iter)
