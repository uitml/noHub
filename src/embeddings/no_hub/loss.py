import torch as th
from torch import nn


class NoHubAlignLoss(nn.Module):
    def __init__(self, kappa):
        super(NoHubAlignLoss, self).__init__()
        self.kappa = kappa

    def forward(self, no_hub):
        embeddings = no_hub()
        loss = -1 * (self.kappa * no_hub.p * (embeddings @ embeddings.transpose(1, 2))).sum(dim=(1, 2))
        return loss.mean()


class NoHubUniformLoss(nn.Module):
    def __init__(self, kappa):
        super(NoHubUniformLoss, self).__init__()
        self.kappa = kappa

    def forward(self, no_hub):
        embeddings = no_hub()
        logits = self.kappa * (embeddings @ embeddings.transpose(1, 2))
        loss = th.logsumexp(logits, dim=(1, 2))
        return loss.mean()


class NoHubSUniformLoss(nn.Module):
    def __init__(self, kappa, different_label_exaggeration, exaggeration_mode):
        super(NoHubSUniformLoss, self).__init__()
        self.kappa = kappa
        self.big_num = 1e9
        self.different_label_exaggeration = different_label_exaggeration
        self.exaggeration_mode = exaggeration_mode

    def forward(self, no_hub):
        embeddings = no_hub()
        logits = embeddings @ embeddings.transpose(1, 2)

        logits[no_hub.mask_same_label] = -1 * self.big_num

        if self.exaggeration_mode == "linear":
            logits[no_hub.mask_different_label] *= self.different_label_exaggeration
        elif self.exaggeration_mode == "exp":
            logits[no_hub.mask_different_label] = th.exp(self.different_label_exaggeration *
                                                      logits[no_hub.mask_different_label])
        elif self.exaggeration_mode == "hyperbolic":
            x = logits[no_hub.mask_different_label]
            logits[no_hub.mask_different_label] = self.different_label_exaggeration * x / (1 - x)
        else:
            raise RuntimeError(f"Unknown exaggeration mode: '{self.exaggeration_mode}'.")

        loss = th.logsumexp(self.kappa * logits, dim=(1, 2))
        return loss.mean()