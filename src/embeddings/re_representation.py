import torch as th
from torch import nn

import config


@config.record_defaults(prefix="rr")
class ReRepresentation(nn.Module):
    def __init__(self, alpha_1=0.5, alpha_2=0.9, tau=0.1):
        super(ReRepresentation, self).__init__()
        
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.tau = tau
        self.big_num = 1e9

    def re_represent_queries(self, query_features):
        dists = th.cdist(query_features, query_features) ** 2
        logits = -1 * self.tau * dists

        rang = th.arange(logits.shape[1], device=config.DEVICE)
        logits[:, rang, rang] = -1 * self.big_num

        attention_weights = nn.functional.softmax(logits, dim=2)

        re_rep = (attention_weights[:, :, :, None] * query_features[:, :, None, :]).sum(dim=2)
        new_query_features = (1 - self.alpha_1) * query_features + self.alpha_1 * re_rep

        return new_query_features

    def re_represent_supports(self, support_features, query_features):
        dists = th.cdist(support_features, query_features) ** 2
        logits = -1 * self.tau * dists
        attention_weights = nn.functional.softmax(logits, dim=2)
        re_rep = (attention_weights[:, :, :, None] * query_features[:, None, :, :]).sum(dim=2)
        new_support_features = (1 - self.alpha_2) * support_features + self.alpha_2 * re_rep
        return new_support_features

    def forward(self, features, support_labels):
        n_support = support_labels.shape[1]
        support_features = features[:, :n_support]
        query_features = features[:, n_support:]

        query_features = self.re_represent_queries(query_features)
        support_features = self.re_represent_supports(support_features, query_features)

        out = th.cat((support_features, query_features), dim=1)
        return out
