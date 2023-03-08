import torch.nn as nn


class FC(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        in_features = cfg.out_dim
        out_features = cfg.num_class
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        logits = self.fc(x)
        return logits
