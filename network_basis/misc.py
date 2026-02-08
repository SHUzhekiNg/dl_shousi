import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(Dropout, self).__init__()
        assert 0 <= dropout_rate < 1, "dropout_rate must be in [0, 1)"
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if self.training and self.dropout_rate > 0:
            mask = (torch.rand_like(x) > self.dropout_rate).float()
            return (x * mask) / (1.0 - self.dropout_rate)
        return x