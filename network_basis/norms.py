import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self,d_model, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        return self.gamma * x + self.beta

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor):
        norm = x.norm(2, dim=-1, keepdim=True)
        return x / (norm + self.eps) * self.gamma
    
class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-6):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        G = self.num_groups
        x = x.view(B, G, C // G, H, W)
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.view(B, C, H, W)
        return x * self.gamma + self.beta