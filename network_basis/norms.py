import torch
import torch.nn as nn

class BatchNorm1d(nn.Module):
    def __init__(self, num_feature, eps=1e-5, rho=0.1):
        super(BatchNorm1d, self).__init__()
        self.gamma = nn.Parameter(torch.ones(num_feature))
        self.beta = nn.Parameter(torch.zeros(num_feature))
        self.eps = eps
        self.rho = rho

        self.register_buffer("running_mean", torch.zeros(num_feature))
        self.register_buffer("running_var", torch.ones(num_feature))

    def forward(self, x:torch.Tensor):
        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True, unbiased=True)
            # 更新running_mean和running_var
            self.running_mean = self.rho * mean.squeeze(0) + (1 - self.rho) * self.running_mean
            self.running_var = self.rho * var.squeeze(0) + (1 - self.rho) * self.running_var
        else:
            mean = self.running_mean.view(1, -1)
            var = self.running_var.view(1, -1)

        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x + self.beta


class BatchNorm2d(nn.Module):
    def __init__(self, num_feature, eps=1e-5, rho=0.1):
        super(BatchNorm2d, self).__init__()
        self.gamma = nn.Parameter(torch.ones(num_feature))
        self.beta = nn.Parameter(torch.zeros(num_feature))
        self.eps = eps
        self.rho = rho

        self.register_buffer("running_mean", torch.zeros(num_feature))
        self.register_buffer("running_var", torch.ones(num_feature))

    def forward(self, x:torch.Tensor):
        if self.training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=True)
            # 更新running_mean和running_var
            self.running_mean = self.rho * mean.squeeze(0).squeeze(-1).squeeze(-1) + (1 - self.rho) * self.running_mean
            self.running_var = self.rho * var.squeeze(0).squeeze(-1).squeeze(-1) + (1 - self.rho) * self.running_var
        else:
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)

        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma.view(1, -1, 1, 1) * x + self.beta.view(1, -1, 1, 1)

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

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

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