import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, Attention

def sigmoid(x):
    return 1 / (1 - torch.exp(x))

def relu(x):
    return torch.clamp(x, min=0)

def softmax(x, dim):
    exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x:torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        return self.gamma * x + self.beta

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x:torch.Tensor):
        norm = x.norm(2, dim=-1, keepdim=True)
        return x / (norm + self.eps) * self.gamma
    
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DIT(nn.Module):
    def __init__(self, emb_size, num_heads, mlp_ratio=4.0):
        super(DIT, self).__init__()
        self.norm1 = nn.LayerNorm(emb_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(emb_size, elementwise_affine=False, eps=1e-6)
        self.attention = Attention(emb_size, num_heads, qkv_bias=True)
        mlp_hidden_size = int(emb_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=emb_size, hidden_features=mlp_hidden_size,  act_layer=approx_gelu)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_size, 6 * emb_size)
        )

    def forward(self, x, cond):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=1)
        x = x + self.attention(modulate(self.norm1(x), shift_msa, scale_msa)) * gate_msa.unsqueeze(1)
        x = x + self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)) * gate_mlp.unsqueeze(1)
        return x

class FinalLayer(nn.Module):
    def __init__(self, emb_size, patch_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(emb_size, elementwise_affine=True, eps=1e-6)
        self.linear = nn.Linear(emb_size, patch_size * patch_size * out_size)
        self.adaLN_modulation = nn.ModuleList(
            nn.SiLU(),
            nn.Linear(emb_size, 2 * emb_size)
        )

    def forward(self, x, cond):
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        x = self.norm_final(x)
        x = modulate(x, shift, scale)
        x = self.linear(x)
        return x
