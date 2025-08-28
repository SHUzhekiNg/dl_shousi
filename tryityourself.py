import torch
import torch.nn as nn
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        assert self.head_dim * num_heads == emb_size, "emb_size must be divisible by num_heads"

        self.qkv_proj = nn.Linear(emb_size, 3 * emb_size)
        self.out_proj = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        qkv = self.qkv_proj(x).view(batch_size, self.num_heads, -1, 3 * self.head_dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        score = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        if mask is not None:
            score = torch.masked_fill(score, mask, float("-inf"))
        
        weights = torch.softmax(score, dim=-1)
        context = torch.matmul(weights, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_size)
        out = self.out_proj(context)
        return out


