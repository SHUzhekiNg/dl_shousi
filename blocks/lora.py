import torch

import torch.nn as nn


class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        
        # 初始化权重
        nn.init.kaiming_uniform_(self.lora_a.weight)
        nn.init.zeros_(self.lora_b.weight)
    
    def forward(self, x):
        return self.lora_b(self.lora_a(x))


class LinearWithLoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lora = LoRA(in_features, out_features, rank)
        self.alpha = alpha
    
    def forward(self, x):
        return self.linear(x) + self.alpha * self.lora(x)