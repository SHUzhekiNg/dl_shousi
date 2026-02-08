# 这是示例代码，用于理解流程；实际工程需要更多优化。
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleExpert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, x):
        return self.net(x)

class Top1MoE(nn.Module):
    def __init__(self, dim, hidden_dim, n_experts, capacity_factor=1.0):
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([SimpleExpert(dim, hidden_dim) for _ in range(n_experts)])
        # gating: linear layer producing logits for experts
        self.gate = nn.Linear(dim, n_experts)

    def forward(self, x):
        # x: [batch, dim]
        batch, dim = x.shape
        logits = self.gate(x)                # [batch, n_experts]
        top1 = torch.argmax(logits, dim=-1)  # [batch]
        # one-hot selection
        dispatch_mask = F.one_hot(top1, num_classes=self.n_experts).float()  # [batch, n_experts]
        # compute gating weights (softmax over entire experts or use one-hot weight=1)
        # Here we use softmax over logits but then pick top1 weight only
        probs = F.softmax(logits, dim=-1)    # [batch, n_experts]
        top1_probs = probs.gather(1, top1.unsqueeze(1)).squeeze(1)  # [batch]

        # For efficiency we gather inputs per-expert
        outputs = x.new_zeros(batch, dim)
        for j in range(self.n_experts):
            mask_j = (top1 == j)
            if mask_j.sum() == 0:
                continue
            x_j = x[mask_j]                        # [n_j, dim]
            y_j = self.experts[j](x_j)            # [n_j, dim]
            # scale by gating weight for each token
            w_j = top1_probs[mask_j].unsqueeze(1) # [n_j, 1]
            outputs[mask_j] = y_j * w_j

        # optionally add load-balance aux loss: encourage uniform counts across experts
        # users typically compute it externally.

        return outputs

class TopkMoE(nn.Module):
    def __init__(self, dim, hidden_dim, n_experts, k=2, capacity_factor=1.0):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.experts = nn.ModuleList([SimpleExpert(dim, hidden_dim) for _ in range(n_experts)])
        self.gate = nn.Linear(dim, n_experts)

    def forward(self, x):
        batch, dim = x.shape
        logits = self.gate(x)                # [batch, n_experts]
        topk_probs, topk_indices = torch.topk(F.softmax(logits, dim=-1), self.k, dim=-1)  # [batch, k]

        outputs = x.new_zeros(batch, dim)
        for i in range(self.k):
            indices_i = topk_indices[:, i]  # [batch]
            probs_i = topk_probs[:, i]      # [batch]
            for j in range(self.n_experts):
                mask_j = (indices_i == j)
                if mask_j.sum() == 0:
                    continue
                x_j = x[mask_j]                        # [n_j, dim]
                y_j = self.experts[j](x_j)            # [n_j, dim]
                w_j = probs_i[mask_j].unsqueeze(1)    # [n_j, 1]
                outputs[mask_j] += y_j * w_j

        return outputs