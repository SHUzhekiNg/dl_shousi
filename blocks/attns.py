import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, is_cross_attn=False):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        assert self.head_dim * num_heads == emb_size, "emb_size must be divisible by num_head"
        self.is_cross_attn = is_cross_attn
        self.enable_flash_attn = False
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(emb_size, emb_size *3)
        self.out_proj = nn.Linear(emb_size, emb_size)
        self.q = nn.Linear(emb_size, emb_size)
        self.k = nn.Linear(emb_size, emb_size)
        self.v = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, cond=None, casual_masked=True):
        B, N, C = x.shape
        if self.is_cross_attn:
            assert cond is not None, "Conditioning input must be provided for cross-attention"
        else:
            assert cond is None
            cond = x

        Bc, Nc, Cc = cond.shape
        assert B == Bc and C == Cc

        bias = None if self.qkv.bias is None else self.qkv.bias[:self.emb_size]
        q = F.linear(x, self.qkv.weight[:self.emb_size, :], bias)
        q = q.view(B, N, self.num_heads, self.head_dim)

        bias = None if self.qkv.bias is None else self.qkv.bias[self.emb_size:]
        kv = F.linear(cond, self.qkv.weight[self.emb_size:, :], bias)
        kv = kv.view(B, Nc, 2, self.num_heads, self.head_dim)

        k, v = kv.unbind(2)

        if not self.enable_flash_attn:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
        
        if self.enable_flash_attn:
            from flash_attn import flash_attn_func
            attn = flash_attn_func(q, k, v)

        else:
            q = q * self.scale
            score = (q @ k.transpose(-2, -1))
            if casual_masked:
                mask = torch.triu(torch.ones(N, N), diagonal=1).to(x.device).bool()
                score = torch.masked_fill(score, mask, float('-inf'))
            attn = score.softmax(dim=-1)
            attn = attn @ v
            attn = attn.transpose(1, 2).contiguous().view(B, N, C)
        x = self.out_proj(attn)
        return x

    def forward_2(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.q(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        score = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if mask is not None:
            score = torch.masked_fill(score, mask, float('-inf'))
        attn_weights = score.softmax(dim=-1)

        attn_output = (attn_weights @ v).transpose(1, 2).reshape(batch_size, -1, self.emb_dim)
        return self.out_proj(attn_output)


# 忽略了attention_mask、attention_dropout
class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head, nums_key_value_head):
        super().__init__()
        assert hidden_dim % nums_head == 0
        assert hidden_dim % nums_key_value_head == 0

        self.hidden_dim = hidden_dim
        self.nums_head = nums_head
        self.nums_key_value_head = nums_key_value_head
        self.head_dim = hidden_dim // nums_head

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, attention_mask=None):
        # x shape(batch_size, seq, hidden_dim)
        batch_size,seq,_ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # attention——weight目标是 (batch, nums_head, seq, seq)
        q = q.view(batch_size, seq, self.nums_head, self.head_dim)
        k = k.view(batch_size, seq, self.nums_key_value_head, self.head_dim)
        v = v.view(batch_size, seq, self.nums_key_value_head, self.head_dim)

        # 关注nums_head和nums_key_value_head的关系
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # k,v repeat:
        k = k.repeat_interleave(self.nums_head // self.nums_key_value_head, dim=1)
        v = v.repeat_interleave(self.nums_head // self.nums_key_value_head, dim=1)

        attention_score = (q @ k.transpose(2,3)) / (self.head_dim ** 0.5)
        attention_weight = torch.softmax(attention_score, dim=-1)
        #attention_mask忽略
        output = attention_weight @ v #(batch_size, nums_head, seq, head_dim)
        output = output.transpose(1,2).contiguous().view(batch_size,seq,-1)
        output = self.o_proj(output)
        return output


if __name__ == "__main__":
    x = torch.rand(2, 10, 32)  # (batch_size, seq_len, emb_size)
    net = MultiHeadAttention(32, 4)
    output = net(x)

    print(output.shape)  # Should be (2, 10, 32)