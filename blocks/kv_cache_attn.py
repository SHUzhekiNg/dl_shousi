import torch
import torch.nn as nn


class KVCache:
    def __init__(self):
        self.k_cache = None
        self.v_cache = None

    def update(self, k, v):
        if self.k_cache is None:
            self.k_cache = k
            self.v_cache = v
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=2)
            self.v_cache = torch.cat([self.v_cache, v], dim=2)
        return self.k_cache, self.v_cache

    def reset(self):
        self.k_cache = None
        self.v_cache = None


class CachedMultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(emb_size, emb_size)
        self.k_proj = nn.Linear(emb_size, emb_size)
        self.v_proj = nn.Linear(emb_size, emb_size)
        self.out_proj = nn.Linear(emb_size, emb_size)

    def forward(self, x, kv_cache=None):
        B, N, C = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        score = (q @ k.transpose(-2, -1)) * self.scale

        seq_k = k.shape[2]
        seq_q = q.shape[2]
        mask = torch.triu(torch.ones(seq_q, seq_k, device=x.device), diagonal=seq_k - seq_q + 1).bool()
        score = score.masked_fill(mask, float('-inf'))

        attn = score.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, C)
        return self.out_proj(out)


if __name__ == "__main__":
    model = CachedMultiHeadAttention(emb_size=32, num_heads=4)
    model.eval()

    cache = KVCache()
    x_prefill = torch.randn(1, 8, 32)
    out_prefill = model(x_prefill, kv_cache=cache)
    print("Prefill output shape:", out_prefill.shape)

    x_decode = torch.randn(1, 1, 32)
    out_decode = model(x_decode, kv_cache=cache)
    print("Decode step 1 output shape:", out_decode.shape)
    print("Cache k shape:", cache.k_cache.shape)

    x_decode2 = torch.randn(1, 1, 32)
    out_decode2 = model(x_decode2, kv_cache=cache)
    print("Decode step 2 output shape:", out_decode2.shape)
    print("Cache k shape:", cache.k_cache.shape)
