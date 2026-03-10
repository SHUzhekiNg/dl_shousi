import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)

    def forward(self, x):
        out = self.weight[x]
        if self.padding_idx is not None:
            mask = (x == self.padding_idx).unsqueeze(-1)
            out = out.masked_fill(mask, 0.0)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)
        self.register_buffer('pe', self.pe)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RoPE(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, emb_size, 2).float() / emb_size))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        B, L, d = x.shape
        position_ids = torch.arange(L, device=x.device).float()
        freqs = torch.outer(position_ids, self.inv_freq)  # (L, d/2)
        cos = freqs.cos().unsqueeze(0)  # (1, L, d/2)
        sin = freqs.sin().unsqueeze(0)  # (1, L, d/2)
        x1, x2 = x[..., 0::2], x[..., 1::2]  # (B, L, d/2) each, 奇偶拆分
        out = torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)  # (B, L, d/2, 2)
        return out.reshape(B, L, d)