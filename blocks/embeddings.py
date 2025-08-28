import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
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
    def __init__(self, embed_dim, max_seq_len=512):
        super(RoPE, self).__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim)))

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds maximum {self.max_seq_len}")

        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(1)
        freqs = torch.einsum("i,j->ij", position_ids, self.inv_freq)
        freqs = freqs.unsqueeze(0).expand(x.size(0), -1, -1)

        cos = freqs.cos()
        sin = freqs.sin()

        x_cos = x * cos
        x_sin = x * sin

        return torch.cat((x_cos, x_sin), dim=-1)