import torch
import torch.nn as nn
import torch.nn.functional as F
from .attns import MultiHeadAttention


##############################  TRANSFORMERS  ##############################
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, src):
        attn_output = self.self_attn(src, src, src)
        src = self.norm1(src + attn_output)
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = self.norm2(src + ff_output)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, tgt, memory):
        attn_output = self.self_attn(tgt, tgt, tgt)
        tgt = self.norm1(tgt + attn_output)
        cross_attn_output = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + cross_attn_output)
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + ff_output)
        return tgt
    

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.w_1(x)))
        x = self.w_2(x)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=512, num_heads=8,
                 feedforward_dim=2048, num_encoder_layers=6, num_decoder_layers=6,
                 max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim)

        # Positional Encoding
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_seq_len, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # Encoder & Decoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output projection
        self.fc_out = nn.Linear(embed_dim, tgt_vocab_size)

    def encode(self, src):
        # src: (batch, src_len)
        x = self.src_embedding(src) * (self.embed_dim ** 0.5)
        x = self.dropout(x + self.pe[:, :x.size(1)])
        for layer in self.encoder_layers:
            x = layer(x)
        return x

    def decode(self, tgt, memory):
        # tgt: (batch, tgt_len), memory: encoder output
        x = self.tgt_embedding(tgt) * (self.embed_dim ** 0.5)
        x = self.dropout(x + self.pe[:, :x.size(1)])
        for layer in self.decoder_layers:
            x = layer(x, memory)
        return x

    def forward(self, src, tgt):
        # src: (batch, src_len), tgt: (batch, tgt_len)
        memory = self.encode(src)
        output = self.decode(tgt, memory)
        output = self.fc_out(output)  # (batch, tgt_len, tgt_vocab_size)
        return output
