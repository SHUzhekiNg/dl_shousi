import torch
import torch.nn as nn
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
