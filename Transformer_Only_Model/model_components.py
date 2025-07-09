import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1) 
        return self.dropout(x)
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

# Multi-head attention with LoRA (updated mask handling)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        assert d_model % nhead == 0
        assert int(d_model/nhead)%2==0
        self.d_k = d_model // nhead
        self.nhead = nhead

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(0.2)
       

    def forward(self, query, key, value, mask=None):
        B, L, D = query.size()

        def transform(x, linear):
            x = linear(x)
            # reshape for multihead: (B, L, nhead, d_k)
            return x.view(B, -1, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, L, d_k)

        Q = transform(query, self.q_linear)
        K = transform(key, self.k_linear)
        V = transform(value, self.v_linear)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, nhead, L_query, L_key)

        if mask is not None:
            # mask shape: (B, 1, L_query, L_key) or (B, nhead, L_query, L_key)
            # Expand mask to nhead dim if needed
            if mask.dim() == 4 and mask.size(1) == 1:
                mask = mask.expand(-1, self.nhead, -1, -1)
            scores = scores.masked_fill(~mask, float('-inf'))  # mask==False means masked out

        attn = self.dropout(torch.softmax(scores, dim=-1))
        context = attn @ V  # (B, nhead, L_query, d_k)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(context)

    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, mask):
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)  
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.cross_attn = MultiHeadAttention(d_model, nhead)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mem, tgt_mask, mem_mask):
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), mem, mem, mem_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x

# Mask helper functions
def generate_padding_mask(seq):
    # seq shape: (B, L)
    # True for tokens that are NOT padding (non-zero)
    return (seq != 0).unsqueeze(1).unsqueeze(2)  # (B,1,1,L)

def generate_subsequent_mask(size):
    # Creates upper-triangular matrix of True values, excluding diagonal
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()  # (L, L)
    return mask  # True means mask
