import torch
import torch.nn as nn
import math


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=1024):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        self.register_buffer("rotary_emb", emb)  # (max_len, dim)

    def forward(self, q, k):
        # q: (..., T, dim), k: (..., S, dim)
        seq_len_q = q.size(-2)
        seq_len_k = k.size(-2)

        # Slice separately
        rot_q = self.rotary_emb[:seq_len_q].to(q.device)  # (T, dim)
        rot_k = self.rotary_emb[:seq_len_k].to(k.device)  # (S, dim)

        # Shape to (1,1,seq,dim)
        rot_q = rot_q.unsqueeze(0).unsqueeze(0)
        rot_k = rot_k.unsqueeze(0).unsqueeze(0)

        q_rot = self.apply_rotary(q, rot_q)
        k_rot = self.apply_rotary(k, rot_k)
        return q_rot, k_rot

    @staticmethod
    def apply_rotary(x, rotary):
        x1, x2 = x[..., ::2], x[..., 1::2]
        sin, cos = rotary[..., ::2], rotary[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin,
                          x1 * sin + x2 * cos], dim=-1)

    
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
class MultiHeadAttention_RoPE(nn.Module):
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
        self.rotary_pe = RotaryPositionalEmbedding(dim=self.d_k, max_len=1024)

    def forward(self, query, key, value, mask=None):
        B, L, D = query.size()

        def transform(x, linear):
            x = linear(x)
            # reshape for multihead: (B, L, nhead, d_k)
            return x.view(B, -1, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, L, d_k)

        Q = transform(query, self.q_linear)
        K = transform(key, self.k_linear)
        V = transform(value, self.v_linear)

        Q, K = self.rotary_pe(Q, K)

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
        # self.rotary_pe = RotaryPositionalEmbedding(dim=self.d_k, max_len=1024)

    def forward(self, query, key, value, mask=None):
        B, L, D = query.size()

        def transform(x, linear):
            x = linear(x)
            # reshape for multihead: (B, L, nhead, d_k)
            return x.view(B, -1, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, L, d_k)

        Q = transform(query, self.q_linear)
        K = transform(key, self.k_linear)
        V = transform(value, self.v_linear)

        # Q, K = self.rotary_pe(Q, K)

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
        self.self_attn = MultiHeadAttention_RoPE(d_model, nhead)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, mask):
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention_RoPE(d_model, nhead)
        self.cross_attn = MultiHeadAttention(d_model, nhead)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)

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