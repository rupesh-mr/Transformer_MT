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
        x = x + self.pe[:x.size(0), :]
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
            if mask.dim() == 4 and mask.size(1) == 1:
                mask = mask.expand(-1, self.nhead, -1, -1)
            scores = scores.masked_fill(~mask, float('-inf'))  
        attn = self.dropout(torch.softmax(scores, dim=-1))
        context = attn @ V  # (B, nhead, L_query, d_k)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(context)


class MoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Gating network: maps each token to expert scores
        self.gate = nn.Linear(d_model, num_experts)

        # Expert networks: each is a FeedForward block
        self.experts = nn.ModuleList([
            FeedForward(d_model, d_ff) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.size()
        gate_scores = self.gate(x)  # (B, L, E)
        # Get top-k expert indices and their scores
        topk_scores, topk_ids = torch.topk(gate_scores, self.top_k, dim=-1)  # both (B, L, k)
        topk_weights = torch.softmax(topk_scores, dim=-1)  # (B, L, k)
        #Load Balancing
        gate_probs = torch.softmax(gate_scores, dim=-1)

        expert_mask = torch.nn.functional.one_hot(topk_ids, num_classes=self.num_experts).float()  # (B, L, k, E)
        expert_mask = expert_mask.sum(dim=2)  # (B, L, E) -> count if expert selected

        # Fraction of tokens assigned to each expert
        expert_selection_freq = expert_mask.mean(dim=(0, 1))  # (E,)

        # Mean probability given by gate (before top-k filtering)
        expert_prob_mean = gate_probs.mean(dim=(0, 1))  # (E,)

        # Load balancing loss: encourage equal routing + equal gate probs
        load_balance_loss = self.num_experts * torch.sum(expert_selection_freq * expert_prob_mean)

        output = torch.zeros_like(x)

        for i in range(self.top_k):
            expert_ids = topk_ids[:, :, i]  # (B, L)
            weights= topk_weights[:, :, i]  # (B, L)

            for expert_id in range(self.num_experts):
                selected = (expert_ids == expert_id)  # (B, L)
                if selected.any():
                    # Gather tokens for this expert
                    expert_input = x[selected]  # (N_selected, D)
                    expert_output = self.experts[expert_id](expert_input)  # (N_selected, D)

                    expert_weight = weights[selected].unsqueeze(-1)  # (N_selected, 1)
                    expert_output*= expert_weight  # Scale expert output by its weight
                    output[selected] += expert_output
                    

        return output,load_balance_loss
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.ffn = MoE(d_model, d_ff//4, num_experts=4, top_k=2)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, mask):
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        moe_output, lb_loss = self.ffn(self.norm2(x))
        x = x + self.dropout(moe_output)  
        return x,lb_loss


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
