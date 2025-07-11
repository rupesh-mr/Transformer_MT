import torch
import torch.nn as nn
from model_components import EncoderLayer, DecoderLayer,PositionalEncoding, generate_padding_mask, generate_subsequent_mask,build_encoder_layers,build_decoder_layers

#moe_conditions
moe_odd = lambda i, n: i % 2 == 1
moe_even = lambda i, n: i % 2 == 0
moe_first_half = lambda i, n: i < n // 2
moe_second_half = lambda i, n: i >= n // 2

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead,
                 num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        
        self.encoder_layers = build_encoder_layers(num_encoder_layers, d_model, nhead, dim_feedforward, moe_even)  # or moe_odd etc.
        self.decoder_layers = build_decoder_layers(num_decoder_layers, d_model, nhead, dim_feedforward, moe_even)  # or moe_odd etc.

        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src:      [B, S]            token IDs
        tgt:      [B, T]            token IDs
        src_mask: [B,1,1,S] or None
        tgt_mask: [B,1,T,T] or None
        """
        device = src.device

        if src_mask is None:
            src_mask = generate_padding_mask(src).to(device)
        if tgt_mask is None:
            pad_mask = generate_padding_mask(tgt).to(device).expand(-1, -1, tgt.size(1), -1)
            causal = generate_subsequent_mask(tgt.size(1)).to(device).unsqueeze(0).unsqueeze(0)
            tgt_mask = pad_mask & (~causal)

        # Embed tokens
        src = self.src_embed(src)  # [B, S, D]
        tgt = self.tgt_embed(tgt)  # [B, T, D]
        src = self.pos_encoder(src)
        tgt = self.pos_decoder(tgt)

        total_lb_loss=0.0
        # Encode
        for layer in self.encoder_layers:
            src,lb_loss= layer(src, src_mask)
            total_lb_loss+=lb_loss

        # Decode
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, tgt_mask, src_mask)

        return self.generator(tgt),total_lb_loss  # [B, T, V]
