import torch
import torch.nn as nn
from model_components import EncoderLayer, DecoderLayer,generate_padding_mask, generate_subsequent_mask
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead,
                 num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        # self.pos_decoder = PositionalEncoding(d_model, dropout)
        
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_decoder_layers)])

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
        # src = self.pos_encoder(src)
        # tgt = self.pos_decoder(tgt)

        # Encode
        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        # Decode
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, tgt_mask, src_mask)

        return self.generator(tgt)  # [B, T, V]
