from model_architecture import TransformerModel
import torch
import pandas as pd
from model_train import load_checkpoint, get_inverse_sqrt_warmup_scheduler, train_model
import sentencepiece as spm
from model_components import generate_padding_mask, generate_subsequent_mask
model = TransformerModel(
        src_vocab_size=20000,  # Adjust based on your SentencePiece vocab size #chnages 32k to 8k
        tgt_vocab_size=20000,  # Adjust based on your SentencePiece vocab size
        d_model=1024,
        nhead=16,
        num_encoder_layers=18,
        num_decoder_layers=18,
        dim_feedforward=8192,
        dropout=0.2
    ).to("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = 'transformer_checkpoint_5.pth'  # Adjust the path as needed
optimizer = torch.optim.Adam(model.parameters(),  betas=(0.9, 0.98),lr=1e-4,eps=1e-9)  #lr=1e-5 this value was chnaged as per the previous lr=5e-4
scheduler = get_inverse_sqrt_warmup_scheduler(optimizer, warmup_steps=4000, peak_lr=1e-4, initial_lr=1e-7)  
criterion = torch.nn.CrossEntropyLoss(ignore_index=0) 
load_checkpoint(model, optimizer, scheduler, checkpoint_path)
df_dev=pd.read_csv("../data/dev.csv")
src_texts = df_dev['src'].tolist()
ref_texts = df_dev['tgt'].tolist()
tokenizer = spm.SentencePieceProcessor(model_file="spm_joint.model")
src = torch.tensor(tokenizer.encode(src_texts[0],add_bos=True, add_eos=True)).reshape(1, -1)
tgt=torch.tensor(tokenizer.encode(ref_texts[0],add_bos=True, add_eos=True)).reshape(1, -1)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
src = src.to(device)
tgt = tgt.to(device)

# Create masks
src_mask = generate_padding_mask(src).to(device)
pad_mask = generate_padding_mask(tgt).to(device).expand(-1, -1, tgt.size(1), -1)
causal_mask = generate_subsequent_mask(tgt.size(1)).to(device).unsqueeze(0).unsqueeze(0)
tgt_mask = pad_mask & (~causal_mask)

# Forward pass: Collect attention
encoder_attns = []
x = model.shared_embed(src)
x = model.pos_encoder(x)
for layer in model.encoder_layers:
    x, attn = layer(x, src_mask, return_attn=True)
    encoder_attns.append(attn[0])  # (nhead, S, S)

decoder_self_attns = []
decoder_cross_attns = []
y = model.shared_embed(tgt)
y = model.pos_decoder(y)
for layer in model.decoder_layers:
    y, self_attn, cross_attn = layer(y, x, tgt_mask, src_mask, return_attn=True)
    decoder_self_attns.append(self_attn[0])    # (nhead, T, T)
    decoder_cross_attns.append(cross_attn[0])  # (nhead, T, S)


import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_across_heads(attn, tokens_x, tokens_y, title):
    """
    attn: (nhead, T, S) attention weights
    tokens_x: key tokens (horizontal)
    tokens_y: query tokens (vertical)
    """
    nhead = attn.shape[0]
    fig, axes = plt.subplots(1, nhead, figsize=(4 * nhead, 4))
    for i in range(nhead):
        ax = axes[i] if nhead > 1 else axes
        sns.heatmap(attn[i].cpu().detach().numpy(), 
                    xticklabels=tokens_x, 
                    yticklabels=tokens_y, 
                    cmap="viridis", ax=ax)
        ax.set_title(f"{title} - Head {i}")
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

def plot_attention(attn,layer,head, tokens_x, tokens_y, title):
    """
    attn: (T, S) attention weights
    tokens_x: key tokens (horizontal)
    tokens_y: query tokens (vertical)
    """
    sns.heatmap(attn[layer-1][head-1].cpu().detach().numpy(), 
                xticklabels=tokens_x, 
                yticklabels=tokens_y, 
                cmap="viridis")
    plt.title(title)
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

tokens_src = [tokenizer.id_to_piece(i) for i in src[0].tolist()]
tokens_tgt = [tokenizer.id_to_piece(i) for i in tgt[0].tolist()]

plot_attention(decoder_cross_attns,1, 1,tokens_src, tokens_tgt, "Decoder Cross Attention (Layer 1, Head 1)")

