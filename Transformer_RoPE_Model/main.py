from torch.utils.data import DataLoader
import os
import torch
from prepare_dataset import TranslationDataset, collate_fn
from tokeniser import prepare_joint_spm_training_text, train_sentencepiece,create_bidirectional_csv
from model_architecture import TransformerModel
from model_train import train_model, load_checkpoint, get_inverse_sqrt_warmup_scheduler
import pandas as pd
import sentencepiece as spm



if __name__ == "__main__":
    if os.path.exists("spm_joint.model"):
        print("SentencePiece model already exist. Skipping training.")
    else:
        print("Training SentencePiece model...")
        prepare_joint_spm_training_text("eng_mtei.csv", "src", "tgt", "joint_spm_corpus.txt")
        train_sentencepiece("joint_spm_corpus.txt", "spm_joint", vocab_size=20000)
    
    csv_file = "eng_mtei.csv"

    bidiredctional_csv_file = "bidirectional_eng_mtei.csv"
    create_bidirectional_csv(csv_file, "bidirectional_eng_mtei.csv", "src", "tgt", src_lang_token="<2mni>", tgt_lang_token="<2en>") 
    dataset = TranslationDataset(bidiredctional_csv_file, "spm_joint.model","src", "tgt", cache_file='tokenized_dataset.pt')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # Load the validation dataset
    bidiredctional_validation_csv_file = "bidirectional_dev.csv"
    create_bidirectional_csv("dev.csv", bidiredctional_validation_csv_file, "src", "tgt", src_lang_token="<2mni>", tgt_lang_token="<2en>")
    valid_dataset = TranslationDataset(bidiredctional_validation_csv_file, "spm_joint.model", "src", "tgt", cache_file='tokenized_valid_dataset.pt')
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = TransformerModel(
        src_vocab_size=20000,  
        tgt_vocab_size=20000,  
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.2
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(),  betas=(0.9, 0.98),lr=1e-4,eps=1e-9)  
    scheduler = get_inverse_sqrt_warmup_scheduler(optimizer, warmup_steps=4000, peak_lr=1e-4, initial_lr=1e-7)  
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1) 
    #load_checkpoint(model, optimizer,scheduler, 'transformer_checkpoint_10.pth')
    train_model(
        model,
        dataloader,
        valid_dataloader,
        optimizer,
        scheduler,
        criterion,
        epochs=5,
        start_epoch=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_path='transformer_checkpoint'
    )
    
