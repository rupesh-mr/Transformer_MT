import torch
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import evaluate
from model_components import generate_padding_mask, generate_subsequent_mask
import pandas as pd
import os
import sentencepiece as spm
import sacrebleu

def log_metrics_to_file(epoch, tokenized_pred, tokenized_ref, preds, ref_texts, src_texts, log_file="results.txt"):
    with open(log_file, "a") as f:
        f.write(f"\n=== Epoch {epoch} ===\n")

        # BLEU
        bleu = sacrebleu.corpus_bleu(tokenized_pred, tokenized_ref, tokenize='none')
        f.write(f"BLEU Score: {bleu.score:.2f}\n")

        # CHRF++
        chrf_calc = sacrebleu.CHRF(word_order=2)
        chrf_score = chrf_calc.corpus_score(tokenized_pred, tokenized_ref)
        f.write(f"CHRF++: {chrf_score}\n")

        # chrF
        chrf = sacrebleu.corpus_chrf(tokenized_pred, tokenized_ref)
        f.write(f"chrF Score: {chrf.score:.2f}\n")

        # METEOR
        meteor = evaluate.load("meteor")
        meteor_result = meteor.compute(predictions=preds, references=ref_texts)
        f.write(f"METEOR Score: {meteor_result['meteor']:.2f}\n")

        f.write("--- Advanced Evaluation ---\n")

        # COMET
        try:
            comet = evaluate.load("comet")
            comet_result = comet.compute(
                predictions=preds,
                references=ref_texts,
                sources=src_texts
            )
            f.write(f"COMET Score: {comet_result['mean_score']:.4f}\n")
        except Exception as e:
            f.write(f"COMET error: {e}\n")

        # BERTScore
        try:
            bertscore = evaluate.load("bertscore")
            bertscore_result = bertscore.compute(predictions=preds, references=ref_texts, lang="en")
            f1 = sum(bertscore_result['f1']) / len(bertscore_result['f1'])
            f.write(f"BERTScore (F1): {f1:.4f}\n")
        except Exception as e:
            f.write(f"BERTScore error: {e}\n")

        f.write("=" * 40 + "\n")


def greedy_decode(model, src_tokens, max_len=200, bos_id=2, eos_id=3, repetition_penalty=2):
    model.eval()
    device = next(model.parameters()).device

    src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)
    src_mask = generate_padding_mask(src_tensor).to(device)

    with torch.no_grad():
        src_emb = model.shared_embed(src_tensor)
        for layer in model.encoder_layers:
            src_emb = layer(src_emb, src_mask)

    tokens = [bos_id]

    for _ in range(max_len):
        tgt_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
        tgt_mask = generate_padding_mask(tgt_tensor).to(device).expand(-1, -1, tgt_tensor.size(1), -1)
        causal = generate_subsequent_mask(tgt_tensor.size(1)).to(device).unsqueeze(0).unsqueeze(0)
        full_mask = tgt_mask & (~causal)

        with torch.no_grad():
            tgt_emb = model.shared_embed(tgt_tensor)
            tgt_out = tgt_emb
            for layer in model.decoder_layers:
                tgt_out = layer(tgt_out, src_emb, full_mask, src_mask)

            logits = model.generator(tgt_out)[0, -1]  # [V]

            # Apply repetition penalty
            for tok in set(tokens):
                if logits[tok] < 0:
                    logits[tok] *= repetition_penalty
                else:
                    logits[tok] /= repetition_penalty

            next_token = torch.argmax(torch.log_softmax(logits, dim=-1)).item()

        tokens.append(next_token)

        if next_token == eos_id:
            break

    return tokens

def compute_rdrop_loss(model, src, tgt_input, tgt_output, src_mask, tgt_mask, alpha=5.0, ignore_index=0):
    # Forward pass twice with different dropout masks
    logits_1,lb_loss1 = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)  # (B, T, V)
    logits_2,lb_loss2 = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)

    # lambda for load balance loss
    lambda_lb= 0.1
    # Average the load balance losses
    load_balance_loss = (lb_loss1 + lb_loss2) / 2
    
    B, T, V = logits_1.size()       
    logits_1_flat = logits_1.view(B * T, V)
    logits_2_flat = logits_2.view(B * T, V)
    tgt_output_flat = tgt_output.reshape(-1)

    # Cross-entropy loss for both outputs
    ce_loss_1 = F.cross_entropy(logits_1_flat, tgt_output_flat, ignore_index=ignore_index,label_smoothing=0.1)
    ce_loss_2 = F.cross_entropy(logits_2_flat, tgt_output_flat, ignore_index=ignore_index,label_smoothing=0.1)
    ce_loss = (ce_loss_1 + ce_loss_2) / 2

    # Compute KL divergence only on non-pad positions
    log_probs_1 = F.log_softmax(logits_1, dim=-1)  # (B, T, V)
    log_probs_2 = F.log_softmax(logits_2, dim=-1)    

    # Per-token KL divergence (B, T)
    
    kl_1 = F.kl_div(log_probs_1, log_probs_2, reduction='none', log_target=True).sum(-1)
    kl_2 = F.kl_div(log_probs_2, log_probs_1, reduction='none', log_target=True).sum(-1)

    # Mask out padding
    non_pad_mask = (tgt_output != ignore_index).float()  # (B, T)
    kl = ((kl_1 + kl_2) / 2) * non_pad_mask  # (B, T)

    kl_div = kl.sum() / non_pad_mask.sum()  # mean over non-pad tokens

    total_loss = ce_loss + alpha * kl_div + lambda_lb * load_balance_loss
    return total_loss

def train_model(model, dataloader,valid_dataloader, optimizer,scheduler, criterion, device, epochs=5,start_epoch=10, checkpoint_path=None):
    total_batches = len(dataloader)
    for epoch in range(1+start_epoch, start_epoch+epochs + 1):
        model.train()
        print(f"Epoch {epoch}/{start_epoch+epochs}")
        with open('training_log.txt', 'a') as log:
                log.write(f"Epoch {epoch}/{epochs}\n")
        epoch_start_time = time.time()
        batch_times = []
        total_loss=0
        for i, batch in enumerate(dataloader, 1):
            batch_start = time.time()

            src, tgt_input, tgt_output, src_mask, tgt_mask = batch

            src        = src.to(device)
            tgt_input  = tgt_input.to(device)
            tgt_output = tgt_output.to(device)
            src_mask   = src_mask.to(device)
            tgt_mask   = tgt_mask.to(device)

            # Compute loss with R-Drop
            loss = compute_rdrop_loss(model, src, tgt_input,tgt_output, src_mask, tgt_mask,alpha=5.0)
            # Backprop
            optimizer.zero_grad()
            # torch.autograd.set_detect_anomaly(True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}") 
            
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)

            avg_batch_time = sum(batch_times) / len(batch_times)
            batches_left = total_batches - i
            est_time_left = batches_left * avg_batch_time
            total_loss += loss.item()
            if i%10==0:
              with open('training_log.txt', 'a') as log:
                log.write(f"Batch {i}/{total_batches} - Loss: {loss.item():.4f}\n")
              print(f"Batch {i}/{total_batches} - Loss: {loss.item():.4f} - "
                    f"Batch time: {batch_time:.2f}s - Est. time left: {est_time_left:.2f}s")

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"\nEpoch {epoch+start_epoch} completed in {epoch_duration:.2f}s - Last Loss: {loss.item():.4f} - Average Loss: {total_loss / total_batches:.4f}")
        with open('training_log.txt', 'a') as log:
            log.write(f"Epoch {epoch} completed in {epoch_duration:.2f}s - Last Loss: {loss.item():.4f} - Average Loss: {total_loss / total_batches:.4f}\n\n")

        ckpt_file = checkpoint_path + '_' + str(epoch) + '.pth'
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item()
            }, ckpt_file)
            print(f"✅ Checkpoint saved to {ckpt_file}")
        except Exception as e:
            print(f"❌ torch.save failed: {e}")

        if not os.path.exists(ckpt_file):
            print(f"❌ File {ckpt_file} does not exist even after save!")
        

        # Validation
        model.eval()
        total_valid_loss = 0

        
        with torch.inference_mode():
            for i, batch in enumerate(valid_dataloader, 1):
                src, tgt_input, tgt_output, src_mask, tgt_mask = batch

                src        = src.to(device)
                tgt_input  = tgt_input.to(device)
                tgt_output = tgt_output.to(device)
                src_mask   = src_mask.to(device)
                tgt_mask   = tgt_mask.to(device)

                logits = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))   # logits.size(-1) as logits.size()
                total_valid_loss += loss.item()
        avg_valid_loss = total_valid_loss / len(valid_dataloader)
        print(f"Validation Loss after Epoch {epoch}: {avg_valid_loss:.4f}") #this uncomment
        with open('validation_log.txt', 'a') as log:
            log.write(f"Validation Loss after Epoch {epoch}: {avg_valid_loss:.4f}\n")
            
        print(f"Epoch {epoch}/{epochs}, Valid Loss: {avg_valid_loss:.4f}") # extra put    
        # Ensure source and target have the same number of lines
        df_dev=pd.read_csv("dev.csv")
        src_texts = df_dev['tgt'].tolist()
        ref_texts = df_dev['src'].tolist()
        #Wrap references for BLEU / chrf
        references = [[ref] for ref in ref_texts]    
        
        
        tokenizer= spm.SentencePieceProcessor(model_file="spm_joint.model")
        model.eval()
        with torch.inference_mode():    
            # Translate
            preds = []
            for src in src_texts:
                src_ids = tokenizer.encode(src,add_bos=True, add_eos=True)
                pred_ids = greedy_decode(model, src_ids, max_len=200, bos_id=2, eos_id=3)
                pred_text = tokenizer.decode(pred_ids)
                preds.append(pred_text)
                # Save predictions to a file
            output_file = "predictions_mni_eng.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                for line in preds:
                    f.write(line.strip() + "\n")
                    
            # ✅ Assert no mismatch
            assert len(preds) == len(ref_texts), "Mismatch between number of predictions and references!"    
            tokenized_pred = ["" for _ in range(len(preds))]
            tokenized_ref = [[""] for _ in range(len(ref_texts))]            
            for i in range(len(preds)):
                tokens_pred = tokenizer.encode(preds[i], out_type=str, add_bos=False, add_eos=False)
                tokens_ref = tokenizer.encode(ref_texts[i], out_type=str, add_bos=False, add_eos=False)

                tokenized_pred[i] = " ".join(tokens_pred)
                tokenized_ref[i] = [" ".join(tokens_ref)]
            # ====== Evaluation ======

            log_metrics_to_file(epoch,tokenized_pred, tokenized_ref, preds,references,src_texts,ref_texts,"log.txt")

# Function to load checkpoint
def load_checkpoint(model, optimizer,scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"Checkpoint loaded from {checkpoint_path}, last loss: {checkpoint['loss']}")

def get_inverse_sqrt_warmup_scheduler(optimizer, warmup_steps, peak_lr, initial_lr=1e-5):
    def lr_lambda(step):
        if step == 0:
            return initial_lr / peak_lr
        if step < warmup_steps:
            # Linear interpolate multiplier from initial_lr/peak_lr to 1
            return ((peak_lr - initial_lr) * step / warmup_steps + initial_lr) / peak_lr
        else:
            return (warmup_steps ** 0.5) * (step ** -0.5)

    for param_group in optimizer.param_groups:
        param_group['lr'] = peak_lr

    return LambdaLR(optimizer, lr_lambda)
