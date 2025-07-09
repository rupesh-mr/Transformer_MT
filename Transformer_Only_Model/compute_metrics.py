import torch
from model_components import generate_padding_mask, generate_subsequent_mask
import sacrebleu
import sentencepiece as spm
import pandas as pd 
from model_architecture import TransformerModel
from model_train import load_checkpoint, get_inverse_sqrt_warmup_scheduler




def beam_search_decode(model, src_tokens, beam_size=5, max_len=100, bos_id=2, eos_id=3, repetition_penalty=2):
    model.eval()
    device = next(model.parameters()).device

    src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)
    src_mask = generate_padding_mask(src_tensor).to(device)

    # Encode source once
    with torch.no_grad():
        src_emb = model.shared_embed(src_tensor)
        for layer in model.encoder_layers:
            src_emb = layer(src_emb, src_mask)

    beams = [([bos_id], 0.0)]  # (tokens, score)
    completed = []

    for _ in range(max_len):
        new_beams = []

        for tokens, score in beams:
            if tokens[-1] == eos_id:
                completed.append((tokens, score))
                continue

            tgt_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
            tgt_mask = generate_padding_mask(tgt_tensor).to(device).expand(-1, -1, tgt_tensor.size(1), -1)
            causal = generate_subsequent_mask(tgt_tensor.size(1)).to(device).unsqueeze(0).unsqueeze(0)
            full_mask = tgt_mask & (~causal)

            with torch.no_grad():
                tgt_emb = model.tgt_embed(tgt_tensor)
                tgt_out = tgt_emb
                for layer in model.decoder_layers:
                    tgt_out = layer(tgt_out, src_emb, full_mask, src_mask)

                logits = model.generator(tgt_out)  # [1, T, V]
                logits = logits[0, -1]  # [V]

                # Apply repetition penalty
                for tok in set(tokens):
                    if logits[tok] < 0:
                        logits[tok] *= repetition_penalty
                    else:
                        logits[tok] /= repetition_penalty

                log_probs = torch.log_softmax(logits, dim=-1)

            top_log_probs, top_tokens = torch.topk(log_probs, beam_size)

            for log_prob, tok in zip(top_log_probs.tolist(), top_tokens.tolist()):
                new_tokens = tokens + [tok]
                new_score = score + log_prob
                new_beams.append((new_tokens, new_score))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        if len(completed) >= beam_size:
            break

    completed.extend(beams)
    best_tokens, best_score = max(completed, key=lambda x: x[1])
    return best_tokens




bos_id = 2  
eos_id = 3  


import evaluate

if __name__ == "__main__":
   
    
    tgt_file = "../data/test.eng_Latn_conv.txt"
    src_file = "../data/test.mni_Mtei_conv.txt"
    
    with open(src_file, "r", encoding="utf-8") as f_src, open(tgt_file, "r", encoding="utf-8") as f_tgt:
        src_lines = f_src.readlines()
        tgt_lines = f_tgt.readlines()

    
    # Ensure source and target have the same number of lines
    assert len(src_lines) == len(tgt_lines), f"Line count mismatch: {len(src_lines)} vs {len(tgt_lines)}"

    # Strip newlines but preserve line order and count
    src_texts = [line.strip() for line in src_lines]
    ref_texts = [line.strip() for line in tgt_lines]
    references = [[ref] for ref in ref_texts]    
    
    
    tokenizer= spm.SentencePieceProcessor(model_file="spm_joint.model")
    model = TransformerModel(
        src_vocab_size=20000,  # Adjust based on your SentencePiece vocab size
        tgt_vocab_size=20000,  # Adjust based on your SentencePiece vocab size
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.2
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(model.parameters(),  betas=(0.9, 0.98),lr=1e-4,eps=1e-9)   #before lr=1e-5
    scheduler = get_inverse_sqrt_warmup_scheduler(optimizer, warmup_steps=4000, peak_lr=1e-4, initial_lr=1e-5)   #before peak_lr=1e-3
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1) 

    load_checkpoint(model, optimizer,scheduler, 'transformer_checkpoint_5.pth')
    model.eval()
    with torch.inference_mode():    
    # Translate
        preds = []
        for src in src_texts:
            src_ids = tokenizer.encode(src,add_bos=True, add_eos=True)
            pred_ids = beam_search_decode(model, src_ids, beam_size=5, max_len=200, bos_id=bos_id, eos_id=eos_id)
            pred_text = tokenizer.decode(pred_ids)
            preds.append(pred_text)
            print(f"Count: {len(preds)}")
            # Save predictions to a file
        output_file = "predictions_engTo_snd.txt"
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

             joined_pred = " ".join(tokens_pred)
             joined_ref = " ".join(tokens_ref)
             print(f"Pred: {tokenized_pred[i]}")
             print(f"Ref: {tokenized_ref[i][0]}")

        # ====== Evaluation ======

        # BLEU
        bleu = sacrebleu.corpus_bleu(tokenized_pred, tokenized_ref, tokenize='none')
        print(f" {bleu.score:.2f}")

        # CHRF++
        chrf_calc = sacrebleu.CHRF(word_order=2)
        chrf_score = chrf_calc.corpus_score(tokenized_pred, tokenized_ref)
        print(f"CHRF++: {chrf_score}")

        # chrF
        chrf = sacrebleu.corpus_chrf(tokenized_pred, tokenized_ref)
        print(f"chrF Score: {chrf.score:.2f}")

        # METEOR
        meteor = evaluate.load("meteor")
        meteor_result = meteor.compute(predictions=preds, references=ref_texts)
        print(f"METEOR Score: {meteor_result['meteor']:.2f}")
        
        print("=== Advanced Evaluation ===")

        # ==============================
        # COMET
        # ==============================
        print("\n--- COMET Score ---")
        try:
            comet = evaluate.load("comet")
            comet_result = comet.compute(
                predictions=preds, 
                references=ref_texts, 
                sources=src_texts
            )
            print(f"COMET Score: {comet_result['mean_score']:.4f}")
        except Exception as e:
            print(f"COMET error: {e}")
        
            
        # ==============================
        # BERTScore
        # ==============================
        print("\n--- BERTScore ---")
        try:
            bertscore = evaluate.load("bertscore")
            bertscore_result = bertscore.compute(predictions=preds, references=ref_texts, lang="en")
            print(f"(F1): {sum(bertscore_result['f1'])/len(bertscore_result['f1']):.4f}")
        except Exception as e:
            print(f" BERTScore error: {e}")


