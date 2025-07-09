import torch
from model_components import generate_padding_mask, generate_subsequent_mask
import sacrebleu
# from indicnlp import common
# from indicnlp.tokenize import indic_tokenize 
import sentencepiece as spm
import pandas as pd 
from model_architecture import TransformerModel
from model_train import load_checkpoint, get_inverse_sqrt_warmup_scheduler


def beam_search_decode(model, src_tokens,beam_size=5, max_len=100, bos_id=2, eos_id=3):
        model.eval()
        device = next(model.parameters()).device

        src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)
        src_mask = generate_padding_mask(src_tensor).to(device)

        # Encode source once
        with torch.no_grad():
            src_emb = model.src_embed(src_tensor)
            for layer in model.encoder_layers:
                src_emb = layer(src_emb, src_mask)

        # Beam: list of tuples (tokens, score)
        beams = [( [bos_id], 0.0 )]  # start with BOS, score=0

        completed = []

        for _ in range(max_len):
            new_beams = []

            # For all candidates in the beam, expand by one token
            for tokens, score in beams:
                if tokens[-1] == eos_id:
                    # Already ended, add to completed
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
                    log_probs = torch.log_softmax(logits[0, -1], dim=-1)  # last token probs

                # Get top beam_size tokens and scores
                top_log_probs, top_tokens = torch.topk(log_probs, beam_size)

                for log_prob, tok in zip(top_log_probs.tolist(), top_tokens.tolist()):
                    new_tokens = tokens + [tok]
                    new_score = score + log_prob  # sum log probs for total score
                    new_beams.append((new_tokens, new_score))

            # Keep top beam_size beams sorted by score (descending)
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

            # Stop if all beams ended
            if len(completed) >= beam_size:
                break

        # Add unfinished beams to completed (if any)
        completed.extend(beams)
        # Choose the sequence with highest score
        best_tokens, best_score = max(completed, key=lambda x: x[1])

        return best_tokens

bos_id = 2  
eos_id = 3  
# def tokenize_mni(text: str) -> str:
#     return " ".join(indic_tokenize.trivial_tokenize(text, "mni"))

import evaluate

if __name__ == "__main__":
   
    # df = pd.read_csv("/export/home/vikas/MT_Work/data_file/test.csv")
    # df = df.head(10)
    # Replace with full paths if needed
    # src_file = "/export/home/vikas/MT_Work/data_file/IN22_gen_test.eng_Latn.txt"
    # tgt_file = "/export/home/vikas/MT_Work/data_file/IN22_gen_test.mni_Mtei.txt"
    
    src_file = "/export/home/vikas/MT_Work/data_file/test.eng_Latn_conv.txt"
    tgt_file = "/export/home/vikas/MT_Work/data_file/test.mni_Mtei_conv.txt"
    
    # Load data from .txt files
    with open(src_file, "r", encoding="utf-8") as f:
        src_texts = [
        f"<2mni> {line.strip()}"
        for line in f
        if line.strip()
        ]
    with open(tgt_file, "r", encoding="utf-8") as f:
        ref_texts = [line.strip() for line in f if line.strip()]
    
    
    # src_texts = df["src"].astype(str).tolist()
    # ref_texts = df["tgt"].astype(str).tolist()
    #references = [[ref] for ref in ref_texts]
    #Wrap references for BLEU / chrF
    references = [[ref] for ref in ref_texts]
    
    tokenizer= spm.SentencePieceProcessor(model_file="spm_joint.model")
    model = TransformerModel(
        src_vocab_size=20000,  # Adjust based on your SentencePiece vocab size
        tgt_vocab_size=20000,  # Adjust based on your SentencePiece vocab size
        d_model=1024,
        nhead=16,
        num_encoder_layers=18,
        num_decoder_layers=18,
        dim_feedforward=8192,
        dropout=0.2
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(model.parameters(),  betas=(0.9, 0.98),lr=1e-4,eps=1e-9)   #before lr=1e-5
    scheduler = get_inverse_sqrt_warmup_scheduler(optimizer, warmup_steps=4000, peak_lr=1e-4, initial_lr=1e-5)   #before peak_lr=1e-3
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0) 

    load_checkpoint(model, optimizer,scheduler, 'transformer_checkpoint_14.pth')
    model.eval()
    with torch.inference_mode():    
    # Translate
        preds = []
        for src in src_texts:
            src_ids = tokenizer.encode(src,add_bos=True, add_eos=True)
            pred_ids = beam_search_decode(model, src_ids, beam_size=5, max_len=200, bos_id=bos_id, eos_id=eos_id)
            pred_text = tokenizer.decode(pred_ids)
            preds.append(pred_text)
            # Save predictions to a file
        output_file = "predictions.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for line in preds:
                f.write(line.strip() + "\n")
        # ====== Evaluation ======

        # BLEU
        bleu = sacrebleu.corpus_bleu(preds, references, tokenize='none')
        print(f"BLEU Score: {bleu.score:.2f}")

        # CHRF++
        chrf_calc = sacrebleu.CHRF(word_order=2)
        chrf_score = chrf_calc.corpus_score(preds, references)
        print(f"CHRF++: {chrf_score}")

        # chrF
        chrf = sacrebleu.corpus_chrf(preds, references)
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
            print(f"BERTScore (F1): {sum(bertscore_result['f1'])/len(bertscore_result['f1']):.4f}")
        except Exception as e:
            print(f"BERTScore error: {e}")


