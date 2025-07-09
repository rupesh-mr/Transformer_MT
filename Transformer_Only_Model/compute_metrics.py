import torch
from model_components import generate_padding_mask, generate_subsequent_mask
import sacrebleu
# from indicnlp import common
# from indicnlp.tokenize import indic_tokenize 
import sentencepiece as spm
import pandas as pd 
from model_architecture import TransformerModel
from model_train import load_checkpoint, get_inverse_sqrt_warmup_scheduler
from IndicTransToolkit.processor import IndicProcessor

def apply_repetition_penalty(logits, prev_tokens, penalty):
    """
    Modifies logits in-place based on repetition penalty.
    """
    for token_id in set(prev_tokens):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits


import torch

def greedy_decode(model, src_tokens, max_len=200, bos_id=2, eos_id=3):
    model.eval()
    device = next(model.parameters()).device

    # Prepare source
    src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)  # (1, S)
    src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2)  # (1, 1, 1, S)

    with torch.no_grad():
        src_emb = model.src_embed(src_tensor)  # (1, S, D)
        for layer in model.encoder_layers:
            src_emb = layer(src_emb, src_mask)

    tokens = [bos_id]

    for _ in range(max_len):
        tgt_tensor = torch.tensor([tokens], dtype=torch.long).to(device)  # (1, T)
        tgt_mask = generate_subsequent_mask(len(tokens)).to(device)  # (T, T)

        with torch.no_grad():
            tgt_emb = model.tgt_embed(tgt_tensor)  # (1, T, D)
            tgt_out = tgt_emb
            for layer in model.decoder_layers:
                tgt_out = layer(tgt_out, src_emb, tgt_mask, src_mask)

            logits = model.generator(tgt_out)  # (1, T, V)

            top_log_probs, top_tokens = torch.topk(torch.log_softmax(logits, dim=-1), 10)
        
            print("Top token IDs:", top_tokens.tolist())
            print("Top token pieces:", [sp_tgt.id_to_piece(int(tok)) for tok in top_tokens.view(-1).tolist()])


            print("Top log probs:", top_log_probs.tolist())
            next_token = torch.argmax(logits[0, -1], dim=-1).item()  # Greedy: take argmax

        tokens.append(next_token)

        if next_token == eos_id:
            break

    return tokens




def beam_search_decode(model, src_tokens, beam_size=5, max_len=100, bos_id=2, eos_id=3, repetition_penalty=1.2):
    model.eval()
    device = next(model.parameters()).device

    src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)
    src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2)  # (1, 1, 1, S)

    with torch.no_grad():
        src_emb = model.src_embed(src_tensor)
        for layer in model.encoder_layers:
            src_emb = layer(src_emb, src_mask)

    beams = [([bos_id], 0.0)]
    completed = []

    for _ in range(max_len):
        new_beams = []

        for tokens, score in beams:
            if tokens[-1] == eos_id:
                completed.append((tokens, score))
                continue

            tgt_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
            tgt_mask = generate_subsequent_mask(tgt_tensor.size(1)).to(device)  # (T, T)

            with torch.no_grad():
                tgt_emb = model.tgt_embed(tgt_tensor)
                tgt_out = tgt_emb
                for layer in model.decoder_layers:
                    tgt_out = layer(tgt_out, src_emb, tgt_mask, src_mask)

                logits = model.generator(tgt_out)  # [1, T, V]
                logits = logits[0, -1]  # (V,)
                # logits = apply_repetition_penalty(logits, tokens, repetition_penalty)
                log_probs = torch.log_softmax(logits, dim=-1)

            top_log_probs, top_tokens = torch.topk(log_probs, beam_size)

            for log_prob, tok in zip(top_log_probs.tolist(), top_tokens.tolist()):
                new_tokens = tokens + [tok]
                new_score = score + log_prob
                new_beams.append((new_tokens, new_score))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        if all(t[-1] == eos_id for t, _ in beams):
            break

    completed.extend(beams)
    best_tokens, best_score = max(completed, key=lambda x: x[1])
    return best_tokens


# def beam_search_decode(model, src_tokens, beam_size=5, max_len=100, bos_id=2, eos_id=3, repetition_penalty=1.2):
#     model.eval()
#     device = next(model.parameters()).device

#     src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)
#     src_mask = generate_padding_mask(src_tensor).to(device)

#     with torch.no_grad():
#         src_emb = model.src_embed(src_tensor)
#         for layer in model.encoder_layers:
#             src_emb = layer(src_emb, src_mask)

#     beams = [([bos_id], 0.0)]
#     completed = []

#     for _ in range(max_len):
#         new_beams = []

#         for tokens, score in beams:
#             if tokens[-1] == eos_id:
#                 completed.append((tokens, score))
#                 continue

#             tgt_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
#             tgt_mask = generate_padding_mask(tgt_tensor).to(device).expand(-1, -1, tgt_tensor.size(1), -1)
#             causal = generate_subsequent_mask(tgt_tensor.size(1)).to(device).unsqueeze(0).unsqueeze(0)
#             full_mask = tgt_mask & (~causal)

#             with torch.no_grad():
#                 tgt_emb = model.tgt_embed(tgt_tensor)
#                 tgt_out = tgt_emb
#                 for layer in model.decoder_layers:
#                     tgt_out = layer(tgt_out, src_emb, full_mask, src_mask)

#                 logits = model.generator(tgt_out)  # [1, T, V]
#                 logits = logits[0, -1]  # (V,)

#                 # === Apply Repetition Penalty ===
#                 logits = apply_repetition_penalty(logits, tokens, repetition_penalty)

#                 log_probs = torch.log_softmax(logits, dim=-1)  # (V,)

#             top_log_probs, top_tokens = torch.topk(log_probs, beam_size)

#             for log_prob, tok in zip(top_log_probs.tolist(), top_tokens.tolist()):
#                 new_tokens = tokens + [tok]
#                 new_score = score + log_prob
#                 new_beams.append((new_tokens, new_score))

#         beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

#         if len(completed) >= beam_size:
#             break

#     completed.extend(beams)
#     best_tokens, best_score = max(completed, key=lambda x: x[1])
#     return best_tokens

bos_id = 2
eos_id = 3
# def tokenize_mni(text: str) -> str:
#     return " ".join(indic_tokenize.trivial_tokenize(text, "mni"))
import sacrebleu
# import evaluate

def evaluate_generation(predictions, references, sources=None, lang="mni_Mtei"):
    """
    Compute BLEU, chrF++, METEOR, COMET, and BERTScore exactly as in IndicTrans2.
    Args:
      predictions: List[str] of model outputs (post-processed)
      references: List[str] of reference texts
      sources:   List[str] of source texts (for COMET)
      lang:      Target language tag (unused here)
    """
    # Prepare references for sacrebleu
    refs = [[r] for r in references]
    refs_transposed = list(zip(*refs))

    # BLEU
    bleu = sacrebleu.corpus_bleu(predictions, refs_transposed, tokenize='none')
    print(f"BLEU = {bleu.score:.2f}")

    # chrF++
    chrf = sacrebleu.CHRF(word_order=2)
    chrf_score = chrf.corpus_score(predictions, refs_transposed)
    print(f"chrF++ = {chrf_score.score:.2f}")

    # # METEOR
    # try:
    #     meteor = evaluate.load('meteor')
    #     meteor_res = meteor.compute(predictions=predictions, references=references)
    #     print(f"METEOR = {meteor_res['meteor'] * 100:.2f}")
    # except Exception:
    #     print("METEOR: error computing")

    # # COMET
    # if sources is not None:
    #     try:
    #         comet = evaluate.load('comet')
    #         comet_res = comet.compute(predictions=predictions, references=references, sources=sources)
    #         print(f"COMET = {comet_res['mean_score'] * 100:.2f}")
    #     except Exception:
    #         print("COMET: error computing")

    # # BERTScore (fallback to lang='en')
    # try:
    #     bert = evaluate.load('bertscore')
    #     bs = bert.compute(predictions=predictions, references=references, lang='en')
    #     f1_avg = sum(bs['f1'])/len(bs['f1'])
    #     print(f"BERTScore-F1 = {f1_avg * 100:.2f}")
    # except Exception:
    #     print("BERTScore: error computing")

# import evaluate

if __name__ == "__main__":

    
    model_checkpoint = "transformer_checkpoint_1.pth"
    src_spm = "sentencepiece/eng_spm.model" 
    tgt_spm = "sentencepiece/mtei_spm.model"
    test_csv = "test.csv"  # with columns 'src' and 'tgt'

    # Load SentencePiece tokenizers
    sp_src = spm.SentencePieceProcessor(model_file=src_spm)
    sp_tgt = spm.SentencePieceProcessor(model_file=tgt_spm)


    print(sp_src.get_piece_size())
    print(sp_tgt.get_piece_size())
    # Load model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TransformerModel(
        src_vocab_size=sp_src.get_piece_size(),
        tgt_vocab_size=sp_tgt.get_piece_size(),
        d_model=512, nhead=8,
        num_encoder_layers=4, num_decoder_layers=4,
        dim_feedforward=1024, dropout=0.2
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), lr=1e-4, eps=1e-9)
    scheduler = get_inverse_sqrt_warmup_scheduler(optimizer, warmup_steps=4000, peak_lr=1e-4, initial_lr=1e-5)
    load_checkpoint(model, optimizer, scheduler, model_checkpoint)
    model.eval()

    # Read test data
    df_test = pd.read_csv(test_csv)
    df_test= df_test.iloc[-100:]  # Limit to last 100 rows for testing
    src_texts = df_test['src'].astype(str).tolist()
    ref_texts = df_test['tgt'].astype(str).tolist()

    # Postprocessor
    processor = IndicProcessor(inference=True)
    # Generate predictions
    preds = []
    srcs_for_comet = []
    # for src in src_texts[:5]:
    #     print("Input:", src)

    #     src_pp = processor.preprocess_batch([src], src_lang="eng_Latn", tgt_lang="mni_Mtei")[0]
    #     print("Preprocessed:", src_pp)

    #     src_ids = sp_src.encode(src_pp, out_type=int, add_bos=True, add_eos=True)
    #     print("Source IDs:", src_ids)
        
    #     print("Encoded src_ids:", src_ids)
    #     print("Decoded back:", sp_src.decode(src_ids))
       

    #     # pred_ids = beam_search_decode(model, src_ids, beam_size=5, max_len=200, bos_id=sp_tgt.bos_id(), eos_id=sp_tgt.eos_id())
    #     pred_ids = greedy_decode(
    #                             model,
    #                             src_tokens=src_ids,
    #                             max_len=200,
    #                             bos_id=sp_tgt.bos_id(),
    #                             eos_id=sp_tgt.eos_id()
    #                         )

        
    #     print("Predicted IDs:", pred_ids)
    #     print("Tokens:", [sp_tgt.id_to_piece(i) for i in pred_ids])
    #     decoded = sp_tgt.decode(pred_ids)
    #     print("Decoded:", decoded)

    #     post = processor.postprocess_batch([decoded], lang="mni_Mtei")[0]
    #     print("Postprocessed:", post)
    for src in src_texts:
        # Preprocess source
        src_pp = processor.preprocess_batch([src], src_lang="eng_Latn", tgt_lang="mni_Mtei")[0]
        # Tokenize
        src_ids = sp_src.encode(src_pp, out_type=int, add_bos=True, add_eos=True)
        print("Source IDs:", src_ids)
        # Beam search decode
        pred_ids = beam_search_decode(model, src_ids, beam_size=5, max_len=200, bos_id=2, eos_id=3)
        # Decode and postprocess
        pred_spm = sp_tgt.decode(pred_ids)
        print("Predicted SPM:", pred_spm)
        pred_text = processor.postprocess_batch([pred_spm], lang="mni_Mtei")[0]
        preds.append(pred_text)

        # Also store preprocessed source for COMET
        srcs_for_comet.append(src_pp)
    print(preds[:100])  # Print first 5 predictions for debugging
    

    evaluate_generation(preds, ref_texts, sources=srcs_for_comet)
    # # df = pd.read_csv("/export/home/vikas/MT_Work/data_file/test.csv")
    # # df = df.head(10)
    # # Replace with full paths if needed
    # # src_file = "/export/home/vikas/MT_Work/data_file/IN22_gen_test.eng_Latn.txt"
    # # tgt_file = "/export/home/vikas/MT_Work/data_file/IN22_gen_test.mni_Mtei.txt"
    
    # src_file = "/export/home/vikas/MT_Work/data_file/test.eng_Latn_conv.txt"
    # tgt_file = "/export/home/vikas/MT_Work/data_file/test.mni_Mtei_conv.txt"
    
    # # Load data from .txt files
    # with open(src_file, "r", encoding="utf-8") as f:
    #     src_texts = [
    #     f"<2mni> {line.strip()}"
    #     for line in f
    #     if line.strip()
    #     ]
    # with open(tgt_file, "r", encoding="utf-8") as f:
    #     ref_texts = [line.strip() for line in f if line.strip()]
    
    
    # # src_texts = df["src"].astype(str).tolist()
    # # ref_texts = df["tgt"].astype(str).tolist()
    # #references = [[ref] for ref in ref_texts]
    # #Wrap references for BLEU / chrF
    # references = [[ref] for ref in ref_texts]
    
    # tokenizer= spm.SentencePieceProcessor(model_file="spm_joint.model")
    # model = TransformerModel(
    #     src_vocab_size=20000,  # Adjust based on your SentencePiece vocab size
    #     tgt_vocab_size=20000,  # Adjust based on your SentencePiece vocab size
    #     d_model=1024,
    #     nhead=16,
    #     num_encoder_layers=18,
    #     num_decoder_layers=18,
    #     dim_feedforward=8192,
    #     dropout=0.2
    # ).to("cuda" if torch.cuda.is_available() else "cpu")

    # optimizer = torch.optim.Adam(model.parameters(),  betas=(0.9, 0.98),lr=1e-4,eps=1e-9)   #before lr=1e-5
    # scheduler = get_inverse_sqrt_warmup_scheduler(optimizer, warmup_steps=4000, peak_lr=1e-4, initial_lr=1e-5)   #before peak_lr=1e-3
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=0) 

    # load_checkpoint(model, optimizer,scheduler, 'transformer_checkpoint_14.pth')
    # model.eval()
    # with torch.inference_mode():    
    # # Translate
    #     preds = []
    #     for src in src_texts:
    #         src_ids = tokenizer.encode(src,add_bos=True, add_eos=True)
    #         pred_ids = beam_search_decode(model, src_ids, beam_size=5, max_len=200, bos_id=bos_id, eos_id=eos_id)
    #         pred_text = tokenizer.decode(pred_ids)
    #         preds.append(pred_text)
    #         # Save predictions to a file
    #     output_file = "predictions.txt"
    #     with open(output_file, "w", encoding="utf-8") as f:
    #         for line in preds:
    #             f.write(line.strip() + "\n")
    #     # ====== Evaluation ======

    #     # BLEU
    #     bleu = sacrebleu.corpus_bleu(preds, references, tokenize='none')
    #     print(f"BLEU Score: {bleu.score:.2f}")

    #     # CHRF++
    #     chrf_calc = sacrebleu.CHRF(word_order=2)
    #     chrf_score = chrf_calc.corpus_score(preds, references)
    #     print(f"CHRF++: {chrf_score}")

    #     # chrF
    #     chrf = sacrebleu.corpus_chrf(preds, references)
    #     print(f"chrF Score: {chrf.score:.2f}")

    #     # METEOR
    #     meteor = evaluate.load("meteor")
    #     meteor_result = meteor.compute(predictions=preds, references=ref_texts)
    #     print(f"METEOR Score: {meteor_result['meteor']:.2f}")
        
    #     print("=== Advanced Evaluation ===")

    #     # ==============================
    #     # COMET
    #     # ==============================
    #     print("\n--- COMET Score ---")
    #     try:
    #         comet = evaluate.load("comet")
    #         comet_result = comet.compute(
    #             predictions=preds, 
    #             references=ref_texts, 
    #             sources=src_texts
    #         )
    #         print(f"COMET Score: {comet_result['mean_score']:.4f}")
    #     except Exception as e:
    #         print(f"COMET error: {e}")
        
            
    #     # ==============================
    #     # BERTScore
    #     # ==============================
    #     print("\n--- BERTScore ---")
    #     try:
    #         bertscore = evaluate.load("bertscore")
    #         bertscore_result = bertscore.compute(predictions=preds, references=ref_texts, lang="en")
    #         print(f"BERTScore (F1): {sum(bertscore_result['f1'])/len(bertscore_result['f1']):.4f}")
    #     except Exception as e:
    #         print(f"BERTScore error: {e}")


