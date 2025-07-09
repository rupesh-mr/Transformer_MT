import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="./sentencepiece/en-indic-spm/model.SRC")

print("Pad ID:", sp.pad_id())
print("Pad Token:", sp.id_to_piece(sp.pad_id()))