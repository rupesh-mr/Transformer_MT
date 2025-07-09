import pandas as pd
import sentencepiece as spm
from IndicTransToolkit.processor import IndicProcessor
import re

def clean_preprocessed(sent):
    # Remove tags like `mni_Mtei`, `eng_Latn`, `None` etc.
    return re.sub(r'\b[a-z]{3}_[A-Za-z]+\b|\bNone\b', '', sent).strip()

def build_spm_model_from_csv(csv_path, col,indic_processor, output_prefix, vocab_size=8000):
    """
    Preprocess the 'tgt' column from a CSV using IndicProcessor and train a SentencePiece model.

    Args:
      csv_path (str): Path to the CSV file with 'tgt' column.
      indic_processor: IndicTransToolkit IndicProcessor instance.
      output_prefix (str): Prefix for output SPM files.
      vocab_size (int): Vocabulary size to train.
    """
    temp_file = f"preprocessed_{col}.txt"
    df = pd.read_csv(csv_path)

    if col not in df.columns:
        raise ValueError(f"CSV must contain a '{col}' column.")
    src="eng_Latn" if col == "src" else "mni_Mtei"
    with open(temp_file, 'w', encoding='utf-8') as fout:
        
        batch = []
        for sent in df[col].dropna():
            sent = str(sent).strip()
            if sent:
                batch.append(sent)
            
            if len(batch) == 4:
                processed = indic_processor.preprocess_batch(batch, src_lang=src)
                processed = [clean_preprocessed(line) for line in processed if line.strip()]
                for line in processed:
                    fout.write(line + "\n")
                batch = []

        if batch:
            processed = indic_processor.preprocess_batch(batch, src_lang="mni_Mtei")
            processed = [clean_preprocessed(line) for line in processed if line.strip()]
            for line in processed:
                fout.write(line + "\n")

    spm.SentencePieceTrainer.Train(
        input=temp_file,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=["<dnt>","</dnt>"]
    )

    print(f"Trained SPM model saved to {output_prefix}.model and .vocab")

build_spm_model_from_csv(
    "eng_mtei.csv",
    col="tgt",
    indic_processor=IndicProcessor(inference=True),
    output_prefix="sentencepiece/mtei_spm",
    vocab_size=8000
)

build_spm_model_from_csv(
    "eng_mtei.csv",
    col="src",
    indic_processor=IndicProcessor(inference=True),
    output_prefix="sentencepiece/eng_spm",
    vocab_size=8000
)