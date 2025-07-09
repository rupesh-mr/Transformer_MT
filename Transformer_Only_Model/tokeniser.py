import pandas as pd
import sentencepiece as spm


def create_bidirectional_csv(original_csv, output_csv, src_col, tgt_col, src_lang_token="<2mni>", tgt_lang_token="<2en>"):
    df = pd.read_csv(original_csv)

    # Add direction tokens to the source text
    df_forward = df.copy()
    df_forward[src_col] = src_lang_token + " " + df_forward[src_col]

    df_backward = df.copy()
    df_backward[[src_col, tgt_col]] = df_backward[[tgt_col, src_col]]
    df_backward[src_col] = tgt_lang_token + " " + df_backward[src_col]

    combined_df = pd.concat([df_forward, df_backward], ignore_index=True)
    combined_df.to_csv(output_csv, index=False)
    print(f"Bidirectional dataset saved to {output_csv}, total {len(combined_df)} samples.")

def prepare_joint_spm_training_text(csv_file, src_col, tgt_col, output_file="joint_spm_corpus.txt"):
    
    df = pd.read_csv(csv_file)
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for _, row in df.iterrows():
            src_text = str(row[src_col]).strip()
            tgt_text = str(row[tgt_col]).strip()

            if pd.notna(src_text):
                f_out.write(src_text + '\n')
            if pd.notna(tgt_text):
                f_out.write(tgt_text + '\n')

    print(f"Joint corpus saved to: {output_file}")
    
def train_sentencepiece(input_file, model_prefix, vocab_size=8000):
    spm.SentencePieceTrainer.Train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=["<2mni>", "<2en>"],
    )
# if __name__ == "__main__":
    # from pathlib import Path
    # import random
    # from datasets import load_dataset

    # Load the full IndicCorp v2 config
    # ds = load_dataset("ai4bharat/IndicCorpV2", split="mni_Mtei", name="indiccorp_v2")

    # Local download path for all .txt files
    # data_dir = Path(ds.cache_files[0]['filename']).parent / "data"

    # Confirm available language files
    # print("Languages present:", sorted([f.name for f in data_dir.glob("*.txt")]))

    # Read sentences
    # en_lines = data_dir / "en.txt"
    # mni_lines = data_dir / "mni.txt"

    # def load_and_sample(file_path, n=3_000_000, seed=42):
    #    lines = file_path.read_text(encoding="utf-8").splitlines()
     #   random.seed(seed)
      #  return random.sample(lines, min(n, len(lines)))

    #en_sents = load_and_sample(en_lines)
    #mni_sents = load_and_sample(mni_lines)

    # Save to training files
    #(Path.cwd() / "train.en.txt").write_text("\n".join(en_sents), encoding="utf-8")
    #(Path.cwd() / "train.mni.txt").write_text("\n".join(mni_sents), encoding="utf-8")

    #print(f"Saved {len(en_sents)} English and {len(mni_sents)} Meitei sentences.")
