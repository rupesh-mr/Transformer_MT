import pandas as pd
import sentencepiece as spm
import re


def clean_preprocessed(sent):
    # Remove tags like `mni_Mtei`, `eng_Latn`, `None` etc.
    return re.sub(r'\b[a-z]{3}_[A-Za-z]+\b|\bNone\b', '', sent).strip()


def create_bidirectional_csv(original_csv, output_csv, src_col, tgt_col, src_lang_token="<2mni>", tgt_lang_token="<2en>"):
    df = pd.read_csv(original_csv)

    # Add direction tokens to the source text
    df_forward = df.copy()
    df_backward = df.copy()
    df_backward[[src_col, tgt_col]] = df_backward[[tgt_col, src_col]]
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
        user_defined_symbols=["mni_Mtei", "eng_Latn"],
        normalization_rule_name='identity'
    )
