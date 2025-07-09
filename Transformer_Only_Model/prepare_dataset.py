import os
import torch
import pandas as pd
import sentencepiece as spm
from torch.utils.data import Dataset
from model_components import generate_subsequent_mask
from torch.nn.utils.rnn import pad_sequence

# Import IndicProcessor for proper preprocessing/postprocessing
def handle_import():
    try:
        from IndicTransToolkit.processor import IndicProcessor
        return IndicProcessor
    except ImportError:
        raise ImportError("Please install IndicTransToolkit: git clone https://github.com/VarunGumma/IndicTransToolkit.git && pip install -e IndicTransToolkit")

IndicProcessor = handle_import()

class TranslationDataset(Dataset):
    def __init__(self, csv_file, src_model_file, tgt_model_file, src_col, tgt_col,
                 src_lang="eng_Latn", tgt_lang="mni_Mtei",
                 isvalid=False, max_len=256, cache_file='tokenized_dataset.pt'):
        """
        Dataset applying IndicTransToolkit preprocessing and SentencePiece tokenization.
        Preprocesses everything in __init__ so __getitem__ is safe for multiprocessing.
        """
        self.cache_file = cache_file
        self.max_len = max_len

        if os.path.exists(cache_file):
            print(f"Loading tokenized dataset from {cache_file}...")
            self.encoded_data = torch.load(cache_file)
            print(f"Loaded {len(self.encoded_data)} samples.")
        else:
            print("Pre-tokenizing dataset with IndicTransToolkit processor...")
            self.data = pd.read_csv(csv_file)
            self.src_tokenizer = spm.SentencePieceProcessor(model_file=src_model_file)
            self.tgt_tokenizer = spm.SentencePieceProcessor(model_file=tgt_model_file)
            self.processor = IndicProcessor(inference=True)

            self.encoded_data = []
            for i in range(len(self.data)):
                src_text = self.data.iloc[i][src_col]
                tgt_text = self.data.iloc[i][tgt_col]

                src_pp = self.processor.preprocess_batch([src_text], src_lang=src_lang, tgt_lang=tgt_lang)[0]
                tgt_pp = self.processor.preprocess_batch([tgt_text], src_lang=tgt_lang, tgt_lang=src_lang)[0]

                src_ids = self.src_tokenizer.encode(src_pp, out_type=int, add_bos=True, add_eos=True)[:max_len]
                tgt_ids = self.tgt_tokenizer.encode(tgt_pp, out_type=int, add_bos=True, add_eos=True)[:max_len]

                self.encoded_data.append((torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)))

                if i % 1000 == 0:
                    print(f"Tokenized {i} samples...")

            torch.save(self.encoded_data, cache_file)
            print(f"Tokenized dataset saved to: {cache_file}")

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]

def collate_fn(batch, pad_id=0):
    """
    batch: list of (src_ids, tgt_ids) tensors
    pad_id: the ID used for padding
    Returns:
      src_batch:     [B, S] padded source sequences
      tgt_input:     [B, T-1] target inputs (<sos> … last-1)
      tgt_output:    [B, T-1] target outputs (1 … <eos>)
      src_mask:      [B, 1, 1, S] boolean mask for src (True=keep)
      tgt_mask:      [B, 1, T-1, T-1] combined padding+causal mask for tgt
    """
    src_seqs, tgt_seqs = zip(*batch)
    # pad to longest in batch
    src_batch = pad_sequence(src_seqs, batch_first=True, padding_value=pad_id)
    tgt_batch = pad_sequence(tgt_seqs, batch_first=True, padding_value=pad_id)

    # prepare tgt_input / tgt_output
    # assume tgt_batch = [B, T] with both <sos> & <eos> in it
    tgt_input  = tgt_batch[:, :-1]  # drop last token
    tgt_output = tgt_batch[:,  1:]  # drop first token

    # masks
    # True where not padding
    src_mask = (src_batch != pad_id).unsqueeze(1).unsqueeze(2)  # [B,1,1,S]
    tgt_pad_mask = (tgt_input != pad_id).unsqueeze(1).unsqueeze(2)  # [B,1,1,T-1]

    # causal mask (upper triangle) of size [T-1, T-1]
    Tm1 = tgt_input.size(1)
    causal = torch.triu(torch.ones(Tm1, Tm1, dtype=torch.bool), diagonal=1)  # True=mask
    causal = causal.unsqueeze(0).unsqueeze(0)  # [1,1,T-1,T-1]

    # combine: keep where pad_mask==True AND causal==False
    tgt_mask = tgt_pad_mask & (~causal)  # [B,1,T-1,T-1]

    return src_batch, tgt_input, tgt_output, src_mask, tgt_mask









# import os
# import torch
# import pandas as pd
# import sentencepiece as spm
# from torch.utils.data import Dataset
# from torch.nn.utils.rnn import pad_sequence


# class TranslationDataset(Dataset):
#     def __init__(self, csv_file, sp_model, src_col, tgt_col,
#                  isvalid=False, max_len=256, cache_file='tokenized_dataset.pt'):
        
#         self.cache_file = cache_file
#         self.max_len = max_len

#         if os.path.exists(cache_file):
#             print(f"Loading tokenized dataset from {cache_file}...")
#             self.encoded_data = torch.load(cache_file)
#             print(f"Loaded {len(self.encoded_data)} samples.")
#         else:
#             print("Pre-tokenizing dataset with IndicTrans preprocessor...")
#             self.data = pd.read_csv(csv_file)
#             self.tokenizer = spm.SentencePieceProcessor(model_file=sp_model)
#             self.processor = IndicProcessor(inference=not isvalid)

#             self.encoded_data = []

#             for i in range(len(self.data)):
#                 src_text = self.data.iloc[i][src_col]
#                 tgt_text = self.data.iloc[i][tgt_col]

#                 # Preprocess source and target using IndicProcessor
#                 src_pp = self.processor.preprocess_batch([src_text], src_lang="eng_Latn", tgt_lang="mni_Mtei")[0]
#                 tgt_pp = self.processor.preprocess_batch([tgt_text], src_lang="mni_Mtei", tgt_lang="eng_Latn")[0]

#                 # Apply SPM tokenization with <s> and </s>
#                 src_ids = self.tokenizer.encode(src_pp, out_type=int, add_bos=True, add_eos=True)[:self.max_len]
#                 tgt_ids = self.tokenizer.encode(tgt_pp, out_type=int, add_bos=True, add_eos=True)[:self.max_len]

#                 self.encoded_data.append((torch.tensor(src_ids), torch.tensor(tgt_ids)))

#                 if i % 1000 == 0:
#                     print(f"Tokenized {i} samples...")

#             torch.save(self.encoded_data, cache_file)
#             print(f"Tokenized dataset saved to: {cache_file}")

#     def __len__(self):
#         return len(self.encoded_data)

#     def __getitem__(self, idx):
#         return self.encoded_data[idx]

# # class TranslationDataset(Dataset):
# #     def __init__(self, csv_file, sp_model, src_col, tgt_col,isvalid=False,max_len=128, cache_file='tokenized_dataset.pt'):  #made the chnages here with max len 256 to 128 and put isvalid=False also
# #             self.cache_file = cache_file
# #             self.max_len = max_len    

# #             if os.path.exists(cache_file):
# #                 print(f"Loading tokenized dataset from {cache_file}...")
# #                 self.encoded_data = torch.load(cache_file)
# #                 print(f"Loaded {len(self.encoded_data)} samples.")
# #             else:
# #                 print("Pre-tokenizing dataset...")
# #                 self.data = pd.read_csv(csv_file)
# #                 # self.data = self.data.loc[:900]
# #                 self.tokenizer = spm.SentencePieceProcessor(model_file=sp_model)
# #                 self.encoded_data = []

# #                 for i in range(len(self.data)):
# #                     src_text = self.data.iloc[i][src_col]
# #                     tgt_text = self.data.iloc[i][tgt_col]

# #                     src_ids = self.tokenizer.encode(src_text, out_type=int,add_bos=True, add_eos=True)[:self.max_len]
# #                     tgt_ids = self.tokenizer.encode(tgt_text, out_type=int,add_bos=True, add_eos=True)[:self.max_len]
# #                     self.encoded_data.append((torch.tensor(src_ids), torch.tensor(tgt_ids)))

# #                     if i % 1000 == 0:
# #                         print(f"Tokenized {i} samples...")

# #                 torch.save(self.encoded_data, cache_file)
# #                 print(f"Tokenized dataset saved to: {cache_file}")

# #     def __len__(self):
# #         return len(self.encoded_data)

# #     def __getitem__(self, idx):
# #         return self.encoded_data[idx]



# def collate_fn(batch, pad_id=0):
#     """
#     batch: list of (src_ids, tgt_ids) tensors
#     pad_id: the ID used for padding
#     Returns:
#       src_batch:     [B, S] padded source sequences
#       tgt_input:     [B, T-1] target inputs (<sos> … last-1)
#       tgt_output:    [B, T-1] target outputs (1 … <eos>)
#       src_mask:      [B, 1, 1, S] boolean mask for src (True=keep)
#       tgt_mask:      [B, 1, T-1, T-1] combined padding+causal mask for tgt
#     """
#     src_seqs, tgt_seqs = zip(*batch)
#     # pad to longest in batch
#     src_batch = pad_sequence(src_seqs, batch_first=True, padding_value=pad_id)
#     tgt_batch = pad_sequence(tgt_seqs, batch_first=True, padding_value=pad_id)

#     # prepare tgt_input / tgt_output
#     # assume tgt_batch = [B, T] with both <sos> & <eos> in it
#     tgt_input  = tgt_batch[:, :-1]  # drop last token
#     tgt_output = tgt_batch[:,  1:]  # drop first token

#     # masks
#     # True where not padding
#     src_mask = (src_batch != pad_id).unsqueeze(1).unsqueeze(2)  # [B,1,1,S]
#     tgt_pad_mask = (tgt_input != pad_id).unsqueeze(1).unsqueeze(2)  # [B,1,1,T-1]

#     # causal mask (upper triangle) of size [T-1, T-1]
#     Tm1 = tgt_input.size(1)
#     causal = torch.triu(torch.ones(Tm1, Tm1, dtype=torch.bool), diagonal=1)  # True=mask
#     causal = causal.unsqueeze(0).unsqueeze(0)  # [1,1,T-1,T-1]

#     # combine: keep where pad_mask==True AND causal==False
#     tgt_mask = tgt_pad_mask & (~causal)  # [B,1,T-1,T-1]

#     return src_batch, tgt_input, tgt_output, src_mask, tgt_mask
