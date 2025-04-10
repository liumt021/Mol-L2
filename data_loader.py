import torch
from torch.utils.data import Dataset
from data_pre import mask_text, mask_smiles_text
import pandas as pd


class Smi_CapAlignedDataset(Dataset):
    def __init__(self, tokenizer, filepath, max_length):
        df = pd.read_csv(filepath, sep="\t")
        self.texts_smi = df.iloc[:, 1].tolist()
        self.texts_cap = df.iloc[:, 2].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts_smi)

    def __getitem__(self, idx):
        smi_text = self.texts_smi[idx]
        cap_text = self.texts_cap[idx]

        smi_input, smi_target = self.mask_smiles_text(smi_text)
        cap_input, cap_target = self.mask_text(cap_text)

        smi_enc = self.tokenizer(smi_input, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        smi_tgt = self.tokenizer(smi_target, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        cap_enc = self.tokenizer(cap_input, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        cap_tgt = self.tokenizer(cap_target, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        return {
            "smi_input_ids": smi_enc["input_ids"].squeeze(),
            "smi_attention_mask": smi_enc["attention_mask"].squeeze(),
            "smi_labels": smi_tgt["input_ids"].squeeze(),

            "cap_input_ids": cap_enc["input_ids"].squeeze(),
            "cap_attention_mask": cap_enc["attention_mask"].squeeze(),
            "cap_labels": cap_tgt["input_ids"].squeeze(),

            "smi_raw": smi_text,
            "cap_raw": cap_text
        }


class Smi_CapDataset(Dataset):
    def __init__(self, tokenizer, filepath, max_length, prompt_prefix="caption:"):
        df = pd.read_csv(filepath, sep="\t")
        self.smi = df.iloc[:, 1].tolist()
        self.cap = df.iloc[:, 2].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_prefix = prompt_prefix

    def __len__(self):
        return len(self.smi)

    def __getitem__(self, idx):
        smi = self.smi[idx]
        cap = self.cap[idx]

        smi_tokens = list(smi)
        smi_input = f"{self.prompt_prefix} {' '.join(smi_tokens)}"

        smi_enc = self.tokenizer(smi_input, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        cap_enc = self.tokenizer(cap, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        return {
            "input_ids": smi_enc["input_ids"].squeeze(),
            "attention_mask": smi_enc["attention_mask"].squeeze(),
            "labels": cap_enc["input_ids"].squeeze(),
            "smiles": smi,
            "caption": cap
        }


class Do_TaskDataset(Dataset):
    def __init__(self, smiles_list, labels, tokenizer, max_length):
        self.smiles = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        label = self.labels[idx]
        smiles_tokens = list(smiles)
        smiles_input = "smiles: " + " ".join(smiles_tokens)

        smiles_enc = self.tokenizer(smiles_input, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        return {
            "input_ids": smiles_enc["input_ids"].squeeze(),
            "attention_mask": smiles_enc["attention_mask"].squeeze(),
            "smiles": smiles,
            "label": torch.tensor(label, dtype=torch.float)
        }