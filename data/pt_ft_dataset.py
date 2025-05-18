import torch
import random
import os
from torch.utils.data import Dataset
import pandas as pd


class Smi_CapAlignedDataset(Dataset):

    def __init__(self, tokenizer, folder_smi, folder_cap, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        smi_files = sorted(set(os.listdir(folder_smi)))
        cap_files = sorted(set(os.listdir(folder_cap)))
        self.common_files = sorted(set(smi_files) & set(cap_files))

        self.folder_smi = folder_smi
        self.folder_cap = folder_cap


    def __len__(self):
        return len(self.common_files)

    def __getitem__(self, idx):

        fname = self.common_files[idx]
        smi_path = os.path.join(self.folder_smi, fname)
        cap_path = os.path.join(self.folder_cap, fname)

        with open(smi_path, 'r', encoding='utf-8') as f:
            smi_text = f.readline().strip()

        with open(cap_path, 'r', encoding='utf-8') as f:
            cap_lines = f.read().strip().splitlines()
        cap_text = random.choice(cap_lines)

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

    def mask_smiles_text(self, smiles, mask_prob=0.15):
        tokens = list(smiles)
        target = tokens[:]

        num_tokens = len(tokens)
        num_to_mask = int(mask_prob * num_tokens)

        mask_indices = sorted(random.sample(range(num_tokens), num_to_mask))

        for i, idx in enumerate(mask_indices):
            target[idx] = f"<extra_id_{i}>"

        masked_smiles = "".join(target)

        target_smiles = []
        for i in range(num_to_mask):
            target_smiles.append(f"<extra_id_{i}> {tokens[mask_indices[i]]}")

        target_smiles.append(f"<extra_id_{num_to_mask}>")  # Add the final <extra_id_x> for the end
        target_smiles = " ".join(target_smiles)

        return masked_smiles, target_smiles


    def mask_text(self, text, mask_prob=0.15):

        words = text.split()
        target = words[:]

        num_words = len(words)
        num_to_mask = int(mask_prob * num_words)

        mask_indices = sorted(random.sample(range(num_words), num_to_mask))

        for i, idx in enumerate(mask_indices):
            target[idx] = f"<extra_id_{i}>"

        masked_text = " ".join(target)

        target_text = []
        for i in range(num_to_mask):
            target_text.append(f"<extra_id_{i}> {words[mask_indices[i]]}")

        target_text.append(f"<extra_id_{num_to_mask}>")  # Add the final <extra_id_x> for the end
        target_text = " ".join(target_text)

        return masked_text, target_text


class Smi_CapDataset(Dataset):
    def __init__(self, tokenizer, filepath, max_length=512, prompt_prefix="caption:"):
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