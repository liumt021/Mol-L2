import torch
import os
import argparse
from torch import nn
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration
from data.pt_ft_dataset import Smi_CapAlignedDataset, Smi_CapDataset


def contraloss(emb_a, emb_b, temperature=0.05):
    emb_a = nn.functional.normalize(emb_a, dim=1)
    emb_b = nn.functional.normalize(emb_b, dim=1)
    similarity_matrix = torch.matmul(emb_a, emb_b.T)
    labels = torch.arange(emb_a.size(0)).to(emb_a.device)
    loss_a2b = nn.functional.cross_entropy(similarity_matrix / temperature, labels)
    loss_b2a = nn.functional.cross_entropy(similarity_matrix.T / temperature, labels)
    return (loss_a2b + loss_b2a) / 2


def trains1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "cs_model"
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    dataset = Smi_CapAlignedDataset(tokenizer, folder_smi="./data/pretraining/stage1_predata1/smiles", folder_cap="./data/pretraining/stage1_predata1/text")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    save_dir = "./checkpoints/pretrain"
    os.makedirs(save_dir, exist_ok=True)
    previous_loss = float('inf')

    for epoch in range(50):
        print(f"\nEpoch {epoch + 1}")
        model.train()

        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            # 1.
            outputs_a = model(
                input_ids=batch["smi_input_ids"].to(device),
                attention_mask=batch["smi_attention_mask"].to(device),
                labels=batch["smi_labels"].to(device)
            )
            loss_a = outputs_a.loss

            # 2.
            outputs_b = model(
                input_ids=batch["b_input_ids"].to(device),
                attention_mask=batch["cap_attention_mask"].to(device),
                labels=batch["cap_labels"].to(device)
            )
            loss_b = outputs_b.loss

            # 3.
            with torch.no_grad():
                emb_smi = model.encoder(
                    input_ids=batch["smi_input_ids"].to(device),
                    attention_mask=batch["cap_attention_mask"].to(device)
                ).last_hidden_state.mean(dim=1)

                emb_cap = model.encoder(
                    input_ids=batch["b_input_ids"].to(device),
                    attention_mask=batch["cap_attention_mask"].to(device)
                ).last_hidden_state.mean(dim=1)

            loss_c = contraloss(emb_smi, emb_cap)

            total_loss = loss_a + loss_b + loss_c
            total_loss.backward()
            optimizer.step()
            current_loss = total_loss.item()

            if step % 10 == 0:
                print(f"Epoch {epoch} | Step {step} | MLM: {loss_a.item():.4f} | Denosing: {loss_b.item():.4f} | Contrast: {loss_c.item():.4f}")
                if current_loss < previous_loss:
                    previous_loss = current_loss
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(save_dir, f"t5_best_{timestamp}.pt")
                    torch.save(model.state_dict(), save_path)

def trains2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "./checkpoints/pretrain/t5_best_"
    tokenizer = AutoTokenizer.from_pretrained("cs_model", model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    dataset = Smi_CapDataset(tokenizer, filepath="./data/finetuning/ChEBI-20_data/train.txt")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    save_dir = "./checkpoints/finetune"
    os.makedirs(save_dir, exist_ok=True)
    previous_loss = float('inf')

    for epoch in range(50):
        print(f"\nEpoch {epoch + 1}")
        model.train()

        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            if step % 10 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {current_loss}")
                if current_loss < previous_loss:
                    previous_loss = current_loss
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(save_dir, f"t5_best_{timestamp}.pt")
                    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pretrain & Finetuning')
    parser.add_argument('--Pretrain', action='store_true', default=False)
    parser.add_argument('--Finetuning', action='store_true', default=False)
    args = parser.parse_args()

    if args.Pretrain:
        print('start Pretrain...')
        trains1()
    else:
        print('start Finetuning...')
        trains2()