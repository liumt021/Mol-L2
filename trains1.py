import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from data_loader import Smi_CapAlignedDataset


def contraloss(emb_a, emb_b, temperature=0.05):
    emb_a = nn.functional.normalize(emb_a, dim=1)
    emb_b = nn.functional.normalize(emb_b, dim=1)
    similarity_matrix = torch.matmul(emb_a, emb_b.T)
    labels = torch.arange(emb_a.size(0)).to(emb_a.device)
    loss_a2b = nn.functional.cross_entropy(similarity_matrix / temperature, labels)
    loss_b2a = nn.functional.cross_entropy(similarity_matrix.T / temperature, labels)
    return (loss_a2b + loss_b2a) / 2


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    dataset = Smi_CapAlignedDataset(tokenizer, filepath="./datav3.txt")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(50):
        print(f"\nEpoch {epoch + 1}")
        model.train()

        for step, batch in enumerate(loader):
            # 1. MLM
            outputs_a = model(
                input_ids=batch["smi_input_ids"].to(device),
                attention_mask=batch["smi_attention_mask"].to(device),
                labels=batch["smi_labels"].to(device)
            )
            loss_a = outputs_a.loss

            # 2. Denoising
            outputs_b = model(
                input_ids=batch["b_input_ids"].to(device),
                attention_mask=batch["cap_attention_mask"].to(device),
                labels=batch["cap_labels"].to(device)
            )
            loss_b = outputs_b.loss

            # 3. Contrastive
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
            optimizer.zero_grad()

            if step % 10 == 0:
                print(f"Step {step} | MLM: {loss_a.item():.4f} | Denosing: {loss_b.item():.4f} | Contrast: {loss_c.item():.4f}")
