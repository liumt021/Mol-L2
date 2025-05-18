import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration

class T5EncoderProjection(nn.Module):

    def __init__(self, model_name, projection_dim=128, dropout=0.1):

        super().__init__()
        self.encoder = T5ForConditionalGeneration.from_pretrained(model_name).encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, projection_dim)
        )

    def forward(self, input_texts):

        inputs = self.tokenizer.batch_encode_plus(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        input_ids = inputs["input_ids"].to(next(self.parameters()).device)
        attention_mask = inputs["attention_mask"].to(next(self.parameters()).device)

        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = encoder_outputs.last_hidden_state

        pooled = (last_hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

        projected = self.proj(pooled)
        return projected