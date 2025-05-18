import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedModel(nn.Module):
    def __init__(self, gnn_model, t5_model, alpha=0.5, combined_dim=256):
        super().__init__()
        self.gnn_model = gnn_model
        self.t5_model = t5_model
        self.alpha = alpha

        self.fusion_proj = nn.Sequential(
            nn.Linear(gnn_model.feat_dim, combined_dim),
            nn.ReLU(),
            nn.Linear(combined_dim, combined_dim // 2)
        )

        self.property_predictor = nn.Sequential(
            nn.Linear(combined_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.ddi_predictor = nn.Sequential(
            nn.Linear((combined_dim // 2) * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def encode(self, graph_data):
        input_texts = graph_data.text
        gnn_feat, _ = self.gnn_model(graph_data)
        t5_feat = self.t5_model(input_texts)

        if gnn_feat.size(1) != t5_feat.size(1):
            raise ValueError("Feature dimensions from GNN and T5 do not match")

        combined = self.alpha * gnn_feat + (1 - self.alpha) * t5_feat
        return self.fusion_proj(combined)

    def forward(self, graph_data, graph_data_2=None):
        if graph_data_2 is None:
            fusion = self.encode(graph_data)
            return None, self.property_predictor(fusion)
        else:
            fusion1 = self.encode(graph_data)
            fusion2 = self.encode(graph_data_2)
            pair = torch.cat([fusion1, fusion2], dim=-1)
            return None, self.ddi_predictor(pair).squeeze(-1)

    def predict_properties(self, graph_data):
        _, out = self.forward(graph_data)
        return out

    def predict_ddi(self, data1, data2):
        _, out = self.forward(data1, data2)
        return out
