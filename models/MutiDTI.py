from T5MN import T5EncoderProjection
import random
from dgl.nn.pytorch.conv import GATConv, GraphConv, TAGConv, GINConv, APPNPConv
from dgl.nn import TWIRLSConv
from dgl.nn.pytorch.glob import MaxPooling, GlobalAttentionPooling
import torch
from torch.autograd import Variable
from layers import MultiHeadAttention
import torch.nn as nn
import torch.nn.functional as F
import time
from dgl import unbatch

class AttentionDTI(nn.Module):
    def __init__(self, model_name, protein_MAX_LENGH = 1000,):
        super(AttentionDTI, self).__init__()
        self.dim = 64
        self.conv = 40

        self.T5EP = T5EncoderProjection(model_name, projection_dim=self.conv*4, dropout=0.1)

        self.protein_MAX_LENGH = protein_MAX_LENGH
        self.protein_kernel = [4, 8, 12]

        self.protein_embed = nn.Embedding(26, self.dim, padding_idx=0)

        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4, kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
        self.Protein_max_pool = nn.MaxPool1d(self.protein_MAX_LENGH - self.protein_kernel[0] - self.protein_kernel[1] - self.protein_kernel[2] + 3)
        self.attention_layer = nn.Linear(self.conv*4, self.conv*4)
        self.protein_attention_layer = nn.Linear(self.conv * 4, self.conv * 4)
        self.drug_attention_layer = nn.Linear(self.conv * 4, self.conv * 4)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.conv*8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

    def forward(self, drug_texts, protein_seq):

        drug_feature = self.t5_encoder(drug_texts)
        drugConv = drug_feature.unsqueeze(2)

        proteinembed = self.protein_embed(protein_seq).permute(0, 2, 1)
        proteinConv = self.Protein_CNNs(proteinembed)

        drug_att = self.drug_attention_layer(drugConv.permute(0, 2, 1))
        protein_att = self.protein_attention_layer(proteinConv.permute(0, 2, 1))

        d_att_layers = drug_att.unsqueeze(2).repeat(1, 1, proteinConv.shape[-1], 1)
        p_att_layers = protein_att.unsqueeze(1).repeat(1, 1, 1, 1)
        Atten_matrix = self.attention_layer(self.relu(d_att_layers + p_att_layers))

        Compound_atte = torch.mean(Atten_matrix, 2)
        Protein_atte = torch.mean(Atten_matrix, 1)
        Compound_atte = self.sigmoid(Compound_atte.permute(0, 2, 1))
        Protein_atte = self.sigmoid(Protein_atte.permute(0, 2, 1))

        drugConv = drugConv * 0.5 + drugConv * Compound_atte
        proteinConv = proteinConv * 0.5 + proteinConv * Protein_atte

        drugConv = drugConv.squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)

        pair = torch.cat([drugConv, proteinConv], dim=1)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)

        return predict

class DTIModelWithoutBatching(nn.Module):
    def __init__(self, model_name):
        super(DTIModelWithoutBatching, self).__init__()

        self.protein_graph_conv = nn.ModuleList()
        for i in range(14):
            self.protein_graph_conv.append(
                TWIRLSConv(31, 31, 8, prop_step=7, alp=1, lam=1, attention=True,
                           num_mlp_before=2, num_mlp_after=0, dropout=0.3)
            )

        self.T5EP = T5EncoderProjection(model_name=model_name, projection_dim=31)

        self.pooling_ligand = nn.Linear(31, 1)
        self.pooling_protein = nn.Linear(31, 1)

        self.dropout = 0.3
        self.bilstm = nn.LSTM(31, 31, num_layers=2, bidirectional=True, dropout=0.3)
        self.fc_in = nn.Linear(8680, 4340)
        self.fc_out = nn.Linear(4340, 1)
        self.attention = MultiHeadAttention(62, 62, 2)

    def forward(self, g, drug_texts):
        feature_protein = g[0].ndata['h']

        for module in self.protein_graph_conv:
            feature_protein = F.relu(module(g[0], feature_protein))

        ligand_rep = self.T5EP(drug_texts).view(-1, 31)

        pool_protein = GlobalAttentionPooling(self.pooling_protein)
        protein_rep = pool_protein(g[0], feature_protein).view(-1, 31)

        sequence = torch.cat((ligand_rep, protein_rep), dim=0).view(1, -1, 31)

        mask = torch.eye(140, dtype=torch.uint8).view(1, 140, 140).to(sequence.device)
        mask[0, sequence.size()[1]:140, :] = 0
        mask[0, :, sequence.size()[1]:140] = 0
        mask[0, :, sequence.size()[1] - 1] = 1
        mask[0, sequence.size()[1] - 1, :] = 1
        mask[0, sequence.size()[1] - 1, sequence.size()[1] - 1] = 0

        sequence = F.pad(input=sequence, pad=(0, 0, 0, 140 - sequence.size()[1]), mode='constant', value=0)
        sequence = sequence.permute(1, 0, 2)

        h_0 = Variable(torch.zeros(4, 1, 31).to(sequence.device))
        c_0 = Variable(torch.zeros(4, 1, 31).to(sequence.device))

        output, _ = self.bilstm(sequence, (h_0, c_0))
        output = output.permute(1, 0, 2)

        out = self.attention(output, mask=mask)
        out = F.relu(self.fc_in(out.view(-1, out.size()[1] * out.size()[2])))
        out = torch.sigmoid(self.fc_out(out))

        return out


class DTITAG(nn.Module):
    def __init__(self, model_name):
        super(DTITAG, self).__init__()

        self.protein_graph_conv = nn.ModuleList()
        for _ in range(5):
            self.protein_graph_conv.append(TAGConv(31, 31, 2))

        self.t5EP = T5EncoderProjection(model_name=model_name, projection_dim=31)

        self.pooling_ligand = nn.Linear(31, 1)
        self.pooling_protein = nn.Linear(31, 1)

        self.dropout = 0.2
        self.bilstm = nn.LSTM(31, 31, num_layers=1, bidirectional=True, dropout=self.dropout)
        self.fc_in = nn.Linear(8680, 4340)
        self.fc_out = nn.Linear(4340, 1)
        self.attention = MultiHeadAttention(62, 62, 2)

    def forward(self, g, drug_texts):
        """
        g[0]: protein graph (DGLGraph with ndata['h'])
        drug_texts: list of SMILES strings
        """

        feature_protein = g[0].ndata['h']
        for module in self.protein_graph_conv:
            feature_protein = F.relu(module(g[0], feature_protein))

        ligand_rep = self.t5EP(drug_texts).view(-1, 31)

        pool_protein = GlobalAttentionPooling(self.pooling_protein)
        protein_rep = pool_protein(g[0], feature_protein).view(-1, 31)

        sequence = torch.cat((ligand_rep, protein_rep), dim=0).view(1, -1, 31)

        mask = torch.eye(140, dtype=torch.uint8).view(1, 140, 140).to(sequence.device)
        mask[0, sequence.size()[1]:140, :] = 0
        mask[0, :, sequence.size()[1]:140] = 0
        mask[0, :, sequence.size()[1] - 1] = 1
        mask[0, sequence.size()[1] - 1, :] = 1
        mask[0, sequence.size()[1] - 1, sequence.size()[1] - 1] = 0

        sequence = F.pad(input=sequence, pad=(0, 0, 0, 140 - sequence.size()[1]), mode='constant', value=0)
        sequence = sequence.permute(1, 0, 2)

        h_0 = Variable(torch.zeros(2, 1, 31).to(sequence.device))
        c_0 = Variable(torch.zeros(2, 1, 31).to(sequence.device))

        output, _ = self.bilstm(sequence, (h_0, c_0))
        output = output.permute(1, 0, 2)

        out = self.attention(output, mask=mask)
        out = F.relu(self.fc_in(out.view(-1, out.size(1) * out.size(2))))
        out = torch.sigmoid(self.fc_out(out))

        return out