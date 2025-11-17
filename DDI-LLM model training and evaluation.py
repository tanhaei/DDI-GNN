# main.py - Main script for DDI-LLM model training and evaluation

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import dgl
from dgl.nn.pytorch import GATConv
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import random

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Dataset class for DDI pairs
class DDIDataset(Dataset):
    def __init__(self, df, tokenizer, gnn_model, llm_model, device):
        self.df = df
        self.tokenizer = tokenizer
        self.gnn_model = gnn_model
        self.llm_model = llm_model
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        drug1_smiles = row['drug1_smiles']
        drug2_smiles = row['drug2_smiles']
        text = row['text_description']  # Assume column with pair description
        label = row['label']

        # GNN embeddings
        g1 = smiles_to_graph(drug1_smiles)
        g2 = smiles_to_graph(drug2_smiles)
        gnn_emb1 = self.gnn_model(g1).mean(dim=0)  # Pool to 256 dim
        gnn_emb2 = self.gnn_model(g2).mean(dim=0)
        gnn_emb = torch.cat([gnn_emb1, gnn_emb2], dim=0)  # 512 dim

        # LLM embeddings (frozen MedGemma)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            llm_emb = self.llm_model(**inputs).last_hidden_state.mean(dim=1).squeeze()  # 1024 dim

        return gnn_emb, llm_emb, label

# Function to convert SMILES to DGL graph
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    g = dgl.graph(([], []))
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)
    
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append([atom.GetAtomicNum(), atom.GetDegree()])  # Simple features
    
    g.ndata['feat'] = torch.tensor(atom_feats, dtype=torch.float)
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        g.add_edges(i, j)
        g.add_edges(j, i)  # Undirected
    
    return g

# GNN Model: GAT
class GATModel(nn.Module):
    def __init__(self, in_feats=2, hidden_dim=128, out_dim=256, num_heads=4):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_feats, hidden_dim // num_heads, num_heads=num_heads)
        self.conv2 = GATConv(hidden_dim, hidden_dim // num_heads, num_heads=num_heads)
        self.conv3 = GATConv(hidden_dim, out_dim // num_heads, num_heads=num_heads)
        self.dropout = nn.Dropout(0.1)

    def forward(self, g):
        h = g.ndata['feat']
        h = self.conv1(g, h).flatten(1)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h).flatten(1)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.conv3(g, h).flatten(1)
        return h  # Node embeddings, will pool later

# Cross-Attention Fusion Module
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=512):
        super(CrossAttentionFusion, self).__init__()
        self.proj_gnn = nn.Linear(512, dim)  # GNN pair emb to 512
        self.proj_llm = nn.Linear(1024, dim)  # LLM to 512
        self.attn = nn.MultiheadAttention(dim, num_heads=8)
        self.norm = nn.LayerNorm(dim)
        self.residual = nn.Identity()

    def forward(self, gnn_emb, llm_emb):
        gnn_proj = self.proj_gnn(gnn_emb.unsqueeze(0))  # [1, dim]
        llm_proj = self.proj_llm(llm_emb.unsqueeze(0))  # [1, dim]
        
        # Cross-attention: Q from LLM, K/V from GNN
        attn_output, _ = self.attn(llm_proj, gnn_proj, gnn_proj)
        output = self.norm(attn_output + self.residual(llm_proj))  # Residual on LLM
        return output.squeeze(0)

# Full DDI-LLM Model
class DDILLM(nn.Module):
    def __init__(self):
        super(DDILLM, self).__init__()
        self.gnn = GATModel()
        self.llm = AutoModel.from_pretrained("sellergren/medgemma-2b")  # Frozen
        for param in self.llm.parameters():
            param.requires_grad = False
        self.fusion = CrossAttentionFusion()
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, gnn_emb, llm_emb):
        fused = self.fusion(gnn_emb, llm_emb)
        return self.classifier(fused)

# Training function
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        for gnn_emb, llm_emb, label in train_loader:
            gnn_emb, llm_emb, label = gnn_emb.to(device), llm_emb.to(device), label.to(device).float()
            output = model(gnn_emb, llm_emb)
            loss = criterion(output, label.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            preds, labels = [], []
            for gnn_emb, llm_emb, label in val_loader:
                gnn_emb, llm_emb = gnn_emb.to(device), llm_emb.to(device)
                output = model(gnn_emb, llm_emb)
                preds.append(output.cpu().numpy())
                labels.append(label.numpy())
            preds = np.concatenate(preds)
            labels = np.concatenate(labels)
            print(f"Epoch {epoch}: F1 {f1_score(labels, preds > 0.5)}")

# Cold-start evaluation function
def cold_start_eval(model, test_df):
    # Assume test_df has unseen drugs
    test_dataset = DDIDataset(test_df, tokenizer, model.gnn, model.llm, device)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model.eval()
    with torch.no_grad():
        preds, labels = [], []
        for gnn_emb, llm_emb, label in test_loader:
            gnn_emb, llm_emb = gnn_emb.to(device), llm_emb.to(device)
            output = model(gnn_emb, llm_emb)
            preds.append(output.cpu().numpy())
            labels.append(label.numpy())
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        precision = precision_score(labels, preds > 0.5)
        recall = recall_score(labels, preds > 0.5)
        auc = roc_auc_score(labels, preds)
        print(f"Cold-start: Precision {precision}, Recall {recall}, AUC {auc}")

# Main execution
if __name__ == "__main__":
    # Load data (assume CSV with drug1_smiles, drug2_smiles, text_description, label)
    df = pd.read_csv('drugbank_ddi.csv')
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Cold-start: Filter test_df to unseen drugs (simulate by removing 10% rare drugs from train/val)
    # ... logic to withhold 10% rare drugs ...
    
    tokenizer = AutoTokenizer.from_pretrained("sellergren/medgemma-2b")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DDILLM()
    
    train_dataset = DDIDataset(train_df, tokenizer, model.gnn, model.llm, device)
    val_dataset = DDIDataset(val_df, tokenizer, model.gnn, model.llm, device)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    train_model(model, train_loader, val_loader)
    
    cold_start_eval(model, test_df)  # Also compute standard metrics on test_df

# Additional ablations can be run by modifying components, e.g., GNN-only by bypassing fusion
