import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import dgl
from dgl.nn.pytorch import GATConv
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
import os

# --- Configuration & Reproducibility [cite: 10] ---
CONFIG = {
    'seed': 42,
    'batch_size': 32,
    'epochs': 50,
    'lr': 0.001,
    'gnn_hidden': 128,
    'gnn_heads': 4,
    'dropout': 0.1,
    'llm_model_id': "sellergren/medgemma-2b", # Or standard Gemma if MedGemma unavailable
    'max_len': 512
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG['seed'])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. Data Preprocessing & Graph Construction ---

def smiles_to_graph(smiles):
    """
    Converts SMILES to DGL Graph with atom features + Global Descriptors.
    Ref: Manuscript Section 3.1 & 3.3 [cite: 25, 133, 162]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    
    # Calculate Molecular Descriptors (MW, LogP, TPSA) [cite: 162]
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    # Normalize roughly to keep gradients stable
    global_feats = torch.tensor([mw/500.0, logp/5.0, tpsa/100.0], dtype=torch.float)
    
    g = dgl.graph(([], []))
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)
    
    atom_feats = []
    for atom in mol.GetAtoms():
        # Atom features: AtomicNum, Degree, Aromaticity
        atom_feats.append([
            atom.GetAtomicNum(), 
            atom.GetDegree(),
            int(atom.GetIsAromatic())
        ])
    
    node_feats = torch.tensor(atom_feats, dtype=torch.float)
    
    # Append global features to every node (simple injection strategy)
    global_feats_repeated = global_feats.repeat(num_atoms, 1)
    combined_feats = torch.cat([node_feats, global_feats_repeated], dim=1) # Dim: 3 + 3 = 6
    
    g.ndata['feat'] = combined_feats
    
    # Add edges (Undirected)
    src, dst = [], []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src.extend([u, v])
        dst.extend([v, u])
    g.add_edges(src, dst)
    
    return g

# --- 2. Strict Cold-Start Splitter  ---

def get_cold_start_split(df, drug_col1='drug1_id', drug_col2='drug2_id', test_ratio=0.1):
    """
    Withholds 10% of drugs completely from training/validation.
    Test set contains ONLY pairs involving these unseen drugs.
    """
    all_drugs = list(set(df[drug_col1].unique()) | set(df[drug_col2].unique()))
    random.shuffle(all_drugs)
    
    # Withhold 10%
    split_idx = int(len(all_drugs) * test_ratio)
    unseen_drugs = set(all_drugs[:split_idx])
    seen_drugs = set(all_drugs[split_idx:])
    
    # Test Mask: Pairs where AT LEAST one drug is unseen
    test_mask = df[drug_col1].isin(unseen_drugs) | df[drug_col2].isin(unseen_drugs)
    test_df = df[test_mask].copy()
    
    # Train/Val Mask: Pairs where BOTH drugs are seen
    train_val_df = df[~test_mask].copy()
    
    # Standard split for Train/Val
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=CONFIG['seed'])
    
    print(f"Total Drugs: {len(all_drugs)}, Unseen (Cold): {len(unseen_drugs)}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Cold-Start Test: {len(test_df)}")
    
    return train_df, val_df, test_df

# --- 3. Dataset & Dataloader ---

class DDIDataset(Dataset):
    def __init__(self, df, llm_embeddings_map):
        self.df = df.reset_index(drop=True)
        self.llm_embeddings_map = llm_embeddings_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        g1 = smiles_to_graph(row['drug1_smiles'])
        g2 = smiles_to_graph(row['drug2_smiles'])
        
        # Retrieve pre-computed embedding by Pair ID or Index
        # Using index here for simplicity, assuming strict order
        llm_emb = self.llm_embeddings_map[idx] 
        
        label = torch.tensor(row['label'], dtype=torch.float)
        return g1, g2, llm_emb, label

def collate_fn(batch):
    g1s, g2s, llm_embs, labels = zip(*batch)
    bg1 = dgl.batch(g1s)
    bg2 = dgl.batch(g2s)
    llm_embs = torch.stack(llm_embs)
    labels = torch.stack(labels)
    return bg1, bg2, llm_embs, labels

# --- 4. Pre-compute LLM Embeddings ---

def precompute_llm_embeddings(df, tokenizer, model):
    """
    Extracts sentence-level embeddings from MedGemma[cite: 4, 19].
    Frozen model, no gradients.
    """
    print("Pre-computing LLM embeddings...")
    embeddings = []
    model.eval()
    model.to(DEVICE)
    
    batch_size = 32
    texts = df['text_description'].tolist()
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, 
                               truncation=True, max_length=CONFIG['max_len']).to(DEVICE)
            # Use last_hidden_state mean
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).cpu() # [Batch, 1024]
            embeddings.append(emb)
            
    return torch.cat(embeddings)

# --- 5. Model Architecture [cite: 8, 9, 161] ---

class GATModel(nn.Module):
    def __init__(self, in_feats=6, hidden_dim=128, out_dim=256, num_heads=4):
        super(GATModel, self).__init__()
        # Multi-head GAT layers [cite: 9]
        self.conv1 = GATConv(in_feats, hidden_dim // num_heads, num_heads, allow_zero_in_degree=True)
        self.conv2 = GATConv(hidden_dim, hidden_dim // num_heads, num_heads, allow_zero_in_degree=True)
        self.conv3 = GATConv(hidden_dim, out_dim // num_heads, num_heads, allow_zero_in_degree=True)
        self.dropout = nn.Dropout(CONFIG['dropout'])

    def forward(self, g):
        h = g.ndata['feat']
        h = self.conv1(g, h).flatten(1)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h).flatten(1)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.conv3(g, h).flatten(1) # [Nodes, 256]
        
        # Graph-level readout (Mean pooling)
        with g.local_scope():
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h') # [Batch, 256]
            return hg

class CrossAttentionFusion(nn.Module):
    def __init__(self, gnn_dim=512, llm_dim=1024, common_dim=512):
        super(CrossAttentionFusion, self).__init__()
        # Projections to Common Space [cite: 8]
        self.proj_gnn = nn.Linear(gnn_dim, common_dim)
        self.proj_llm = nn.Linear(llm_dim, common_dim)
        
        # Cross-Attention: Q=LLM, K/V=GNN 
        self.attn = nn.MultiheadAttention(embed_dim=common_dim, num_heads=8, batch_first=True)
        
        self.norm = nn.LayerNorm(common_dim)
        self.residual = nn.Identity()

    def forward(self, gnn_emb, llm_emb):
        # gnn_emb: [Batch, 512] (Concat of 2 drugs)
        # llm_emb: [Batch, 1024]
        
        # 1. Project and reshape for Attention (Batch, Seq=1, Dim)
        gnn_proj = self.proj_gnn(gnn_emb).unsqueeze(1) # K, V source
        llm_proj = self.proj_llm(llm_emb).unsqueeze(1) # Q source
        
        # 2. Attention(Q, K, V)
        # Query comes from LLM (Semantic), Key/Value from GNN (Structural)
        attn_out, _ = self.attn(query=llm_proj, key=gnn_proj, value=gnn_proj)
        
        # 3. Add & Norm (Residual connection to semantic query)
        fused = self.norm(attn_out + self.residual(llm_proj))
        
        return fused.squeeze(1)

class DDILLM(nn.Module):
    def __init__(self):
        super(DDILLM, self).__init__()
        self.gnn = GATModel(in_feats=6, hidden_dim=CONFIG['gnn_hidden'], 
                            num_heads=CONFIG['gnn_heads']) # 6 = 3 atom + 3 global feats
        self.fusion = CrossAttentionFusion()
        
        # Final Classifier [cite: 156]
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(CONFIG['dropout']),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, g1, g2, llm_emb):
        # Get structural embeddings
        h1 = self.gnn(g1) # [Batch, 256]
        h2 = self.gnn(g2) # [Batch, 256]
        gnn_pair = torch.cat([h1, h2], dim=1) # [Batch, 512]
        
        # Fuse with textual embeddings
        fused = self.fusion(gnn_pair, llm_emb) # [Batch, 512]
        
        return self.classifier(fused)

# --- 6. Training & Evaluation ---

def evaluate(model, loader, criterion):
    model.eval()
    loss_accum = 0
    preds_all, labels_all = [], []
    
    with torch.no_grad():
        for g1, g2, llm_emb, label in loader:
            g1, g2 = g1.to(DEVICE), g2.to(DEVICE)
            llm_emb, label = llm_emb.to(DEVICE), label.to(DEVICE)
            
            output = model(g1, g2, llm_emb).squeeze()
            loss = criterion(output, label)
            loss_accum += loss.item()
            
            preds_all.extend(output.cpu().numpy())
            labels_all.extend(label.cpu().numpy())
            
    preds_binary = np.array(preds_all) > 0.5
    
    metrics = {
        'loss': loss_accum / len(loader),
        'precision': precision_score(labels_all, preds_binary, zero_division=0),
        'recall': recall_score(labels_all, preds_binary, zero_division=0),
        'f1': f1_score(labels_all, preds_binary, zero_division=0),
        'auc': roc_auc_score(labels_all, preds_all)
    }
    return metrics

def main():
    # --- Load Data ---
    # Replace with actual file path
    if not os.path.exists('ddi_data.csv'):
        print("Error: 'ddi_data.csv' not found. Please provide dataset.")
        return
        
    df = pd.read_csv('ddi_data.csv') # Required cols: drug1_smiles, drug2_smiles, text_description, label
    
    # --- Cold-Start Split ---
    train_df, val_df, test_df = get_cold_start_split(df)
    
    # --- LLM Embedding Pre-computation ---
    # Load Tokenizer & Model (Frozen)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['llm_model_id'])
    llm_model = AutoModel.from_pretrained(CONFIG['llm_model_id'])
    
    # Map embeddings to dataframe indices to handle splits correctly
    # We compute for specific DFs to keep order clean
    train_emb = precompute_llm_embeddings(train_df, tokenizer, llm_model)
    val_emb = precompute_llm_embeddings(val_df, tokenizer, llm_model)
    test_emb = precompute_llm_embeddings(test_df, tokenizer, llm_model)
    
    # Free up memory (Unload LLM)
    del llm_model
    torch.cuda.empty_cache()
    
    # --- Dataloaders ---
    train_loader = DataLoader(DDIDataset(train_df, train_emb), batch_size=CONFIG['batch_size'], 
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(DDIDataset(val_df, val_emb), batch_size=CONFIG['batch_size'], 
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(DDIDataset(test_df, test_emb), batch_size=CONFIG['batch_size'], 
                             shuffle=False, collate_fn=collate_fn)
    
    # --- Initialize Model ---
    model = DDILLM().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.BCELoss()
    
    print("Starting Training...")
    best_f1 = 0
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        for g1, g2, llm_emb, label in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            g1, g2 = g1.to(DEVICE), g2.to(DEVICE)
            llm_emb, label = llm_emb.to(DEVICE), label.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(g1, g2, llm_emb).squeeze()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        val_metrics = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}: Loss {train_loss/len(train_loader):.4f} | Val F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), 'best_model.pth')
            
    # --- Final Cold-Start Evaluation ---
    print("\n--- Cold-Start Evaluation (Unseen Drugs) ---")
    model.load_state_dict(torch.load('best_model.pth'))
    test_metrics = evaluate(model, test_loader, criterion)
    
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1-Score:  {test_metrics['f1']:.4f}")
    print(f"AUC:       {test_metrics['auc']:.4f}")

if __name__ == "__main__":
    main()
