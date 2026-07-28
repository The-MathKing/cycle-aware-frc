import os
import zipfile
import urllib.request
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.nn import LightGCN
from torch_geometric.utils import structured_negative_sampling
from sklearn.metrics import roc_auc_score
from scipy.sparse.linalg import svds
import pandas as pd
import gzip
import json

from compute_frc import compute_signed_frc

def download_datasets():
    print("Downloading Amazon Video Games 5-core...")
    # SNAP McAuley dataset (5-core video games)
    amz_url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games_5.json.gz"
    if not os.path.exists("amazon_vg.json.gz"):
        urllib.request.urlretrieve(amz_url, "amazon_vg.json.gz")
        
    print("Amazon Video Games downloaded.")

def load_amazon():
    print("Parsing Amazon Video Games...")
    user_map = {}
    item_map = {}
    rows, cols, data = [], [], []
    
    # Process gzipped json line by line
    with gzip.open("amazon_vg.json.gz", 'rt') as f:
        for line in f:
            js = json.loads(line)
            u = js.get('reviewerID')
            i = js.get('asin')
            r = js.get('overall')
            if u and i and r is not None:
                if r >= 4.0: # Positive edge
                    if u not in user_map: user_map[u] = len(user_map)
                    if i not in item_map: item_map[i] = len(item_map)
                    rows.append(user_map[u])
                    cols.append(item_map[i])
                    data.append(1)
                    
    num_users = len(user_map)
    cols = [c + num_users for c in cols] # Shift items
    return rows, cols, data, num_users

def load_ml1m():
    print("Parsing ML-1M...")
    rows, cols, data = [], [], []
    num_users = 6040 + 10 
    
    zip_path = "ml-1m.zip"
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve("https://files.grouplens.org/datasets/movielens/ml-1m.zip", zip_path)
        
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open('ml-1m/ratings.dat') as f:
            for line in f:
                parts = line.decode('utf-8').strip().split('::')
                if len(parts) >= 3:
                    u = int(parts[0]) - 1
                    i = int(parts[1]) - 1 + num_users
                    r = int(parts[2])
                    if r >= 4:
                        rows.append(u)
                        cols.append(i)
                        data.append(1)
    return rows, cols, data, num_users

def run_pipeline(dataset_name, rows, cols, data, num_users):
    print(f"\n--- Running Pipeline for {dataset_name} ---")
    
    # Make symmetric and unique
    n = max(max(rows), max(cols)) + 1
    edge_set = set(zip(rows, cols))
    
    # Generate 10% test mask
    pos_edges = list(edge_set)
    np.random.shuffle(pos_edges)
    test_size = len(pos_edges) // 10
    
    test_pos = pos_edges[:test_size]
    train_pos = pos_edges[test_size:]
    
    # Remove masked edges strictly to prevent leakage
    test_pos_set = set(test_pos)
    train_rows, train_cols, train_data = [], [], []
    
    for u, v in train_pos:
        train_rows.extend([u, v])
        train_cols.extend([v, u])
        train_data.extend([1, 1])
        
    print(f"Graph nodes: {n}, Train edges (symmetric): {len(train_data)}")
    
    A_train = sp.csr_matrix((train_data, (train_rows, train_cols)), shape=(n, n))
    
    # Generate Negative testing set
    test_neg = []
    while len(test_neg) < test_size:
        u = np.random.randint(0, num_users)
        v = np.random.randint(num_users, n)
        if (u, v) not in edge_set:
            test_neg.append((u, v))
            
    # --- 1. Compute Truncated SVD Baseline ---
    print("1. Computing SVD...")
    A_float = A_train.asfptype()
    u_svd, s_svd, vt_svd = svds(A_float, k=64)
    user_emb_svd = u_svd * np.sqrt(s_svd)
    item_emb_svd = vt_svd.T * np.sqrt(s_svd)
    
    # --- 2. Compute F* Cycle-Aware Curvature ---
    print("2. Computing F*...")
    A_sq = A_train.dot(A_train)
    d = np.array(np.abs(A_train).sum(axis=1)).flatten()
    
    # --- 3. Train LightGCN Baseline ---
    print("3. Training LightGCN (PyTorch Geometric)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert training graph to edge_index
    edge_index = torch.tensor([train_rows, train_cols], dtype=torch.long).to(device)
    
    model = LightGCN(
        num_nodes=n,
        embedding_dim=64,
        num_layers=2,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Fast BPR training loop for local simulation (50 epochs)
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        # Compute embeddings for all nodes
        emb = model(edge_index)
        
        # sample negatives
        out = structured_negative_sampling(edge_index, num_nodes=n)
        i, j, k = out 
        
        # calculate BPR loss
        pos_rank = emb[i] * emb[j]
        neg_rank = emb[i] * emb[k]
        
        pos_score = pos_rank.sum(dim=-1)
        neg_score = neg_rank.sum(dim=-1)
        
        loss = -F.logsigmoid(pos_score - neg_score).mean()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f" Epoch {epoch+1:02d}, Loss: {loss.item():.4f}")
            
    model.eval()
    with torch.no_grad():
        all_emb = model.get_embedding(edge_index)
        
    print("Scoring Link Prediction...")
    
    def score_svd(u, v):
        return np.dot(user_emb_svd[u], item_emb_svd[v])
        
    def score_gcn(u, v):
        return torch.dot(all_emb[u], all_emb[v]).item()
        
    def score_fstar(u, v):
        A3_uv = A_train[u, :].dot(A_sq[:, v])[0, 0]
        du, dv = d[u], d[v]
        if du == 0 or dv == 0:
            return 0
        C4_sum = 1 * A3_uv - du - dv + 1
        max_bound = max(1, (du - 1) * (dv - 1))
        normalized = C4_sum / max_bound
        return 4 - du - dv + 3.0 * min(du - 1, dv - 1) * normalized
        
    pos_scores_svd = [score_svd(u,v) for u,v in test_pos]
    neg_scores_svd = [score_svd(u,v) for u,v in test_neg]
    
    pos_scores_gcn = [score_gcn(u,v) for u,v in test_pos]
    neg_scores_gcn = [score_gcn(u,v) for u,v in test_neg]
    
    # F* metric is inversely correlated with edge presence
    pos_scores_fs = [-score_fstar(u,v) for u,v in test_pos]
    neg_scores_fs = [-score_fstar(u,v) for u,v in test_neg]
    
    y_true = [1]*test_size + [0]*test_size
    
    y_svd = pos_scores_svd + neg_scores_svd
    y_gcn = pos_scores_gcn + neg_scores_gcn
    y_fs = pos_scores_fs + neg_scores_fs
    
    auc_svd = roc_auc_score(y_true, y_svd)
    auc_gcn = roc_auc_score(y_true, y_gcn)
    auc_fs = roc_auc_score(y_true, y_fs)
    
    print(f"Results for {dataset_name}:")
    print(f" SVD AUC: {auc_svd:.4f}")
    print(f" LightGCN AUC: {auc_gcn:.4f}")
    print(f" F* (Inverse) AUC: {auc_fs:.4f}")
    
    return auc_svd, auc_gcn, auc_fs

if __name__ == "__main__":
    download_datasets()
    
    # Amazon
    a_rows, a_cols, a_data, a_num_users = load_amazon()
    a_svd, a_gcn, a_fs = run_pipeline("Amazon Video Games", a_rows, a_cols, a_data, a_num_users)
    
    # ML-1M
    m_rows, m_cols, m_data, m_num_users = load_ml1m()
    m_svd, m_gcn, m_fs = run_pipeline("MovieLens-1M", m_rows, m_cols, m_data, m_num_users)
    
    with open("final_results.txt", "w") as f:
        f.write("Dataset,SVD,LightGCN,F_star\n")
        f.write(f"Amazon Video Games,{a_svd:.4f},{a_gcn:.4f},{a_fs:.4f}\n")
        f.write(f"MovieLens-1M,{m_svd:.4f},{m_gcn:.4f},{m_fs:.4f}\n")
    print("Saved final_results.txt")
