import numpy as np
import scipy.sparse as sp
import time
import urllib.request
import zipfile
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def compute_signed_frc(A: sp.csr_matrix, gamma=3.0) -> np.ndarray:
    n = A.shape[0]
    A_sq = A.dot(A)
    # Extract structural edges
    u, v = A.nonzero()
    edge_weights = np.array(A[u, v]).flatten()
    
    # Degrees
    d = np.array(np.abs(A).sum(axis=1)).flatten()
    du = d[u]
    dv = d[v]
    
    # Native extraction of A^3 at existing edges via row masking
    A3_uv = np.zeros(len(u))
    for i in range(len(u)):
        row = u[i]
        col = v[i]
        A3_uv[i] = A[row, :].dot(A_sq[:, col])[0, 0]
        
    # Non-overcounting formula: A_uv (A^3)_uv - d(u) - d(v) + 1
    C4_sum = edge_weights * A3_uv - du - dv + 1
    
    # Normalization bound: max(1, (du-1)(dv-1))
    max_bound = np.maximum(1, (du - 1) * (dv - 1))
    normalized_term = C4_sum / max_bound
    
    F_star = 4 - du - dv + gamma * np.minimum(du - 1, dv - 1) * normalized_term
    return F_star

def generate_sbm_sweep():
    print("Running Stochastic Block Model (SBM) Noise Sweep...")
    # Bipartite SBM: 2 blocks of users (U1, U2), 2 blocks of items (V1, V2)
    n_u, n_v = 100, 100
    n = n_u + n_v
    
    noise_levels = np.linspace(0, 0.5, 10)
    variances = []
    
    for noise in noise_levels:
        data, rows, cols = [], [], []
        
        # Block 1
        for i in range(50):
            for j in range(50):
                if np.random.rand() < 0.2: # 20% density
                    w = 1 if np.random.rand() > noise else -1
                    u_idx, v_idx = i, j + n_u
                    rows.extend([u_idx, v_idx])
                    cols.extend([v_idx, u_idx])
                    data.extend([w, w])
                    
        # Block 2
        for i in range(50, 100):
            for j in range(50, 100):
                if np.random.rand() < 0.2:
                    w = 1 if np.random.rand() > noise else -1
                    u_idx, v_idx = i, j + n_u
                    rows.extend([u_idx, v_idx])
                    cols.extend([v_idx, u_idx])
                    data.extend([w, w])
                    
        A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        F_star = compute_signed_frc(A)
        if len(F_star) > 0:
            variances.append(np.var(F_star))
        else:
            variances.append(0)
            
    plt.figure(figsize=(6,4))
    plt.plot(noise_levels, variances, marker='o', color='gray', linestyle='--')
    plt.title("F* Variance vs. SBM Structural Noise")
    plt.xlabel("Noise Level (Sign Flip Probability)")
    plt.ylabel("Variance of F*")
    plt.grid(True)
    plt.savefig('sbm_variance.pdf', bbox_inches='tight')
    print("Saved SBM sweep plot to sbm_variance.pdf")
    
def benchmark_link_prediction():
    print("Running Link Prediction on MovieLens 1M...")
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = "ml-1m.zip"
    
    if not os.path.exists(zip_path):
        print(f"Downloading ML-1M dataset from {url}...")
        urllib.request.urlretrieve(url, zip_path)
    
    rows, cols, data = [], [], []
    MAX_USERS = 6040 + 10 
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open('ml-1m/ratings.dat') as f:
            for line in f:
                parts = line.decode('utf-8').strip().split('::')
                if len(parts) >= 3:
                    u, v, rating = int(parts[0]), int(parts[1]) + MAX_USERS, int(parts[2])
                    w = 1 if rating >= 4 else -1
                    rows.extend([u, v])
                    cols.extend([v, u])
                    data.extend([w, w])
                    
    n = max(max(rows), max(cols)) + 1
    
    # Mask a small number of positive edges for testing to save time
    pos_edges = [(rows[i], cols[i]) for i in range(len(data)) if data[i] == 1 and rows[i] < MAX_USERS]
    np.random.shuffle(pos_edges)
    test_size = min(1000, len(pos_edges) // 10)
    test_pos = pos_edges[:test_size]
    
    # Generate negative samples
    test_neg = []
    while len(test_neg) < test_size:
        u = np.random.randint(1, 6040)
        v = np.random.randint(MAX_USERS, n)
        test_neg.append((u, v))
        
    print(f"Test Positive: {len(test_pos)}, Test Negative: {len(test_neg)}")
    
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    A_sq = A.dot(A)
    d = np.array(np.abs(A).sum(axis=1)).flatten()
    
    def score_edges(edges):
        scores = []
        for u, v in edges:
            A3_uv = A[u, :].dot(A_sq[:, v])[0, 0]
            du, dv = d[u], d[v]
            if du == 0 or dv == 0:
                scores.append(0)
                continue
            # Assume edge is +1 for prediction scoring
            C4_sum = 1 * A3_uv - du - dv + 1
            max_bound = max(1, (du - 1) * (dv - 1))
            normalized = C4_sum / max_bound
            F_star = 4 - du - dv + 3.0 * min(du - 1, dv - 1) * normalized
            scores.append(F_star)
        return np.array(scores)
        
    pos_scores = score_edges(test_pos)
    neg_scores = score_edges(test_neg)
    
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_scores = np.concatenate([pos_scores, neg_scores])
    
    auc = roc_auc_score(y_true, y_scores)
    print(f"Link Prediction AUC using F*: {auc:.4f}")
    
    with open("link_prediction_results.txt", "w") as f:
        f.write(f"Link Prediction AUC: {auc:.4f}\n")

if __name__ == "__main__":
    generate_sbm_sweep()
    benchmark_link_prediction()
