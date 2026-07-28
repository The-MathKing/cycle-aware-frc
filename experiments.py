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
    print("Running Stochastic Block Model (SBM) Noise Sweep (Averaged over 30 runs)...")
    n_u, n_v = 100, 100
    n = n_u + n_v
    
    noise_levels = np.linspace(0, 0.5, 10)
    avg_variances = []
    
    for noise in noise_levels:
        variances = []
        for _ in range(30):
            data, rows, cols = [], [], []
            
            # Block 1
            for i in range(50):
                for j in range(50):
                    if np.random.rand() < 0.2:
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
        avg_variances.append(np.mean(variances))
            
    plt.figure(figsize=(6,4))
    plt.plot(noise_levels, avg_variances, marker='o', color='gray', linestyle='-')
    plt.title("F* Variance vs. SBM Structural Noise (Avg 30 Runs)")
    plt.xlabel("Noise Level (Sign Flip Probability)")
    plt.ylabel("Average Variance of F*")
    plt.grid(True)
    plt.savefig('sbm_variance.pdf', bbox_inches='tight')
    print("Saved smoothed SBM sweep plot to sbm_variance.pdf")
    
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
    
    # Identify positive edges (only users pointing to items, and rating == 1)
    pos_edges_idx = [i for i in range(len(data)) if data[i] == 1 and rows[i] < MAX_USERS]
    np.random.shuffle(pos_edges_idx)
    test_size = min(1000, len(pos_edges_idx) // 10)
    test_idx = pos_edges_idx[:test_size]
    
    test_pos = [(rows[i], cols[i]) for i in test_idx]
    
    # Remove test edges from training data to prevent leakage
    train_rows = np.delete(rows, test_idx)
    train_cols = np.delete(cols, test_idx)
    train_data = np.delete(data, test_idx)
    
    # Also remove the symmetric counterparts to fully prevent leakage
    sym_test_idx = []
    for i in range(len(train_rows)):
        if (train_cols[i], train_rows[i]) in test_pos:
            sym_test_idx.append(i)
    
    train_rows = np.delete(train_rows, sym_test_idx)
    train_cols = np.delete(train_cols, sym_test_idx)
    train_data = np.delete(train_data, sym_test_idx)
    
    # Generate negative samples
    test_neg = []
    while len(test_neg) < test_size:
        u = np.random.randint(1, 6040)
        v = np.random.randint(MAX_USERS, n)
        test_neg.append((u, v))
        
    print(f"Test Positive: {len(test_pos)}, Test Negative: {len(test_neg)}")
    
    from scipy.sparse.linalg import svds
    
    A = sp.csr_matrix((train_data, (train_rows, train_cols)), shape=(n, n))
    
    print("Computing Truncated SVD Baseline...")
    A_float = A.asfptype()
    u, s, vt = svds(A_float, k=64)
    user_embeddings = u * np.sqrt(s)
    item_embeddings = vt.T * np.sqrt(s)
    
    A_sq = A.dot(A)
    d = np.array(np.abs(A).sum(axis=1)).flatten()
    
    def score_edges_fstar(edges):
        scores = []
        for u_idx, v_idx in edges:
            A3_uv = A[u_idx, :].dot(A_sq[:, v_idx])[0, 0]
            du, dv = d[u_idx], d[v_idx]
            if du == 0 or dv == 0:
                scores.append(0)
                continue
            C4_sum = 1 * A3_uv - du - dv + 1
            max_bound = max(1, (du - 1) * (dv - 1))
            normalized = C4_sum / max_bound
            F_star = 4 - du - dv + 3.0 * min(du - 1, dv - 1) * normalized
            scores.append(F_star)
        return np.array(scores)
        
    def score_edges_svd(edges):
        scores = []
        for u_idx, v_idx in edges:
            scores.append(np.dot(user_embeddings[u_idx], item_embeddings[v_idx]))
        return np.array(scores)
        
    pos_scores_f = score_edges_fstar(test_pos)
    neg_scores_f = score_edges_fstar(test_neg)
    
    pos_scores_s = score_edges_svd(test_pos)
    neg_scores_s = score_edges_svd(test_neg)
    
    y_true = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])
    
    y_scores_f = np.concatenate([pos_scores_f, neg_scores_f])
    y_scores_s = np.concatenate([pos_scores_s, neg_scores_s])
    
    # F* is inversely correlated with edge presence due to degree penalty
    auc_f = roc_auc_score(y_true, -y_scores_f)
    auc_s = roc_auc_score(y_true, y_scores_s)
    
    print(f"Link Prediction AUC using F*: {auc_f:.4f}")
    print(f"Link Prediction AUC using TruncatedSVD (k=64): {auc_s:.4f}")
    
    with open("link_prediction_results.txt", "w") as f:
        f.write(f"Link Prediction AUC (F*): {auc_f:.4f}\n")
        f.write(f"Link Prediction AUC (SVD): {auc_s:.4f}\n")

if __name__ == "__main__":
    generate_sbm_sweep()
    benchmark_link_prediction()
