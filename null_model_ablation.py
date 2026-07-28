import os
import zipfile
import urllib.request
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

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

def generate_bipartite_configuration_model(rows, cols):
    """
    Generates a strict Bipartite Configuration Model by performing edge swaps.
    This exactly preserves the degree sequence of all nodes while destroying 
    the mesoscopic 4-cycle structure.
    """
    print("Generating Bipartite Configuration Model via Edge Swapping...")
    E = len(rows)
    
    # Perform 1 * E edge swaps to randomize topology rapidly
    num_swaps = 1 * E
    edges = set(zip(rows, cols))
    
    successful_swaps = 0
    for _ in range(num_swaps):
        idx1 = np.random.randint(0, E)
        idx2 = np.random.randint(0, E)
        if idx1 == idx2:
            continue
            
        u1, v1 = rows[idx1], cols[idx1]
        u2, v2 = rows[idx2], cols[idx2]
        
        # Proposed new edges: Ensure they don't already exist to preserve graph simplicity
        if (u1, v2) not in edges and (u2, v1) not in edges:
            edges.remove((u1, v1))
            edges.remove((u2, v2))
            
            rows[idx1], cols[idx1] = u1, v2
            rows[idx2], cols[idx2] = u2, v1
            
            edges.add((u1, v2))
            edges.add((u2, v1))
            successful_swaps += 1
            
    print(f"Completed {successful_swaps} successful structural swaps.")
    return rows, cols, [1]*E

def compute_fstar_scores(train_rows, train_cols, train_data, test_pos, test_neg, n):
    A_train = sp.csr_matrix((train_data, (train_rows, train_cols)), shape=(n, n))
    A_sq = A_train.dot(A_train)
    d = np.array(np.abs(A_train).sum(axis=1)).flatten()
    
    def score_fstar(u, v):
        # A_sq is symmetric, so A_sq[:, v] == A_sq[v, :].T
        # Row extraction on CSR is O(1), column extraction is O(N)
        A3_uv = A_train[u, :].dot(A_sq[v, :].T)[0, 0]
        du, dv = d[u], d[v]
        if du == 0 or dv == 0:
            return 0
        C4_sum = 1 * A3_uv - du - dv + 1
        max_bound = max(1, (du - 1) * (dv - 1))
        normalized = C4_sum / max_bound
        return 4 - du - dv + 3.0 * min(du - 1, dv - 1) * normalized
        
    pos_scores_fs = [-score_fstar(u,v) for u,v in test_pos] # Inverse correlation
    neg_scores_fs = [-score_fstar(u,v) for u,v in test_neg]
    return pos_scores_fs, neg_scores_fs

def run_ablation():
    rows, cols, data, num_users = load_ml1m()
    n = max(max(rows), max(cols)) + 1
    
    print("\n--- Running Ablation on True Graph ---")
    edge_set = set(zip(rows, cols))
    pos_edges = list(edge_set)
    np.random.shuffle(pos_edges)
    test_size = len(pos_edges) // 10
    
    test_pos = pos_edges[:test_size]
    train_pos = pos_edges[test_size:]
    
    train_rows, train_cols, train_data = [], [], []
    for u, v in train_pos:
        train_rows.extend([u, v])
        train_cols.extend([v, u])
        train_data.extend([1, 1])
        
    test_neg = []
    while len(test_neg) < test_size:
        u = np.random.randint(0, num_users)
        v = np.random.randint(num_users, n)
        if (u, v) not in edge_set:
            test_neg.append((u, v))
            
    pos_fs, neg_fs = compute_fstar_scores(train_rows, train_cols, train_data, test_pos, test_neg, n)
    y_true = [1]*test_size + [0]*test_size
    auc_true = roc_auc_score(y_true, pos_fs + neg_fs)
    print(f"True Graph F* AUC: {auc_true:.4f}")
    
    print("\n--- Running Ablation on Degree-Preserving Null Graph ---")
    null_rows, null_cols, null_data = generate_bipartite_configuration_model(
        [u for u,v in train_pos], [v for u,v in train_pos]
    )
    
    sym_null_rows, sym_null_cols, sym_null_data = [], [], []
    for u, v in zip(null_rows, null_cols):
        sym_null_rows.extend([u, v])
        sym_null_cols.extend([v, u])
        sym_null_data.extend([1, 1])
        
    null_pos_fs, null_neg_fs = compute_fstar_scores(sym_null_rows, sym_null_cols, sym_null_data, test_pos, test_neg, n)
    auc_null = roc_auc_score(y_true, null_pos_fs + null_neg_fs)
    print(f"Configuration Null Graph F* AUC: {auc_null:.4f}")
    
if __name__ == "__main__":
    run_ablation()
