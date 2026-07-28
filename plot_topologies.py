import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
from compute_frc import compute_signed_frc

import zipfile

def plot_topologies():
    print("Loading MovieLens-1M for Topological Case Studies...")
    data, rows, cols = [], [], []
    MAX_USERS = 6040 + 10
    
    with zipfile.ZipFile('ml-1m.zip', 'r') as z:
        with z.open('ml-1m/ratings.dat') as f:
            for line in f:
                parts = line.decode('utf-8').strip().split('::')
                if len(parts) >= 3:
                    u = int(parts[0]) - 1
                    i = int(parts[1]) - 1 + MAX_USERS
                    r = float(parts[2])
                    if r >= 4.0:
                        rows.extend([u, i])
                        cols.extend([i, u])
                        data.extend([1, 1])
                
    n = max(max(rows), max(cols)) + 1
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    
    print("Computing F* to extract core and fringe edges...")
    A_sq = A.dot(A)
    d = np.array(np.abs(A).sum(axis=1)).flatten()
    
    # We just need to score a subset of edges to find extremes
    # Let's just pick the first 10,000 edges to speed up
    edges_to_score = list(zip(rows[:10000:2], cols[:10000:2])) 
    
    scores = []
    for u, v in edges_to_score:
        A3_uv = A[u, :].dot(A_sq[:, v])[0, 0]
        du, dv = d[u], d[v]
        if du == 0 or dv == 0:
            scores.append(0)
            continue
        C4_sum = 1 * A3_uv - du - dv + 1
        max_bound = max(1, (du - 1) * (dv - 1))
        normalized = C4_sum / max_bound
        F_star = 4 - du - dv + 3.0 * min(du - 1, dv - 1) * normalized
        scores.append(F_star)
        
    scores = np.array(scores)
    sorted_indices = np.argsort(scores)
    
    print("Extracting and plotting Core (Highest F*) and Fringe (Lowest F*) Topologies...")
    
    # Highest F* (Note: F* is deeply negative for high degrees. High curvature actually means CLOSE TO ZERO or POSITIVE, 
    # but wait: in our metric F* is inversely correlated with dense edge formation because of the massive negative degree penalty.
    # Therefore, the "Bipartite Core" (densest) actually has the LOWEST F* (most negative). 
    # The "Fringes" (tree-like, low degree) have the HIGHEST F* (closest to +2).
    # This is a critical mathematical reality of our normalization.
    
    fringe_indices = sorted_indices[-100:] # Highest F* (closest to 4) - low degrees
    core_indices = sorted_indices[:100] # Lowest F* (most negative) - high degrees
    
    G_core = nx.Graph()
    for idx in core_indices:
        u, v = edges_to_score[idx]
        G_core.add_edge(u, v)
        
    G_fringe = nx.Graph()
    for idx in fringe_indices:
        u, v = edges_to_score[idx]
        G_fringe.add_edge(u, v)
        
    plt.figure(figsize=(8,8))
    pos_core = nx.spring_layout(G_core, seed=42)
    nx.draw(G_core, pos_core, node_size=20, edge_color='darkred', node_color='black', alpha=0.7)
    plt.title("Bipartite Core (Lowest F*)")
    plt.savefig('core_topology.pdf', bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8,8))
    pos_fringe = nx.spring_layout(G_fringe, seed=42)
    nx.draw(G_fringe, pos_fringe, node_size=20, edge_color='blue', node_color='gray', alpha=0.7)
    plt.title("Bipartite Fringe (Highest F*)")
    plt.savefig('fringe_topology.pdf', bbox_inches='tight')
    plt.close()
    
    print("Topologies plotted and saved.")

if __name__ == "__main__":
    plot_topologies()
