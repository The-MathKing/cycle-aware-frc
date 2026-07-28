import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
import urllib.request
import gzip
import time
import os

def compute_signed_frc(A: sp.csr_matrix, gamma=3.0) -> np.ndarray:
    """
    Computes normalized Cycle-Aware Forman-Ricci Curvature (F*) for a signed bipartite network.
    Extracts 4-cycle participation algebraically and enforces an O(d) normalization bound.
    """
    A = A.tocsr()
    A.eliminate_zeros() # Prevents nnz mismatches when duplicates sum to zero
    
    # Extract absolute degrees
    abs_A = abs(A)
    degrees = np.array(abs_A.sum(axis=1)).flatten()
    
    # Extract structural edges
    u, v = A.nonzero()
    edge_weights = A.data
    
    # Memory-efficient A3 extraction at non-zero entries (SDDMM-like approach)
    # Avoids building the full A^3 matrix which causes OOM on dense graphs.
    A3_uv = np.empty(len(u), dtype=np.int32)
    A_csr = A
    
    print("Computing A^2...")
    A2 = A.dot(A)
    print("Extracting A^3 natively...")
    A2_csr = A2.tocsr()
    
    for i in range(A_csr.shape[0]):
        row_start = A_csr.indptr[i]
        row_end = A_csr.indptr[i+1]
        if row_start == row_end:
            continue
        
        neighbors = A_csr.indices[row_start:row_end]
        weights = A_csr.data[row_start:row_end]
        
        # Calculate the i-th row of A3: A[i, :] @ A2
        # Since A[i, :] only has non-zeros at `neighbors` with `weights`,
        # it is exactly `weights @ A2[neighbors, :]`
        row_A3 = weights @ A2_csr[neighbors, :]
        
        # Extract the values at `neighbors` (the non-zero structure of A)
        if sp.issparse(row_A3):
            A3_uv[row_start:row_end] = row_A3.toarray()[0, neighbors]
        else:
            # If row_A3 is a numpy matrix or array
            A3_uv[row_start:row_end] = np.asarray(row_A3).flatten()[neighbors]
    
    deg_u = degrees[u]
    deg_v = degrees[v]
    
    # Raw 4-cycle sum: Sigma_C4(u,v) = A_{uv} * (A^3)_{uv} - d(u) - d(v) + 1
    raw_sigma_c4 = (edge_weights * A3_uv) - deg_u - deg_v + 1
    
    # Normalization bound: prevent O(d^2) explosion by dividing by max possible 4-cycles
    # Max possible 4-cycles sharing edge e=(u,v) in bipartite graph is (d(u)-1)*(d(v)-1)
    max_c4 = np.maximum(1, (deg_u - 1) * (deg_v - 1))
    
    # Normalized cycle contribution
    norm_sigma_c4 = raw_sigma_c4 / max_c4
    
    # To mirror standard FRC O(d) scaling for triangle counts, we scale the normalized
    # fraction by the geometric mean or min degree bound.
    # In standard FRC, +3 * |T|. Max |T| is min(d(u)-1, d(v)-1).
    # We will scale by min(d(u)-1, d(v)-1) to bound the metric to O(d) correctly.
    scale_bound = np.maximum(1, np.minimum(deg_u - 1, deg_v - 1))
    
    # Cycle-Aware FRC formulation
    # F*(e) = 4 - d(u) - d(v) + gamma * scale_bound * norm_sigma_c4
    F_star = 4 - deg_u - deg_v + (gamma * scale_bound * norm_sigma_c4)
    
    return F_star, u, v, edge_weights

def generate_synthetic_test():
    """
    Generates a non-homogeneous signed bipartite block model to prove F* differentiates subsets.
    """
    # Nodes U: 0,1,2,3,4 | Nodes V: 5,6,7,8,9
    # Dense cohesive cluster (X): 0,1,2 <-> 5,6,7 (fully connected, positive)
    # Sparse bridge cluster (Y): 3,4 <-> 8,9 (sparse, negative) + bridge 2<->8
    edges = [
        # Dense Cluster
        (0,5,1), (0,6,1), (0,7,1),
        (1,5,1), (1,6,1), (1,7,1),
        (2,5,1), (2,6,1), (2,7,1),
        # Bridge
        (2,8,-1),
        # Sparse Cluster
        (3,8,-1), (4,9,-1), (3,9,1)
    ]
    
    row = np.array([e[0] for e in edges] + [e[1] for e in edges])
    col = np.array([e[1] for e in edges] + [e[0] for e in edges])
    data = np.array([e[2] for e in edges] + [e[2] for e in edges], dtype=np.int32)
    
    A = sp.csr_matrix((data, (row, col)), shape=(10, 10))
    F_star, u, v, w = compute_signed_frc(A)
    
    print("Non-Homogeneous Graph Execution Results:")
    for i in range(len(edges)):
        idx = np.where((u == edges[i][0]) & (v == edges[i][1]))[0][0]
        print(f"Edge ({edges[i][0]}-{edges[i][1]}) | F*: {F_star[idx]:.2f}")

    # Plot
    G = nx.Graph()
    for i in range(len(edges)):
        r, c, weight = edges[i]
        idx = np.where((u == r) & (v == c))[0][0]
        G.add_edge(r, c, weight=weight, f_star=round(F_star[idx], 2))
        
    pos = nx.bipartite_layout(G, [0, 1, 2, 3, 4])
    
    plt.figure(figsize=(10, 6))
    pos_edges = [e for e in G.edges(data=True) if e[2]['weight'] > 0]
    neg_edges = [e for e in G.edges(data=True) if e[2]['weight'] < 0]
    
    nx.draw_networkx_nodes(G, pos, nodelist=[0,1,2,5,6,7], node_color='lightgray', node_shape='s', node_size=800, label='Dense X')
    nx.draw_networkx_nodes(G, pos, nodelist=[3,4,8,9], node_color='darkgray', node_shape='o', node_size=800, label='Sparse Y')
    nx.draw_networkx_labels(G, pos, font_size=16)
    
    nx.draw_networkx_edges(G, pos, edgelist=pos_edges, width=2, edge_color='black', style='solid', label='Positive Edge')
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges, width=2, edge_color='black', style='dashed', label='Negative Edge')
    
    edge_labels = {(u_, v_): f"{d['f_star']}" for u_, v_, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, label_pos=0.3)
    
    plt.title('Normalized Cycle-Aware Forman-Ricci Curvature (F*)', fontsize=14)
    plt.legend(scatterpoints=1, loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('graph_viz.pdf', bbox_inches='tight')
    print("Figure saved to graph_viz.pdf")

def benchmark_epinions():
    """
    Downloads and evaluates the SNAP soc-sign-epinions dataset.
    131k nodes, 841k signed edges. Treats as undirected for structural geometry check.
    """
    url = "https://snap.stanford.edu/data/soc-sign-epinions.txt.gz"
    filepath = "soc-sign-epinions.txt.gz"
    
    if not os.path.exists(filepath):
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, filepath)
    
    print("Parsing dataset...")
    rows, cols, data = [], [], []
    with gzip.open(filepath, 'rt') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            if len(parts) >= 3:
                u, v, w = int(parts[0]), int(parts[1]), int(parts[2])
                # Make symmetric for undirected structural metric
                rows.extend([u, v])
                cols.extend([v, u])
                data.extend([w, w])
                
    n = max(max(rows), max(cols)) + 1
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    
    print(f"Evaluating F* on network with {n} nodes and {len(data)//2} undirected edges...")
    start_time = time.time()
    F_star, _, _, _ = compute_signed_frc(A)
    elapsed = time.time() - start_time
    
    print(f"Benchmark completed in {elapsed:.3f} seconds.")
    print(f"F* distribution stats: Min={np.min(F_star):.2f}, Max={np.max(F_star):.2f}, Mean={np.mean(F_star):.2f}")
    
if __name__ == "__main__":
    generate_synthetic_test()
    benchmark_epinions()
