import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
import urllib.request
import gzip
import zipfile
import time
import os
import tracemalloc

def compute_signed_frc(A: sp.csr_matrix, gamma=3.0) -> np.ndarray:
    """
    Computes normalized Cycle-Aware Forman-Ricci Curvature (F*) for a signed bipartite network.

    Extracts 4-cycle participation algebraically via the identity:
        Sigma_C4(u,v) = A_uv * (A^3)_uv - d(u) - d(v) + 1
    where (A^3)_uv is computed without ever materializing A^2 in full:
    for each row u of A we compute one row of A^2 on the fly, extract only
    the values needed at neighbor positions, then discard it.  This keeps
    the working-set size O(|E| + |V|) rather than O(nnz(A^2)).
    """
    A = A.tocsr()
    A.eliminate_zeros()  # Prevents nnz mismatches when duplicates sum to zero

    # Extract absolute degrees
    abs_A = abs(A)
    degrees = np.array(abs_A.sum(axis=1)).flatten()

    # Extract structural edges
    u_idx, v_idx = A.nonzero()
    edge_weights = np.array(A[u_idx, v_idx]).flatten()

    n_edges = len(u_idx)
    A3_uv = np.zeros(n_edges, dtype=np.float64)

    # -----------------------------------------------------------------------
    # Row-by-row A^3 extraction — O(|E| + |V|) peak space.
    #
    # Key identity: (A^3)_{u,v} = (A[u,:] @ A^2)_{v}
    #                           = sum_w  A[u,w] * (A^2)[w,v]
    #                           = A[u,:] @ (A @ A^T)[v,:]   (using symmetry A^T=A)
    #
    # For row u with neighbors {w_1,...,w_k}:
    #   row_A2 = A[u,:] @ A   (one sparse row of A^2, discarded after use)
    # Then for each neighbor v of u, (A^3)_{u,v} = row_A2[0, v].
    # -----------------------------------------------------------------------
    # Build a map from row -> list of (edge_index, column) so we loop over
    # distinct rows of A only once.
    from collections import defaultdict
    row_to_edges = defaultdict(list)
    for eidx, (r, c) in enumerate(zip(u_idx, v_idx)):
        row_to_edges[r].append((eidx, c))

    for row, edge_list in row_to_edges.items():
        # Compute one row of A^2 on the fly; shape (1, n).
        # scipy returns a sparse matrix for sparse row slicing.
        row_A = A.getrow(row)           # 1 x n, sparse
        row_A2 = row_A.dot(A)           # 1 x n, sparse or dense depending on fill

        # Extract the entries at the needed column positions.
        cols_needed = [c for (_, c) in edge_list]
        if sp.issparse(row_A2):
            vals = np.asarray(row_A2[:, cols_needed].todense()).flatten()
        else:
            vals = np.asarray(row_A2).flatten()[cols_needed]

        for (eidx, _), val in zip(edge_list, vals):
            A3_uv[eidx] = val

    deg_u = degrees[u_idx]
    deg_v = degrees[v_idx]

    # Raw 4-cycle sum: Sigma_C4(u,v) = A_{uv} * (A^3)_{uv} - d(u) - d(v) + 1
    raw_sigma_c4 = (edge_weights * A3_uv) - deg_u - deg_v + 1

    # Normalization bound: max possible 4-cycles sharing edge e=(u,v) is (d(u)-1)*(d(v)-1)
    max_c4 = np.maximum(1, (deg_u - 1) * (deg_v - 1))

    # Normalized cycle contribution
    norm_sigma_c4 = raw_sigma_c4 / max_c4

    # Scale by min(d(u)-1, d(v)-1) to mirror the O(d) upper bound of standard FRC
    # where each triangle contributes gamma=3 and max triangles is min(d(u)-1, d(v)-1).
    scale_bound = np.maximum(1, np.minimum(deg_u - 1, deg_v - 1))

    # Cycle-Aware FRC: F*(e) = 4 - d(u) - d(v) + gamma * scale_bound * norm_sigma_c4
    F_star = 4 - deg_u - deg_v + (gamma * scale_bound * norm_sigma_c4)

    return F_star, u_idx, v_idx, edge_weights


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


def benchmark_movielens():
    """
    Downloads and evaluates the MovieLens-1M dataset.
    A genuine bipartite graph. Binarizes ratings: >=4 is +1, <=3 is -1.
    """
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = "ml-1m.zip"

    if not os.path.exists(zip_path):
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, zip_path)

    print("Parsing MovieLens 1M dataset...")
    rows, cols, data = [], [], []

    # Offset movies by max_user_id to create a bipartite adjacency matrix
    MAX_USERS = 6040 + 10  # ML-1M has 6040 users

    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open('ml-1m/ratings.dat') as f:
            for line in f:
                # Format: UserID::MovieID::Rating::Timestamp
                parts = line.decode('utf-8').strip().split('::')
                if len(parts) >= 3:
                    u = int(parts[0])
                    v = int(parts[1]) + MAX_USERS  # Shift movie index to make it bipartite
                    rating = int(parts[2])

                    # Binarize rating
                    w = 1 if rating >= 4 else -1

                    rows.extend([u, v])
                    cols.extend([v, u])
                    data.extend([w, w])

    n = max(max(rows), max(cols)) + 1
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    print(f"Evaluating F* on network with {n} nodes and {len(data)//2} undirected bipartite edges...")

    tracemalloc.start()
    start_time = time.time()

    F_star, _, _, _ = compute_signed_frc(A)

    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / 10**6
    print(f"Benchmark completed in {elapsed:.3f} seconds.")
    print(f"Peak Memory Usage: {peak_mb:.2f} MB")
    print(f"F* distribution stats: Min={np.min(F_star):.2f}, Max={np.max(F_star):.2f}, Mean={np.mean(F_star):.2f}, Var={np.var(F_star):.2f}")

    with open("benchmark_results.txt", "w") as f:
        f.write(f"Time: {elapsed:.3f}s\nPeak RAM: {peak_mb:.2f} MB\n")
        f.write(f"Min: {np.min(F_star):.2f}\nMax: {np.max(F_star):.2f}\nMean: {np.mean(F_star):.2f}\nVar: {np.var(F_star):.2f}\n")


if __name__ == "__main__":
    generate_synthetic_test()
    benchmark_movielens()
