import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt

def compute_signed_frc(A: sp.csr_matrix) -> np.ndarray:
    """
    Computes Cycle-Aware Forman-Ricci Curvature (F*) for a signed bipartite network.
    Extracts 4-cycle participation algebraically utilizing exact integer 
    sparse matrix multiplication to avoid floating point degradation.
    """
    A = A.tocsr()
    
    # Extract absolute degrees (unweighted node degrees)
    abs_A = abs(A)
    degrees = np.array(abs_A.sum(axis=1)).flatten()
    
    # Sparse matrix multiplication for paths of length 3 (A^3)
    A2 = A.dot(A)
    A3 = A2.dot(A)
    
    # Extract structural edges natively avoiding nested loops
    u, v = A.nonzero()
    edge_weights = A.data
    
    # Extract (A^3)_{uv} entries directly corresponding to edges
    A3_uv = np.asarray(A3[u, v]).flatten()
    
    deg_u = degrees[u]
    deg_v = degrees[v]
    
    # Net structurally balanced 4-cycles contribution
    sigma_c4 = (edge_weights * A3_uv) - deg_u - deg_v + 1
    
    # F*(e) calculation
    F_star = 4 - deg_u - deg_v + sigma_c4
    
    return F_star, u, v, edge_weights

def generate_synthetic_test():
    """
    Generates a strictly structurally balanced signed bipartite graph and 
    computes F* to verify algorithmic execution and absence of multiplicity errors.
    """
    # Structurally balanced bipartite signed network (6 nodes)
    # Partitions: U = {0, 1, 2}, V = {3, 4, 5}
    # Balanced grouping (positive intra-group, negative inter-group): 
    # Group X = {0, 3, 4}, Group Y = {1, 2, 5}
    row = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
    col = np.array([3, 4, 5, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    data = np.array([1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1], dtype=np.int32)
    
    A = sp.csr_matrix((data, (row, col)), shape=(6, 6))
    
    F_star, u, v, edge_weights = compute_signed_frc(A)
    
    print("Structurally Balanced Signed Bipartite Execution Results:")
    for i in range(len(data)):
        print(f"Edge ({row[i]}, {col[i]}) | Weight: {data[i]:>2} | F*: {F_star[i]}")

    # Generate Figure
    G = nx.Graph()
    upper_mask = row < col
    for i in range(len(row[upper_mask])):
        r, c, w = row[upper_mask][i], col[upper_mask][i], data[upper_mask][i]
        f_val = F_star[upper_mask][i]
        G.add_edge(r, c, weight=w, f_star=f_val)
        
    for node in [0, 1, 2]:
        G.nodes[node]['bipartite'] = 0
    for node in [3, 4, 5]:
        G.nodes[node]['bipartite'] = 1

    pos = nx.bipartite_layout(G, [0, 1, 2])
    
    # Plotting
    plt.figure(figsize=(8, 6))
    edges = G.edges(data=True)
    pos_edges = [e for e in edges if e[2]['weight'] > 0]
    neg_edges = [e for e in edges if e[2]['weight'] < 0]
    
    # Draw nodes with grayscale safe colors and distinct shapes
    nx.draw_networkx_nodes(G, pos, nodelist=[0,3,4], node_color='lightgray', node_shape='s', node_size=800, label='Group X')
    nx.draw_networkx_nodes(G, pos, nodelist=[1,2,5], node_color='darkgray', node_shape='o', node_size=800, label='Group Y')
    nx.draw_networkx_labels(G, pos, font_size=16)
    
    # Draw edges with distinct line styles
    nx.draw_networkx_edges(G, pos, edgelist=pos_edges, width=2, edge_color='black', style='solid', label='Positive Edge (+1)')
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges, width=2, edge_color='black', style='dashed', label='Negative Edge (-1)')
    
    # Edge labels (F* values)
    edge_labels = {(u, v): f"F*={d['f_star']}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, label_pos=0.3)
    
    plt.title('Cycle-Aware Forman-Ricci Curvature (F*) on Signed Bipartite Graph', fontsize=14)
    # Adjust legend position to avoid overlapping
    plt.legend(scatterpoints=1, loc='upper left', bbox_to_anchor=(1, 1))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('graph_viz.pdf', bbox_inches='tight')
    print("Figure saved to graph_viz.pdf")

if __name__ == "__main__":
    generate_synthetic_test()
