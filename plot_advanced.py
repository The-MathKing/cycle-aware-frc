import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp

# Set strict academic formatting
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
    'font.family': 'serif'
})

def plot_kde_ablation(real_fstar, null_fstar, save_path="kde_ablation.png"):
    """
    1. Dual-Distribution KDE Plot (Degree Bias Ablation)
    Visually proves that the null model collapses to a narrow distribution, 
    while the real network exhibits a wide geometric spread.
    """
    plt.figure(figsize=(8, 6))
    
    # Use color-blind friendly palette (coolwarm for divergence)
    sns.kdeplot(real_fstar, fill=True, color='#d73027', alpha=0.5, label='Genuine Topology (MovieLens)')
    sns.kdeplot(null_fstar, fill=True, color='#4575b4', alpha=0.5, label='Null Model (Degree-Preserved)')
    
    plt.xlabel(r'Cycle-Aware Curvature ($F^*$)')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved KDE Ablation plot to {save_path}")

def plot_loglog_complexity(save_path="complexity_loglog.png"):
    """
    2. Computational Complexity (Log-Log Chart)
    Plots Execution Time vs Edge Count for SDDMM and LightGCN.
    Uses simulated benchmarking data for structural demonstration.
    """
    # Simulated Edge Counts: 10k to 10M
    edges = np.logspace(4, 7, num=10)
    
    # SDDMM is O(|E| * d_max) -> roughly linear with a slight density penalty
    # Simulated taking ~5 seconds for 1M edges
    time_sddmm = (edges / 1e6) * 5.0
    
    # LightGCN is O(|E| * epochs * embedding_dim) + GPU tensor overhead
    # Simulated taking ~300 seconds for 1M edges (50 epochs)
    time_gcn = (edges / 1e6) * 300.0
    
    plt.figure(figsize=(8, 6))
    plt.loglog(edges, time_gcn, marker='s', linestyle='--', color='#d73027', linewidth=2, markersize=8, label='LightGCN (50 Epochs)')
    plt.loglog(edges, time_sddmm, marker='o', linestyle='-', color='#4575b4', linewidth=2, markersize=8, label='Cycle-Aware $F^*$ (SpSpMM)')
    
    plt.xlabel(r'Edge Count ($|E|$)')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Complexity plot to {save_path}")

def plot_bipartite_spy(adj_matrix, fstar_weights, save_path="bipartite_spy.png"):
    """
    3. Topological Adjacency Spy Plot
    Visualizes a sparse bipartite adjacency matrix colored by F* curvature weights.
    """
    # Convert to COO for easy plotting
    coo = sp.coo_matrix(adj_matrix)
    rows, cols, data = coo.row, coo.col, fstar_weights
    
    plt.figure(figsize=(10, 8))
    
    # Scatter plot with coolwarm colormap for F* values
    # High F* (Core) -> Warm colors (Red)
    # Low F* (Fringe) -> Cool colors (Blue)
    scatter = plt.scatter(cols, rows, c=data, cmap='coolwarm', s=15, alpha=0.8, edgecolors='none')
    
    plt.colorbar(scatter, label=r'Cycle-Aware Curvature ($F^*$)')
    plt.xlabel('Item Nodes (Partition $V$)')
    plt.ylabel('User Nodes (Partition $U$)')
    plt.gca().invert_yaxis() # Matrix convention (0,0 at top-left)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Spy plot to {save_path}")

if __name__ == "__main__":
    # Generate dummy data to verify output formatting locally
    real_f = np.random.normal(loc=0, scale=1.5, size=1000)
    null_f = np.random.normal(loc=-2, scale=0.3, size=1000)
    plot_kde_ablation(real_f, null_f)
    
    plot_loglog_complexity()
    
    # Generate dummy bipartite matrix block structure
    block1 = np.ones((100, 100))
    block2 = np.ones((50, 50))
    adj = sp.block_diag((block1, block2)).tocoo()
    weights = np.concatenate([np.random.normal(2, 0.5, 100*100), np.random.normal(-2, 0.5, 50*50)])
    plot_bipartite_spy(adj, weights)
