# Cycle-Aware Forman-Ricci Curvature (FRC)

This repository contains the code to reproduce the experiments and benchmarking results for the Cycle-Aware Forman-Ricci Curvature framework applied to signed and bipartite networks.

## Requirements

Ensure you have the required Python libraries installed:
```bash
pip install numpy scipy scikit-learn matplotlib pandas torch torch-geometric
```

## Running the Tests

To reproduce the experiments, you can execute the provided Python scripts. Data will be automatically downloaded when running the experiments.

### 1. Stochastic Block Model (SBM) Noise Sweep & Basic Benchmark
To run the SBM variance degradation sweep and a basic MovieLens-1M benchmark:
```bash
python experiments.py
```
This script evaluates the robustness of the \( F^* \) metric to topological sign noise and runs a simple baseline Link Prediction experiment on the MovieLens-1M dataset.

### 2. Comprehensive Baselines (LightGCN & SVD)
To run the full pipeline on both Amazon Video Games and MovieLens-1M, including training the state-of-the-art LightGCN architecture and Truncated SVD baseline, run:
```bash
python run_baselines.py
```
This will automatically:
1. Download and parse the `amazon_vg.json.gz` and `ml-1m.zip` datasets.
2. Formulate 10% link prediction splits.
3. Output the final AUC-ROC metrics for all three models into `final_results.txt`.

### 3. Additional Visualizations and Ablations
- **Null Model Ablation:** Run `python null_model_ablation.py` to evaluate the effect of the underlying degree distribution vs. genuine 4-cycle formations.
- **Topological Plots:** Run `python plot_topologies.py` and `python plot_advanced.py` to recreate the spy-plots and core/fringe topology distributions.
