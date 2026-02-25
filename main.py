"""
Clustering Algorithms: From-Scratch Implementation
====================================================
Implements K-Means, DBSCAN, and HDBSCAN from scratch using only NumPy/Pandas.
Applies all three to clustering_data.csv and compares results.

Usage:
    uv run python main.py
"""
import os
import matplotlib
matplotlib.use("Agg")

from pipeline import (
    load_and_explore_data,
    run_kmeans,
    run_dbscan,
    run_hdbscan,
    evaluate_and_compare,
    print_analysis
)

os.makedirs("figures", exist_ok=True)


def main():
    """
    Main orchestrator for the clustering pipeline.
    
    This function:
    1. Loads the clustering dataset and explores its properties.
    2. Runs K-Means to find an optimal number of clusters (k=5).
    3. Runs DBSCAN with parameter sweep to detect clusters and noise.
    4. Runs HDBSCAN for hierarchical density-based clustering.
    5. Computes evaluation metrics (Silhouette, Calinski-Harabasz, Inertia).
    6. Outputs a comparison summary, plots, and a final text analysis.
    """
    print("=" * 70)
    print("  CLUSTERING ALGORITHMS: FROM-SCRATCH IMPLEMENTATION")
    print("=" * 70)

    # 1. Load data
    data, _ = load_and_explore_data()

    # 2. Run algorithms
    km_res = run_kmeans(data)
    db_res = run_dbscan(data)
    hdb_res = run_hdbscan(data)

    # 3. Evaluate and Compare
    metrics = evaluate_and_compare(data, km_res, db_res, hdb_res)
    print_analysis(km_res, db_res, hdb_res, metrics)


if __name__ == "__main__":
    main()
