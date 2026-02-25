import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kmeans import KMeans
from dbscan import DBSCAN
from hdbscan_cluster import HDBSCAN
from metrics import silhouette_score, calinski_harabasz_score, compute_inertia
from visualization import plot_clusters, plot_clusters_detailed, plot_cluster_sizes


def load_and_explore_data():
    print("\n[1/7] Loading data...")
    df = pd.read_csv("clustering_data.csv")
    data = df[["x", "y"]].values
    print(f"  Shape: {df.shape} | X range: [{data[:, 0].min():.2f}, {data[:, 0].max():.2f}] | Y range: [{data[:, 1].min():.2f}, {data[:, 1].max():.2f}]")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].scatter(df["x"], df["y"], s=3, alpha=0.5, c="steelblue", edgecolors="none")
    axes[0].set_title("Raw Data Scatter Plot", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[1].hist(df["x"], bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    axes[1].set_title("Distribution of x", fontsize=14, fontweight="bold")
    axes[2].hist(df["y"], bins=50, color="coral", alpha=0.7, edgecolor="white")
    axes[2].set_title("Distribution of y", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/01_raw_data.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 8))
    hb = ax.hexbin(df["x"], df["y"], gridsize=40, cmap="YlOrRd", mincnt=1)
    plt.colorbar(hb, ax=ax, label="Count")
    ax.set_title("2D Density Plot (Hexbin)", fontsize=14, fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig("figures/02_density.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    return data, df


def run_kmeans(data):
    print("\n[2/7] Running K-Means elbow method...")
    k_range = range(2, 11)
    inertias = []
    for k in k_range:
        km = KMeans(k=k, n_init=5, random_state=42).fit(data)
        inertias.append(km.inertia)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(k_range), inertias, "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Clusters (k)", fontsize=13)
    ax.set_ylabel("Inertia (SSE)", fontsize=13)
    ax.set_title("K-Means Elbow Method", fontsize=15, fontweight="bold")
    ax.set_xticks(list(k_range))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/03_elbow.png", dpi=150, bbox_inches="tight")
    plt.close()

    optimal_k = 5
    print("\n[3/7] Running K-Means with optimal k...")
    kmeans = KMeans(k=optimal_k, n_init=10, random_state=42).fit(data)
    
    fig, ax = plt.subplots(figsize=(14, 9))
    plot_clusters_detailed(ax, data, kmeans.labels, optimal_k, f"K-Means Clustering (k={optimal_k})")
    ax.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], s=250, c="black", marker="X", edgecolors="white", linewidths=2, label="Centroids", zorder=6)
    ax.legend(fontsize=9, markerscale=2.5, loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig("figures/04_kmeans.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    return optimal_k, kmeans


def run_dbscan(data):
    print("\n[4/7] Computing k-distance graph for DBSCAN eps selection...")
    from_k = 5
    n = data.shape[0]
    k_dists = np.zeros(n)
    for start in range(0, n, 500):
        end = min(start + 500, n)
        dists = np.sqrt(((data[start:end, np.newaxis, :] - data[np.newaxis, :, :]) ** 2).sum(axis=2))
        k_dists[start:end] = np.sort(dists, axis=1)[:, from_k]

    k_dists_ascending = np.sort(k_dists)
    k_dists_sorted = k_dists_ascending[::-1]
    
    window = max(len(k_dists_ascending) // 100, 5)
    smoothed = np.convolve(k_dists_ascending, np.ones(window)/window, mode='valid')
    second_deriv = np.diff(smoothed, n=2)
    elbow_idx = np.argmax(second_deriv) + window // 2 + 1
    suggested_eps = k_dists_ascending[min(elbow_idx, len(k_dists_ascending) - 1)]
    suggested_eps = max(suggested_eps, np.percentile(k_dists, 75)) if suggested_eps < np.percentile(k_dists, 50) else suggested_eps

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_dists_sorted, linewidth=1.5, color="steelblue")
    ax.axhline(y=suggested_eps, color="red", linestyle="--", linewidth=1.5, label=f"Suggested eps ≈ {suggested_eps:.1f}")
    ax.set_xlabel("Points (sorted by distance)", fontsize=13)
    ax.set_ylabel(f"{from_k}-th Nearest Neighbor Distance", fontsize=13)
    ax.set_title(f"k-Distance Graph (k={from_k})", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/05_kdist.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n[5/7] Running DBSCAN with parameter sweep...")
    min_samples_value = 5
    eps_candidates = np.concatenate([np.linspace(np.percentile(k_dists, 50), np.percentile(k_dists, 99), 5), np.linspace(np.percentile(k_dists, 99), np.percentile(k_dists, 99) * 5, 10)])
    best_eps, best_score, best_dbscan = float(suggested_eps), -2, None

    for eps_try in eps_candidates:
        db_try = DBSCAN(eps=float(eps_try), min_samples=min_samples_value).fit(data)
        n_cl, n_noise = db_try.n_clusters, np.sum(db_try.labels == -1)
        if 2 <= n_cl <= 20 and n_noise <= len(data) * 0.4:
            sc = silhouette_score(data, db_try.labels)
            if sc > best_score:
                best_score, best_eps, best_dbscan = sc, eps_try, db_try

    dbscan = best_dbscan if best_dbscan else DBSCAN(eps=best_eps, min_samples=min_samples_value).fit(data)
    
    fig, ax = plt.subplots(figsize=(14, 9))
    plot_clusters_detailed(ax, data, dbscan.labels, dbscan.n_clusters, f"DBSCAN (eps={best_eps:.1f}, min_samples={min_samples_value})")
    plt.tight_layout()
    plt.savefig("figures/06_dbscan.png", dpi=150, bbox_inches="tight")
    plt.close()

    return best_eps, min_samples_value, dbscan


def run_hdbscan(data):
    print("\n[6/7] Running HDBSCAN...")
    hdbscan = HDBSCAN(min_cluster_size=50, min_samples=5).fit(data)

    fig, ax = plt.subplots(figsize=(14, 9))
    plot_clusters_detailed(ax, data, hdbscan.labels, hdbscan.n_clusters, "HDBSCAN (min_cluster_size=50, min_samples=5)")
    plt.tight_layout()
    plt.savefig("figures/07_hdbscan.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    return hdbscan


def evaluate_and_compare(data, km_res, db_res, hdb_res):
    print("\n[7/7] Computing evaluation metrics & generating comparison plots...")
    optimal_k, kmeans = km_res
    eps_value, min_samples_value, dbscan = db_res
    hdbscan = hdb_res

    metrics = {}
    for name, model in [("K-Means", kmeans), ("DBSCAN", dbscan), ("HDBSCAN", hdbscan)]:
        metrics[name] = {
            "sil": silhouette_score(data, model.labels),
            "ch": calinski_harabasz_score(data, model.labels),
            "inertia": getattr(model, "inertia", compute_inertia(data, model.labels)),
            "n_clusters": optimal_k if name == "K-Means" else model.n_clusters,
            "noise": 0 if name == "K-Means" else np.sum(model.labels == -1)
        }

    metrics_df = pd.DataFrame({
        "Metric": ["Silhouette Score", "Calinski-Harabasz Index", "Inertia (SSE)", "Number of Clusters", "Noise Points", "Noise %"],
        "K-Means": [f"{metrics['K-Means']['sil']:.4f}", f"{metrics['K-Means']['ch']:.2f}", f"{metrics['K-Means']['inertia']:,.0f}", metrics["K-Means"]["n_clusters"], 0, "0.0%"],
        "DBSCAN": [f"{metrics['DBSCAN']['sil']:.4f}", f"{metrics['DBSCAN']['ch']:.2f}", f"{metrics['DBSCAN']['inertia']:,.0f}", metrics["DBSCAN"]["n_clusters"], metrics["DBSCAN"]["noise"], f"{metrics['DBSCAN']['noise']/len(data)*100:.1f}%"],
        "HDBSCAN": [f"{metrics['HDBSCAN']['sil']:.4f}", f"{metrics['HDBSCAN']['ch']:.2f}", f"{metrics['HDBSCAN']['inertia']:,.0f}", metrics["HDBSCAN"]["n_clusters"], metrics["HDBSCAN"]["noise"], f"{metrics['HDBSCAN']['noise']/len(data)*100:.1f}%"],
    })
    print("\n" + "=" * 70 + "\n  COMPARISON SUMMARY\n" + "=" * 70)
    print(metrics_df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    plot_clusters(axes[0], data, kmeans.labels, optimal_k, f"K-Means (k={optimal_k})")
    plot_clusters(axes[1], data, dbscan.labels, dbscan.n_clusters, f"DBSCAN (eps={eps_value:.1f})")
    plot_clusters(axes[2], data, hdbscan.labels, hdbscan.n_clusters, f"HDBSCAN (min_size={hdbscan.min_cluster_size})")
    plt.suptitle("Clustering Algorithm Comparison", fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("figures/08_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    algorithms = ["K-Means", "DBSCAN", "HDBSCAN"]
    bar_colors = ["#2196F3", "#FF9800", "#4CAF50"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sil_scores = [metrics[a]["sil"] for a in algorithms]
    bars = axes[0].bar(algorithms, sil_scores, color=bar_colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Silhouette Score", fontsize=14, fontweight="bold")
    axes[0].set_ylim(min(0, min(sil_scores) - 0.1), max(sil_scores) + 0.1)

    ch_scores = [metrics[a]["ch"] for a in algorithms]
    bars = axes[1].bar(algorithms, ch_scores, color=bar_colors, edgecolor="white", linewidth=1.5)
    axes[1].set_title("Calinski-Harabasz Index", fontsize=14, fontweight="bold")

    x = np.arange(len(algorithms))
    width = 0.35
    n_cl_list = [metrics[a]["n_clusters"] for a in algorithms]
    n_noise_list = [metrics[a]["noise"] for a in algorithms]
    axes[2].bar(x - width/2, n_cl_list, width, label="Clusters", color=bar_colors, edgecolor="white", linewidth=1.5)
    axes[2].bar(x + width/2, [nn/len(data)*100 for nn in n_noise_list], width, label="Noise %", color=["#90CAF9", "#FFE0B2", "#A5D6A7"], edgecolor="white", linewidth=1.5)
    axes[2].set_title("Clusters Found & Noise %", fontsize=14, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(algorithms)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("figures/09_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_cluster_sizes(axes[0], kmeans.labels, optimal_k, "K-Means", "#2196F3")
    plot_cluster_sizes(axes[1], dbscan.labels, dbscan.n_clusters, "DBSCAN", "#FF9800")
    plot_cluster_sizes(axes[2], hdbscan.labels, hdbscan.n_clusters, "HDBSCAN", "#4CAF50")
    plt.tight_layout()
    plt.savefig("figures/10_cluster_sizes.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    return metrics


def print_analysis(km_res, db_res, hdb_res, metrics):
    optimal_k, _ = km_res
    eps_value, min_samples_value, _ = db_res
    print("\n" + "=" * 70 + "\n  ANALYSIS & DISCUSSION\n" + "=" * 70)
    print(f"""
K-MEANS (k={optimal_k}):
  • Silhouette: {metrics['K-Means']['sil']:.4f} | CH Index: {metrics['K-Means']['ch']:.2f}
  • Assigns EVERY point to a cluster — no noise detection.

DBSCAN (eps={eps_value:.1f}, min_samples={min_samples_value}):
  • Silhouette: {metrics['DBSCAN']['sil']:.4f} | CH Index: {metrics['DBSCAN']['ch']:.2f}
  • Found {metrics['DBSCAN']['n_clusters']} clusters + {metrics['DBSCAN']['noise']} noise points.
  • Single global eps struggles with varying density.

HDBSCAN (min_cluster_size=50, min_samples=5):
  • Silhouette: {metrics['HDBSCAN']['sil']:.4f} | CH Index: {metrics['HDBSCAN']['ch']:.2f}
  • Found {metrics['HDBSCAN']['n_clusters']} clusters + {metrics['HDBSCAN']['noise']} noise points.
  • Handles varying densities via hierarchical approach.
""")
    print("All figures saved to figures/ directory.\nDone!")
