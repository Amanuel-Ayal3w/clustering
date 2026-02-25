import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

CLUSTER_COLORS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#264653", "#6A0572", "#AB83A1", "#1D3557", "#A8DADC",
]


def draw_cluster_ellipse(ax, points, color, alpha=0.15, n_std=2.0):
    """Draw a 2σ confidence ellipse around cluster points."""
    if len(points) < 3:
        return

    mean = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2 * n_std * np.sqrt(max(eigenvalues[0], 0))
    height = 2 * n_std * np.sqrt(max(eigenvalues[1], 0))

    # Filled semi-transparent background
    ellipse_fill = Ellipse(
        xy=mean, width=width, height=height, angle=angle,
        facecolor=color, edgecolor="none", alpha=alpha
    )
    ax.add_patch(ellipse_fill)

    # Solid, thick edge for clear, visible boundary
    ellipse_edge = Ellipse(
        xy=mean, width=width, height=height, angle=angle,
        facecolor="none", edgecolor=color,
        linewidth=3, linestyle="-", alpha=0.9
    )
    ax.add_patch(ellipse_edge)


def plot_clusters_detailed(ax, data, labels, n_clusters, title,
                           show_ellipses=True, show_centroids=True,
                           show_annotations=True):
    """Enhanced cluster plot with ellipses, centroid markers, and annotations."""
    non_empty = [i for i in range(n_clusters) if np.sum(labels == i) > 0]

    noise = labels == -1
    if noise.any():
        ax.scatter(
            data[noise, 0], data[noise, 1],
            s=8, alpha=0.15, c="#CCCCCC", marker=".",
            label=f"Noise ({noise.sum()})", zorder=1,
        )

    for ci, i in enumerate(non_empty):
        m = labels == i
        pts = data[m]
        color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
        centroid = pts.mean(axis=0)

        if show_ellipses and len(pts) > 10:
            draw_cluster_ellipse(ax, pts, color, alpha=0.20, n_std=2.0)

        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=10, alpha=0.55, color=color, edgecolors="none",
            label=f"C{i} ({m.sum()} pts)", zorder=2,
        )

        if show_centroids:
            ax.scatter(
                centroid[0], centroid[1],
                s=180, color=color, marker="D",
                edgecolors="white", linewidths=2, zorder=4,
            )

        if show_annotations:
            ax.annotate(
                f"C{i}\n({m.sum()})",
                xy=centroid, fontsize=8, fontweight="bold",
                ha="center", va="bottom",
                xytext=(0, 14), textcoords="offset points",
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="white",
                    edgecolor=color, alpha=0.85, linewidth=1.5,
                ),
                zorder=5,
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.legend(fontsize=7, markerscale=2.5, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.15, linestyle="--")


def plot_clusters(ax, data, labels, n_clusters, title):
    """Compact version for side-by-side comparison (ellipses, no annotations)."""
    plot_clusters_detailed(
        ax, data, labels, n_clusters, title,
        show_ellipses=True, show_centroids=True, show_annotations=False,
    )


def plot_cluster_sizes(ax, labels, n_clusters, title, color):
    """Bar chart of cluster sizes."""
    sizes, names = [], []
    for i in range(n_clusters):
        sizes.append(np.sum(labels == i))
        names.append(f"C{i}")
    n_noise = np.sum(labels == -1)
    if n_noise > 0:
        sizes.append(n_noise)
        names.append("Noise")
    c_list = [color] * n_clusters + (["lightgray"] if n_noise > 0 else [])
    bars = ax.bar(names, sizes, color=c_list, edgecolor="white", linewidth=1.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Points")
    for bar, val in zip(bars, sizes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(sizes) * 0.02,
            str(val), ha="center", fontsize=10,
        )
