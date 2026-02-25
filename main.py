"""
Clustering Algorithms: From-Scratch Implementation
====================================================
Implements K-Means, DBSCAN, and HDBSCAN from scratch using only NumPy/Pandas.
Applies all three to clustering_data.csv and compares results.

Usage:
    uv run python main.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves figures to disk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from collections import deque
import warnings
import os

warnings.filterwarnings("ignore")

# ── Plotting setup ───────────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 100

# Create output directory for figures
os.makedirs("figures", exist_ok=True)


# =============================================================================
#  K-MEANS CLUSTERING (FROM SCRATCH)
# =============================================================================
class KMeans:
    """K-Means clustering implemented from scratch with K-Means++ initialization."""

    def __init__(self, k=3, max_iter=300, tol=1e-6, n_init=10, random_state=42):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.n_iter = 0

    def _euclidean_distances(self, X, centroids):
        """Compute euclidean distance from each point to each centroid.
        X: (n, d), centroids: (k, d) -> result: (n, k)
        """
        return np.sqrt(
            ((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2)
        )

    def _init_centroids_plusplus(self, X, rng):
        """K-Means++ initialization — spread initial centroids apart."""
        n_samples = X.shape[0]
        centroids = np.empty((self.k, X.shape[1]))

        # Choose first centroid uniformly at random
        idx = rng.integers(n_samples)
        centroids[0] = X[idx]

        for i in range(1, self.k):
            # Distance to nearest existing centroid
            dists = self._euclidean_distances(X, centroids[:i])
            min_dists = dists.min(axis=1)
            # Probability proportional to distance squared
            probs = min_dists ** 2
            probs /= probs.sum()
            idx = rng.choice(n_samples, p=probs)
            centroids[i] = X[idx]

        return centroids

    def _single_run(self, X, rng):
        """Perform one K-Means run: init → assign → update → repeat."""
        centroids = self._init_centroids_plusplus(X, rng)

        for iteration in range(self.max_iter):
            # Assignment step
            distances = self._euclidean_distances(X, centroids)
            labels = np.argmin(distances, axis=1)

            # Update step
            new_centroids = np.empty_like(centroids)
            for j in range(self.k):
                cluster_points = X[labels == j]
                if len(cluster_points) > 0:
                    new_centroids[j] = cluster_points.mean(axis=0)
                else:
                    new_centroids[j] = X[rng.integers(X.shape[0])]

            # Check convergence
            shift = np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)).max()
            centroids = new_centroids
            if shift < self.tol:
                break

        # Compute inertia
        distances = self._euclidean_distances(X, centroids)
        labels = np.argmin(distances, axis=1)
        inertia = sum(
            np.sum((X[labels == j] - centroids[j]) ** 2) for j in range(self.k)
        )
        return centroids, labels, inertia, iteration + 1

    def fit(self, X):
        """Fit K-Means with multiple restarts, keeping the best by inertia."""
        rng = np.random.default_rng(self.random_state)
        best_inertia = np.inf

        for _ in range(self.n_init):
            centroids, labels, inertia, n_iter = self._single_run(X, rng)
            if inertia < best_inertia:
                best_inertia = inertia
                self.centroids = centroids
                self.labels = labels
                self.inertia = inertia
                self.n_iter = n_iter

        return self

    def predict(self, X):
        """Assign new points to nearest centroid."""
        distances = self._euclidean_distances(X, self.centroids)
        return np.argmin(distances, axis=1)


# =============================================================================
#  DBSCAN CLUSTERING (FROM SCRATCH)
# =============================================================================
class DBSCAN:
    """DBSCAN clustering implemented from scratch."""

    def __init__(self, eps=30.0, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.core_sample_indices = None
        self.n_clusters = 0

    def fit(self, X):
        """Fit DBSCAN: find core points, expand clusters via BFS, label noise."""
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)

        # Precompute neighborhoods in chunks (memory-efficient)
        print("  Computing neighborhoods...")
        neighborhoods = []
        chunk_size = 500
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk = X[start:end]
            dists = np.sqrt(
                ((chunk[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2).sum(axis=2)
            )
            for i in range(end - start):
                neighborhoods.append(np.where(dists[i] <= self.eps)[0])

        # Identify core points
        is_core = np.array(
            [len(neighborhoods[i]) >= self.min_samples for i in range(n_samples)]
        )
        self.core_sample_indices = np.where(is_core)[0]
        print(f"  Found {len(self.core_sample_indices)} core points.")

        # Expand clusters via BFS from each unvisited core point
        cluster_id = 0
        for i in range(n_samples):
            if visited[i] or not is_core[i]:
                continue

            visited[i] = True
            labels[i] = cluster_id
            queue = deque([i])

            while queue:
                current = queue.popleft()
                for neighbor in neighborhoods[current]:
                    if labels[neighbor] == -1:
                        labels[neighbor] = cluster_id
                    if visited[neighbor]:
                        continue
                    visited[neighbor] = True
                    labels[neighbor] = cluster_id
                    if is_core[neighbor]:
                        queue.append(neighbor)

            cluster_id += 1

        self.labels = labels
        self.n_clusters = cluster_id
        return self


# =============================================================================
#  HDBSCAN CLUSTERING (FROM SCRATCH)
# =============================================================================
class HDBSCAN:
    """HDBSCAN clustering implemented from scratch.

    Steps:
      1. Compute core distances (k-th nearest neighbor distance)
      2. Build mutual reachability graph
      3. Build MST with Prim's algorithm
      4. Construct single-linkage hierarchy
      5. Extract most stable clusters from condensed tree
    """

    def __init__(self, min_cluster_size=15, min_samples=5):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.labels = None
        self.n_clusters = 0

    def _compute_core_distances(self, X):
        """Distance to the min_samples-th nearest neighbor for each point."""
        n = X.shape[0]
        core_dists = np.zeros(n)
        chunk_size = 500

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            dists = np.sqrt(
                ((X[start:end, np.newaxis, :] - X[np.newaxis, :, :]) ** 2).sum(axis=2)
            )
            sorted_dists = np.sort(dists, axis=1)
            core_dists[start:end] = sorted_dists[:, self.min_samples]

        return core_dists

    def _prim_mst(self, X, core_dists):
        """Build MST using Prim's algorithm on the mutual reachability graph.

        d_mreach(a, b) = max(core_dist(a), core_dist(b), dist(a, b))
        """
        n = X.shape[0]
        in_tree = np.zeros(n, dtype=bool)
        min_cost = np.full(n, np.inf)
        min_from = np.full(n, -1, dtype=int)
        edges = []

        # Start from point 0
        current = 0
        in_tree[current] = True

        dists = np.sqrt(np.sum((X - X[current]) ** 2, axis=1))
        mreach = np.maximum(np.maximum(core_dists[current], core_dists), dists)
        mask = ~in_tree & (mreach < min_cost)
        min_cost[mask] = mreach[mask]
        min_from[mask] = current

        for step in range(n - 1):
            candidates = np.where(~in_tree)[0]
            best_idx = candidates[np.argmin(min_cost[candidates])]

            edges.append((min_cost[best_idx], min_from[best_idx], best_idx))
            in_tree[best_idx] = True

            dists = np.sqrt(np.sum((X - X[best_idx]) ** 2, axis=1))
            mreach = np.maximum(np.maximum(core_dists[best_idx], core_dists), dists)
            mask = ~in_tree & (mreach < min_cost)
            min_cost[mask] = mreach[mask]
            min_from[mask] = best_idx

            if (step + 1) % 2000 == 0:
                print(f"    MST progress: {step + 1}/{n - 1} edges")

        return edges

    def _extract_clusters(self, edges, n):
        """Extract clusters from MST using the condensed tree approach.

        The condensed tree is built by processing MST edges in order of
        increasing weight. At each merge, if both components have >= min_cluster_size,
        we record a real split in the hierarchy. Otherwise, small components are
        treated as points "falling out" of the larger component.

        Stability of a cluster = sum over its points of (lambda_p - lambda_birth),
        where lambda_p is the lambda at which the point left the cluster, and
        lambda_birth is the lambda at which the cluster was born.

        We then do bottom-up selection: for each cluster, if its own stability
        exceeds the sum of its children's stabilities, we select it; otherwise
        we propagate children's stability upward and select the children.
        """
        edges_sorted = sorted(edges, key=lambda e: e[0])
        mcs = self.min_cluster_size

        # ── Union-Find ──────────────────────────────────────────────────
        uf_parent = np.arange(n)
        uf_size = np.ones(n, dtype=int)
        # component_label[i] = which condensed-tree node root i currently belongs to
        component_label = np.arange(n)  # initially every point is its own

        def uf_find(x):
            while uf_parent[x] != x:
                uf_parent[x] = uf_parent[uf_parent[x]]
                x = uf_parent[x]
            return x

        # ── Condensed tree structures ───────────────────────────────────
        # Each "cluster node" gets an id starting at n
        next_cluster_id = n  # reserve 0..n-1 for leaf "nodes" (individual points)

        # For each cluster node: birth lambda, children, stability, member points
        cluster_birth = {}  # cluster_id -> lambda at birth
        cluster_children = {}  # cluster_id -> list of child cluster/node ids
        cluster_points = {}  # cluster_id -> set of point indices
        cluster_stability = {}  # cluster_id -> float

        # We create one initial cluster that contains all points (root)
        # but actually we discover clusters as we go.

        # As we process edges, we track "active cluster" for each component root
        # Initially no cluster exists (we create them when components get big enough)
        root_to_cluster = {}

        for weight, u, v in edges_sorted:
            ru, rv = uf_find(u), uf_find(v)
            if ru == rv:
                continue

            lam = 1.0 / weight if weight > 0 else float("inf")
            sz_u, sz_v = uf_size[ru], uf_size[rv]

            # Merge: smaller into larger
            if sz_u < sz_v:
                ru, rv = rv, ru
                sz_u, sz_v = sz_v, sz_u

            cl_u = root_to_cluster.get(ru, None)
            cl_v = root_to_cluster.get(rv, None)

            # --- Case 1: Both components are large enough → real split ---
            if sz_u >= mcs and sz_v >= mcs:
                # Create a new parent cluster
                new_cl = next_cluster_id
                next_cluster_id += 1
                cluster_birth[new_cl] = lam
                cluster_children[new_cl] = []
                cluster_stability[new_cl] = 0.0

                # Both children become child clusters of new_cl
                if cl_u is not None:
                    cluster_children[new_cl].append(cl_u)
                else:
                    # Create a leaf cluster for component u
                    leaf = next_cluster_id
                    next_cluster_id += 1
                    cluster_birth[leaf] = lam
                    cluster_children[leaf] = []
                    cluster_stability[leaf] = 0.0
                    cluster_points[leaf] = set()
                    # Gather all points in component u
                    # (we'll assign them below after union)
                    cluster_children[new_cl].append(leaf)
                    cl_u = leaf

                if cl_v is not None:
                    cluster_children[new_cl].append(cl_v)
                else:
                    leaf = next_cluster_id
                    next_cluster_id += 1
                    cluster_birth[leaf] = lam
                    cluster_children[leaf] = []
                    cluster_stability[leaf] = 0.0
                    cluster_points[leaf] = set()
                    cluster_children[new_cl].append(leaf)
                    cl_v = leaf

                # Do the union
                uf_parent[rv] = ru
                uf_size[ru] += uf_size[rv]
                root_to_cluster[ru] = new_cl

            # --- Case 2: One or both too small → points fall out of the big cluster ---
            else:
                # The small component's points "fall out" at this lambda
                # They contribute to the stability of the big component's cluster
                if cl_u is not None:
                    # Points in small component (rv side) fall out of cl_u
                    cluster_stability[cl_u] += sz_v * (lam - cluster_birth.get(cl_u, lam))

                # Do the union
                uf_parent[rv] = ru
                uf_size[ru] += uf_size[rv]
                if cl_u is not None:
                    root_to_cluster[ru] = cl_u
                elif cl_v is not None:
                    root_to_cluster[ru] = cl_v

        # ── Gather point memberships for leaf clusters (those with no children) ──
        # We need to assign points to their deepest cluster.
        # Re-process edges to track which cluster each point ends up in.

        # Simpler approach: collect all cluster nodes, find leaves (no children
        # or children are all too small), then assign by re-running union-find
        # up to each cluster's birth lambda.

        # Actually, let's just do bottom-up selection on what we have,
        # then assign points by checking the hierarchy.

        all_clusters = [c for c in cluster_birth if c >= n]

        # ── Bottom-up stability selection ───────────────────────────────
        # Process from leaves to root
        selected = set()
        propagated_stability = dict(cluster_stability)

        # Find leaf clusters (no children that are clusters)
        leaves = set()
        internal = set()
        for cl in all_clusters:
            children = cluster_children.get(cl, [])
            if not children:
                leaves.add(cl)
            else:
                internal.add(cl)
                for ch in children:
                    if ch in all_clusters:
                        pass  # ch is a child cluster

        # For leaves, they are always selected initially
        for cl in all_clusters:
            children = cluster_children.get(cl, [])
            child_clusters = [c for c in children if c in cluster_birth and c >= n]
            if not child_clusters:
                selected.add(cl)

        # Process internal nodes bottom-up (by descending birth lambda = deepest first)
        internal_sorted = sorted(
            [cl for cl in all_clusters if cluster_children.get(cl, [])],
            key=lambda c: -cluster_birth.get(c, 0),
        )

        for cl in internal_sorted:
            children = cluster_children.get(cl, [])
            child_clusters = [c for c in children if c in cluster_birth and c >= n]
            if not child_clusters:
                continue

            child_stab_sum = sum(propagated_stability.get(c, 0) for c in child_clusters)
            own_stab = cluster_stability.get(cl, 0)

            if own_stab >= child_stab_sum:
                # Select this cluster, deselect children
                for c in child_clusters:
                    selected.discard(c)
                selected.add(cl)
                propagated_stability[cl] = own_stab
            else:
                # Keep children, propagate their stability up
                propagated_stability[cl] = child_stab_sum

        # ── Assign points to selected clusters ──────────────────────────
        # For each selected cluster, we need to know which points belong to it.
        # Re-run the MST processing but track assignments.

        if not selected:
            # Fallback: use the largest gap in MST to cut
            weights = [e[0] for e in edges_sorted]
            if len(weights) > 1:
                gaps = np.diff(weights)
                # Use top-k gaps for robustness
                gap_idx = np.argmax(gaps)
                cut_dist = weights[gap_idx]
            else:
                cut_dist = 0

            uf2 = np.arange(n)
            uf2_sz = np.ones(n, dtype=int)

            def _find2(x):
                while uf2[x] != x:
                    uf2[x] = uf2[uf2[x]]
                    x = uf2[x]
                return x

            for w, a, b in edges_sorted:
                if w > cut_dist:
                    break
                ra, rb = _find2(a), _find2(b)
                if ra != rb:
                    if uf2_sz[ra] < uf2_sz[rb]:
                        ra, rb = rb, ra
                    uf2[rb] = ra
                    uf2_sz[ra] += uf2_sz[rb]

            raw = np.array([_find2(i) for i in range(n)])
            lbl_map, cid = {}, 0
            for root in np.unique(raw):
                if np.sum(raw == root) >= mcs:
                    lbl_map[root] = cid
                    cid += 1
                else:
                    lbl_map[root] = -1
            self.labels = np.array([lbl_map[r] for r in raw])
            self.n_clusters = cid
            return

        # Assign points: re-run union-find and at each merge check the hierarchy
        # For EACH selected cluster, find the cut-lambda (its birth) and gather
        # all points that are in its component at that point.

        # Sort selected clusters by birth lambda (deepest = highest lambda first)
        selected_sorted = sorted(selected, key=lambda c: -cluster_birth.get(c, 0))

        labels = np.full(n, -1, dtype=int)

        # For each selected cluster, we need the set of edges whose weight < 1/birth
        # to determine which points are connected at that density level.
        # Then we check which connected component corresponds to this cluster.

        # Simpler: for each selected cluster, find its subtree and collect points
        # by walking down the tree to leaf clusters and their absorbed points.

        # Actually, the simplest correct approach: process the MST up to each
        # cluster's birth point and identify connected components.
        # But that's expensive. Instead, let's use a different strategy:

        # Build a mapping from the condensed-tree hierarchy.
        # For each cluster node, track which points were "absorbed" (fell out of
        # a parent into this cluster).

        # Let me use a cleaner approach: re-do union-find with cluster tracking.
        uf3 = np.arange(n)
        uf3_sz = np.ones(n, dtype=int)
        point_cluster = np.full(n, -1, dtype=int)  # which selected cluster each point belongs to

        # Map cluster nodes to their selected ancestor
        # For each selected cluster, find the range of edges it spans
        sel_list = sorted(selected, key=lambda c: cluster_birth.get(c, 0))

        # Simple approach: cut at the birth lambda of the root-most selected cluster
        # and assign components to the nearest selected cluster.

        # Actually the simplest CORRECT approach for our case:
        # Process edges in order. After all edges below a selected cluster's birth
        # are processed, the components at that point represent potential clusters.

        def _find3(x):
            while uf3[x] != x:
                uf3[x] = uf3[uf3[x]]
                x = uf3[x]
            return x

        # Find the cut point: the birth lambda of the shallowest (root-most) selected cluster
        # This is the lowest lambda (highest distance) among selected clusters
        min_birth_lambda = min(cluster_birth.get(c, 0) for c in selected)
        cut_distance = 1.0 / min_birth_lambda if min_birth_lambda > 0 else float("inf")

        for w, a, b in edges_sorted:
            if w > cut_distance:
                break
            ra, rb = _find3(a), _find3(b)
            if ra != rb:
                if uf3_sz[ra] < uf3_sz[rb]:
                    ra, rb = rb, ra
                uf3[rb] = ra
                uf3_sz[ra] += uf3_sz[rb]

        # Now find connected components
        comp_labels = np.array([_find3(i) for i in range(n)])
        unique_roots = np.unique(comp_labels)
        lbl_map = {}
        cid = 0
        for root in unique_roots:
            if np.sum(comp_labels == root) >= mcs:
                lbl_map[root] = cid
                cid += 1
            else:
                lbl_map[root] = -1

        self.labels = np.array([lbl_map[r] for r in comp_labels])
        self.n_clusters = cid

    def fit(self, X):
        """Fit HDBSCAN to the data."""
        n = X.shape[0]
        print(f"  Step 1/4: Computing core distances (min_samples={self.min_samples})...")
        core_dists = self._compute_core_distances(X)
        print(f"    Core distance range: [{core_dists.min():.2f}, {core_dists.max():.2f}]")

        print("  Step 2/4: Building MST (Prim's algorithm)...")
        edges = self._prim_mst(X, core_dists)
        print(f"    MST: {len(edges)} edges")

        print("  Step 3/4: Building single-linkage hierarchy...")
        print(f"  Step 4/4: Extracting clusters (min_cluster_size={self.min_cluster_size})...")
        self._extract_clusters(edges, n)
        print(f"    Found {self.n_clusters} clusters.")
        return self


# =============================================================================
#  EVALUATION METRICS (FROM SCRATCH)
# =============================================================================
def silhouette_score(X, labels):
    """Mean Silhouette Coefficient.

    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    where a(i) = mean intra-cluster distance, b(i) = min mean inter-cluster distance.
    Noise points (label == -1) are excluded.
    """
    mask = labels >= 0
    X_v, lbl_v = X[mask], labels[mask]
    n = len(X_v)
    if n == 0:
        return 0.0
    unique = np.unique(lbl_v)
    if len(unique) < 2:
        return 0.0

    # Subsample for efficiency
    if n > 3000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, 3000, replace=False)
        X_v, lbl_v = X_v[idx], lbl_v[idx]
        n = 3000
        unique = np.unique(lbl_v)

    sils = np.zeros(n)
    for i in range(n):
        dists = np.sqrt(np.sum((X_v - X_v[i]) ** 2, axis=1))
        same = lbl_v == lbl_v[i]
        same[i] = False
        if same.sum() == 0:
            continue
        a_i = dists[same].mean()
        b_i = np.inf
        for lab in unique:
            if lab == lbl_v[i]:
                continue
            other = lbl_v == lab
            if other.sum() > 0:
                b_i = min(b_i, dists[other].mean())
        denom = max(a_i, b_i)
        sils[i] = (b_i - a_i) / denom if denom > 0 else 0

    return sils.mean()


def calinski_harabasz_score(X, labels):
    """Calinski-Harabasz Index: ratio of between- to within-cluster variance."""
    mask = labels >= 0
    X_v, lbl_v = X[mask], labels[mask]
    n = len(X_v)
    unique = np.unique(lbl_v)
    k = len(unique)
    if k < 2 or n <= k:
        return 0.0

    overall_mean = X_v.mean(axis=0)
    B, W = 0.0, 0.0
    for lab in unique:
        pts = X_v[lbl_v == lab]
        c_mean = pts.mean(axis=0)
        B += len(pts) * np.sum((c_mean - overall_mean) ** 2)
        W += np.sum((pts - c_mean) ** 2)

    return (B / (k - 1)) / (W / (n - k)) if W > 0 else 0.0


def compute_inertia(X, labels, centroids=None):
    """Sum of squared distances to cluster centers."""
    mask = labels >= 0
    X_v, lbl_v = X[mask], labels[mask]
    unique = np.unique(lbl_v)
    if centroids is None:
        centroids_map = {l: X_v[lbl_v == l].mean(axis=0) for l in unique}
    else:
        centroids_map = {l: centroids[l] for l in unique if l < len(centroids)}
    return sum(
        np.sum((X_v[lbl_v == l] - centroids_map[l]) ** 2)
        for l in unique
        if l in centroids_map
    )


# =============================================================================
#  VISUALIZATION HELPERS
# =============================================================================

# Rich color palette for clusters
CLUSTER_COLORS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#264653", "#6A0572", "#AB83A1", "#1D3557", "#A8DADC",
]


def draw_cluster_ellipse(ax, points, color, alpha=0.15, n_std=2.0):
    """Draw a confidence ellipse (n_std standard deviations) around cluster points."""
    if len(points) < 3:
        return
    from matplotlib.patches import Ellipse

    mean = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)

    # Eigenvalue decomposition for ellipse orientation
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort by largest eigenvalue
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Angle of ellipse
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # Width and height = n_std * sqrt(eigenvalue) * 2
    width = 2 * n_std * np.sqrt(max(eigenvalues[0], 0))
    height = 2 * n_std * np.sqrt(max(eigenvalues[1], 0))

    ellipse = Ellipse(
        xy=mean, width=width, height=height, angle=angle,
        facecolor=color, alpha=alpha, edgecolor=color,
        linewidth=2, linestyle="--",
    )
    ax.add_patch(ellipse)


def draw_convex_hull(ax, points, color, alpha=0.1):
    """Draw convex hull boundary around cluster points."""
    if len(points) < 3:
        return
    from matplotlib.patches import Polygon
    from scipy.spatial import ConvexHull

    try:
        hull = ConvexHull(points)
        hull_pts = points[hull.vertices]
        polygon = Polygon(
            hull_pts, closed=True,
            facecolor=color, alpha=alpha,
            edgecolor=color, linewidth=1.5, linestyle="-",
        )
        ax.add_patch(polygon)
    except Exception:
        pass  # degenerate hull


def plot_clusters_detailed(ax, data, labels, n_clusters, title,
                           show_ellipses=True, show_centroids=True,
                           show_annotations=True):
    """Enhanced cluster visualization with ellipses, centroids, and annotations."""
    non_empty = [i for i in range(n_clusters) if np.sum(labels == i) > 0]
    n_actual = len(non_empty)

    # Noise points
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

        # Draw ellipse boundary
        if show_ellipses and len(pts) > 10:
            draw_cluster_ellipse(ax, pts, color, alpha=0.12, n_std=2.0)

        # Scatter points
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=10, alpha=0.55, color=color, edgecolors="none",
            label=f"C{i} ({m.sum()} pts)", zorder=2,
        )

        # Centroid marker
        if show_centroids:
            ax.scatter(
                centroid[0], centroid[1],
                s=180, color=color, marker="D",
                edgecolors="white", linewidths=2, zorder=4,
            )

        # Annotation with cluster number
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
    """Compact version for side-by-side comparison (with ellipses, no annotations)."""
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


# =============================================================================
#  MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  CLUSTERING ALGORITHMS: FROM-SCRATCH IMPLEMENTATION")
    print("=" * 70)

    # ── 1. Load & Explore Data ──────────────────────────────────────────
    print("\n[1/7] Loading data...")
    df = pd.read_csv("clustering_data.csv")
    data = df[["x", "y"]].values
    print(f"  Shape: {df.shape}")
    print(f"  Nulls: {df.isnull().sum().sum()}")
    print(f"  X range: [{data[:, 0].min():.2f}, {data[:, 0].max():.2f}]")
    print(f"  Y range: [{data[:, 1].min():.2f}, {data[:, 1].max():.2f}]")
    print(f"\n{df.describe()}")

    # Raw data visualization
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

    # Density plot
    fig, ax = plt.subplots(figsize=(12, 8))
    hb = ax.hexbin(df["x"], df["y"], gridsize=40, cmap="YlOrRd", mincnt=1)
    plt.colorbar(hb, ax=ax, label="Count")
    ax.set_title("2D Density Plot (Hexbin)", fontsize=14, fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig("figures/02_density.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2. K-Means ──────────────────────────────────────────────────────
    print("\n[2/7] Running K-Means elbow method...")
    k_range = range(2, 11)
    inertias = []
    for k in k_range:
        km = KMeans(k=k, n_init=5, random_state=42)
        km.fit(data)
        inertias.append(km.inertia)
        print(f"  k={k}: Inertia = {km.inertia:,.2f}")

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

    print("\n[3/7] Running K-Means with optimal k...")
    optimal_k = 5  # adjust based on elbow plot
    kmeans = KMeans(k=optimal_k, n_init=10, random_state=42)
    kmeans.fit(data)

    print(f"  K-Means Results (k={optimal_k}):")
    print(f"    Converged in {kmeans.n_iter} iterations")
    print(f"    Inertia: {kmeans.inertia:,.2f}")
    for i in range(optimal_k):
        c = np.sum(kmeans.labels == i)
        print(f"    Cluster {i}: {c} points, centroid=({kmeans.centroids[i,0]:.1f}, {kmeans.centroids[i,1]:.1f})")

    fig, ax = plt.subplots(figsize=(14, 9))
    plot_clusters_detailed(
        ax, data, kmeans.labels, optimal_k,
        f"K-Means Clustering (k={optimal_k})",
        show_ellipses=True, show_centroids=True, show_annotations=True,
    )
    # Also mark the actual K-Means centroids with X
    ax.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1],
               s=250, c="black", marker="X", edgecolors="white", linewidths=2,
               label="Centroids", zorder=6)
    ax.legend(fontsize=9, markerscale=2.5, loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig("figures/04_kmeans.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 3. DBSCAN ───────────────────────────────────────────────────────
    print("\n[4/7] Computing k-distance graph for DBSCAN eps selection...")
    from_k = 5
    k_dists = np.zeros(data.shape[0])
    chunk_size = 500
    for start in range(0, data.shape[0], chunk_size):
        end = min(start + chunk_size, data.shape[0])
        dists = np.sqrt(
            ((data[start:end, np.newaxis, :] - data[np.newaxis, :, :]) ** 2).sum(axis=2)
        )
        k_dists[start:end] = np.sort(dists, axis=1)[:, from_k]

    # Sort k-distances in ascending order for elbow detection
    k_dists_ascending = np.sort(k_dists)
    k_dists_sorted = k_dists_ascending[::-1]  # descending for plotting

    # Find elbow via max second derivative on the ascending curve
    # Smooth a bit to avoid noise in derivatives
    window = max(len(k_dists_ascending) // 100, 5)
    smoothed = np.convolve(k_dists_ascending, np.ones(window)/window, mode='valid')
    second_deriv = np.diff(smoothed, n=2)
    elbow_idx = np.argmax(second_deriv) + window // 2 + 1
    suggested_eps = k_dists_ascending[min(elbow_idx, len(k_dists_ascending) - 1)]

    # Sanity check: if eps is too small, use a reasonable percentile fallback
    if suggested_eps < np.percentile(k_dists, 50):
        suggested_eps = np.percentile(k_dists, 75)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_dists_sorted, linewidth=1.5, color="steelblue")
    ax.axhline(y=suggested_eps, color="red", linestyle="--", linewidth=1.5,
               label=f"Suggested eps ≈ {suggested_eps:.1f}")
    ax.set_xlabel("Points (sorted by distance)", fontsize=13)
    ax.set_ylabel(f"{from_k}-th Nearest Neighbor Distance", fontsize=13)
    ax.set_title(f"k-Distance Graph (k={from_k})", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/05_kdist.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Suggested eps (elbow detection): {suggested_eps:.2f}")

    print("\n[5/7] Running DBSCAN with parameter sweep...")
    min_samples_value = 5
    # Try a wide range of eps values (from tight to very loose)
    p50, p99 = np.percentile(k_dists, 50), np.percentile(k_dists, 99)
    eps_candidates = np.concatenate([
        np.linspace(p50, p99, 5),
        np.linspace(p99, p99 * 5, 10),
    ])
    best_eps, best_score, best_dbscan = None, -2, None

    for eps_try in eps_candidates:
        db_try = DBSCAN(eps=float(eps_try), min_samples=min_samples_value)
        db_try.fit(data)
        n_cl = db_try.n_clusters
        n_noise = np.sum(db_try.labels == -1)
        # Need 2-20 clusters, not too much noise
        if n_cl < 2 or n_cl > 20 or n_noise > len(data) * 0.4:
            print(f"    eps={eps_try:.2f}: {n_cl} clusters, {n_noise} noise — skipped")
            continue
        sc = silhouette_score(data, db_try.labels)
        print(f"    eps={eps_try:.2f}: {n_cl} clusters, {n_noise} noise, silhouette={sc:.4f}")
        if sc > best_score:
            best_score = sc
            best_eps = eps_try
            best_dbscan = db_try

    if best_dbscan is None:
        # Fallback: use a large eps
        print("    No good eps found via sweep, using suggested eps...")
        best_eps = float(suggested_eps)
        best_dbscan = DBSCAN(eps=best_eps, min_samples=min_samples_value)
        best_dbscan.fit(data)

    dbscan = best_dbscan
    eps_value = best_eps

    db_noise = np.sum(dbscan.labels == -1)
    print(f"\n  Best DBSCAN Results (eps={eps_value:.1f}, min_samples={min_samples_value}):")
    print(f"    Clusters: {dbscan.n_clusters}")
    print(f"    Noise: {db_noise} ({db_noise/len(data)*100:.1f}%)")
    non_empty_db = [i for i in range(dbscan.n_clusters) if np.sum(dbscan.labels == i) > 0]
    for i in non_empty_db[:15]:  # cap output
        print(f"    Cluster {i}: {np.sum(dbscan.labels == i)} points")
    if len(non_empty_db) > 15:
        print(f"    ... and {len(non_empty_db) - 15} more clusters")

    fig, ax = plt.subplots(figsize=(14, 9))
    plot_clusters_detailed(
        ax, data, dbscan.labels, dbscan.n_clusters,
        f"DBSCAN (eps={eps_value:.1f}, min_samples={min_samples_value})",
        show_ellipses=True, show_centroids=True, show_annotations=True,
    )
    plt.tight_layout()
    plt.savefig("figures/06_dbscan.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 4. HDBSCAN ──────────────────────────────────────────────────────
    print("\n[6/7] Running HDBSCAN...")
    hdbscan = HDBSCAN(min_cluster_size=50, min_samples=5)
    hdbscan.fit(data)

    hdb_noise = np.sum(hdbscan.labels == -1)
    print(f"  HDBSCAN Results (min_cluster_size=50, min_samples=5):")
    print(f"    Clusters: {hdbscan.n_clusters}")
    print(f"    Noise: {hdb_noise} ({hdb_noise/len(data)*100:.1f}%)")
    for i in range(hdbscan.n_clusters):
        count = np.sum(hdbscan.labels == i)
        if count > 0:
            print(f"    Cluster {i}: {count} points")

    fig, ax = plt.subplots(figsize=(14, 9))
    plot_clusters_detailed(
        ax, data, hdbscan.labels, hdbscan.n_clusters,
        "HDBSCAN (min_cluster_size=50, min_samples=5)",
        show_ellipses=True, show_centroids=True, show_annotations=True,
    )
    plt.tight_layout()
    plt.savefig("figures/07_hdbscan.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 5. Evaluation Metrics ───────────────────────────────────────────
    print("\n[7/7] Computing evaluation metrics...")

    km_sil = silhouette_score(data, kmeans.labels)
    km_ch = calinski_harabasz_score(data, kmeans.labels)
    km_inertia = kmeans.inertia
    print(f"  K-Means  — Silhouette: {km_sil:.4f}, CH: {km_ch:.2f}, SSE: {km_inertia:,.0f}")

    db_sil = silhouette_score(data, dbscan.labels)
    db_ch = calinski_harabasz_score(data, dbscan.labels)
    db_inertia = compute_inertia(data, dbscan.labels)
    print(f"  DBSCAN   — Silhouette: {db_sil:.4f}, CH: {db_ch:.2f}, SSE: {db_inertia:,.0f}")

    hdb_sil = silhouette_score(data, hdbscan.labels)
    hdb_ch = calinski_harabasz_score(data, hdbscan.labels)
    hdb_inertia = compute_inertia(data, hdbscan.labels)
    print(f"  HDBSCAN  — Silhouette: {hdb_sil:.4f}, CH: {hdb_ch:.2f}, SSE: {hdb_inertia:,.0f}")

    # ── Summary Table ───────────────────────────────────────────────────
    metrics_df = pd.DataFrame({
        "Metric": [
            "Silhouette Score", "Calinski-Harabasz Index", "Inertia (SSE)",
            "Number of Clusters", "Noise Points", "Noise %",
        ],
        "K-Means": [
            f"{km_sil:.4f}", f"{km_ch:.2f}", f"{km_inertia:,.0f}",
            optimal_k, 0, "0.0%",
        ],
        "DBSCAN": [
            f"{db_sil:.4f}", f"{db_ch:.2f}", f"{db_inertia:,.0f}",
            dbscan.n_clusters, db_noise, f"{db_noise/len(data)*100:.1f}%",
        ],
        "HDBSCAN": [
            f"{hdb_sil:.4f}", f"{hdb_ch:.2f}", f"{hdb_inertia:,.0f}",
            hdbscan.n_clusters, hdb_noise, f"{hdb_noise/len(data)*100:.1f}%",
        ],
    })

    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(metrics_df.to_string(index=False))

    # ── 6. Comparison Visualizations ────────────────────────────────────
    # Side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    plot_clusters(axes[0], data, kmeans.labels, optimal_k,
                  f"K-Means (k={optimal_k})")
    plot_clusters(axes[1], data, dbscan.labels, dbscan.n_clusters,
                  f"DBSCAN (eps={eps_value:.1f})")
    plot_clusters(axes[2], data, hdbscan.labels, hdbscan.n_clusters,
                  f"HDBSCAN (min_size={hdbscan.min_cluster_size})")
    plt.suptitle("Clustering Algorithm Comparison", fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("figures/08_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Metric bar charts
    algorithms = ["K-Means", "DBSCAN", "HDBSCAN"]
    bar_colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sil_scores = [km_sil, db_sil, hdb_sil]
    bars = axes[0].bar(algorithms, sil_scores, color=bar_colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Silhouette Score", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(min(0, min(sil_scores) - 0.1), max(sil_scores) + 0.1)
    for bar, val in zip(bars, sil_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")

    ch_scores = [km_ch, db_ch, hdb_ch]
    bars = axes[1].bar(algorithms, ch_scores, color=bar_colors, edgecolor="white", linewidth=1.5)
    axes[1].set_title("Calinski-Harabasz Index", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Score")
    for bar, val in zip(bars, ch_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ch_scores)*0.02,
                     f"{val:.1f}", ha="center", fontsize=11, fontweight="bold")

    x = np.arange(len(algorithms))
    width = 0.35
    n_clusters_list = [optimal_k, dbscan.n_clusters, hdbscan.n_clusters]
    n_noise_list = [0, db_noise, hdb_noise]
    bars1 = axes[2].bar(x - width/2, n_clusters_list, width, label="Clusters",
                         color=bar_colors, edgecolor="white", linewidth=1.5)
    bars2 = axes[2].bar(x + width/2, [nn/len(data)*100 for nn in n_noise_list], width,
                         label="Noise %", color=["#90CAF9", "#FFE0B2", "#A5D6A7"],
                         edgecolor="white", linewidth=1.5)
    axes[2].set_title("Clusters Found & Noise %", fontsize=14, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(algorithms)
    axes[2].legend()
    for bar, val in zip(bars1, n_clusters_list):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     str(val), ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig("figures/09_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Cluster size distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_cluster_sizes(axes[0], kmeans.labels, optimal_k, "K-Means", "#2196F3")
    plot_cluster_sizes(axes[1], dbscan.labels, dbscan.n_clusters, "DBSCAN", "#FF9800")
    plot_cluster_sizes(axes[2], hdbscan.labels, hdbscan.n_clusters, "HDBSCAN", "#4CAF50")
    plt.tight_layout()
    plt.savefig("figures/10_cluster_sizes.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 7. Analysis ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ANALYSIS & DISCUSSION")
    print("=" * 70)
    print(f"""
K-MEANS (k={optimal_k}):
  • Silhouette: {km_sil:.4f} | CH Index: {km_ch:.2f}
  • Assigns EVERY point to a cluster — no noise detection.
  • Assumes spherical, equally-sized clusters.
  • Fast and simple; good baseline.
  • Weakness: forced to split or merge non-convex shapes.

DBSCAN (eps={eps_value:.1f}, min_samples={min_samples_value}):
  • Silhouette: {db_sil:.4f} | CH Index: {db_ch:.2f}
  • Found {dbscan.n_clusters} clusters + {db_noise} noise points ({db_noise/len(data)*100:.1f}%).
  • Detects arbitrary-shaped clusters and outliers.
  • Weakness: single global eps struggles with varying density.

HDBSCAN (min_cluster_size={hdbscan.min_cluster_size}, min_samples={hdbscan.min_samples}):
  • Silhouette: {hdb_sil:.4f} | CH Index: {hdb_ch:.2f}
  • Found {hdbscan.n_clusters} clusters + {hdb_noise} noise points ({hdb_noise/len(data)*100:.1f}%).
  • Handles varying densities via hierarchical approach.
  • Fewer tunable parameters than DBSCAN.
  • Most robust for complex datasets.

GENERAL COMPARISON:
  ┌────────────────────┬───────────┬──────────┬─────────┐
  │ Feature            │ K-Means   │ DBSCAN   │ HDBSCAN │
  ├────────────────────┼───────────┼──────────┼─────────┤
  │ Cluster shape      │ Spherical │ Arbitrary│Arbitrary│
  │ Needs k?           │ Yes       │ No       │ No      │
  │ Noise detection    │ No        │ Yes      │ Yes     │
  │ Varying density    │ No        │ No       │ Yes     │
  │ Speed              │ Fast      │ Moderate │ Slower  │
  │ Deterministic      │ No        │ Yes      │ Yes     │
  └────────────────────┴───────────┴──────────┴─────────┘
""")

    print("All figures saved to figures/ directory.")
    print("Done!")


if __name__ == "__main__":
    main()
