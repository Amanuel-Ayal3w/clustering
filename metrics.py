import numpy as np


def silhouette_score(X, labels):
    """Mean Silhouette Coefficient (subsampled for efficiency)."""
    mask = labels >= 0
    X_v, lbl_v = X[mask], labels[mask]
    n = len(X_v)
    if n == 0:
        return 0.0
    unique = np.unique(lbl_v)
    if len(unique) < 2:
        return 0.0

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
    """Calinski-Harabasz Index: between- to within-cluster variance ratio."""
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
        for l in unique if l in centroids_map
    )
