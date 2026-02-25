import numpy as np


class KMeans:
    """K-Means clustering with K-Means++ initialization."""

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
        """(n, d) x (k, d) -> (n, k) distance matrix."""
        return np.sqrt(
            ((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2)
        )

    def _init_centroids_plusplus(self, X, rng):
        n = X.shape[0]
        centroids = np.empty((self.k, X.shape[1]))
        centroids[0] = X[rng.integers(n)]

        for i in range(1, self.k):
            dists = self._euclidean_distances(X, centroids[:i])
            probs = dists.min(axis=1) ** 2
            probs /= probs.sum()
            centroids[i] = X[rng.choice(n, p=probs)]

        return centroids

    def _single_run(self, X, rng):
        centroids = self._init_centroids_plusplus(X, rng)

        for iteration in range(self.max_iter):
            distances = self._euclidean_distances(X, centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.empty_like(centroids)
            for j in range(self.k):
                pts = X[labels == j]
                new_centroids[j] = pts.mean(axis=0) if len(pts) > 0 else X[rng.integers(X.shape[0])]

            shift = np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)).max()
            centroids = new_centroids
            if shift < self.tol:
                break

        distances = self._euclidean_distances(X, centroids)
        labels = np.argmin(distances, axis=1)
        inertia = sum(np.sum((X[labels == j] - centroids[j]) ** 2) for j in range(self.k))
        return centroids, labels, inertia, iteration + 1

    def fit(self, X):
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
        distances = self._euclidean_distances(X, self.centroids)
        return np.argmin(distances, axis=1)
