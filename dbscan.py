import numpy as np
from collections import deque


class DBSCAN:
    """DBSCAN clustering implemented from scratch."""

    def __init__(self, eps=30.0, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.core_sample_indices = None
        self.n_clusters = 0

    def fit(self, X):
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)

        print("  Computing neighborhoods...")
        neighborhoods = []
        chunk_size = 500
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            dists = np.sqrt(
                ((X[start:end, np.newaxis, :] - X[np.newaxis, :, :]) ** 2).sum(axis=2)
            )
            for i in range(end - start):
                neighborhoods.append(np.where(dists[i] <= self.eps)[0])

        is_core = np.array([len(neighborhoods[i]) >= self.min_samples for i in range(n_samples)])
        self.core_sample_indices = np.where(is_core)[0]
        print(f"  Found {len(self.core_sample_indices)} core points.")

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
