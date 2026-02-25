import numpy as np


class HDBSCAN:
    """HDBSCAN clustering implemented from scratch.

    Pipeline: core distances -> mutual reachability graph -> MST (Prim's)
    -> condensed tree -> stability-based cluster extraction.
    """

    def __init__(self, min_cluster_size=15, min_samples=5):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.labels = None
        self.n_clusters = 0

    def _compute_core_distances(self, X):
        n = X.shape[0]
        core_dists = np.zeros(n)
        chunk_size = 500
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            dists = np.sqrt(
                ((X[start:end, np.newaxis, :] - X[np.newaxis, :, :]) ** 2).sum(axis=2)
            )
            core_dists[start:end] = np.sort(dists, axis=1)[:, self.min_samples]
        return core_dists

    def _prim_mst(self, X, core_dists):
        """MST via Prim's on the mutual reachability graph."""
        n = X.shape[0]
        in_tree = np.zeros(n, dtype=bool)
        min_cost = np.full(n, np.inf)
        min_from = np.full(n, -1, dtype=int)
        edges = []

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
        """Extract clusters via condensed tree with stability-based selection."""
        edges_sorted = sorted(edges, key=lambda e: e[0])
        mcs = self.min_cluster_size

        uf_parent = np.arange(n)
        uf_size = np.ones(n, dtype=int)

        def uf_find(x):
            while uf_parent[x] != x:
                uf_parent[x] = uf_parent[uf_parent[x]]
                x = uf_parent[x]
            return x

        next_cluster_id = n
        cluster_birth = {}
        cluster_children = {}
        cluster_stability = {}
        cluster_points = {}
        root_to_cluster = {}

        for weight, u, v in edges_sorted:
            ru, rv = uf_find(u), uf_find(v)
            if ru == rv:
                continue

            lam = 1.0 / weight if weight > 0 else float("inf")
            sz_u, sz_v = uf_size[ru], uf_size[rv]

            if sz_u < sz_v:
                ru, rv = rv, ru
                sz_u, sz_v = sz_v, sz_u

            cl_u = root_to_cluster.get(ru)
            cl_v = root_to_cluster.get(rv)

            if sz_u >= mcs and sz_v >= mcs:
                new_cl = next_cluster_id
                next_cluster_id += 1
                cluster_birth[new_cl] = lam
                cluster_children[new_cl] = []
                cluster_stability[new_cl] = 0.0

                for cl_ref, component_root in [(cl_u, ru), (cl_v, rv)]:
                    if cl_ref is not None:
                        cluster_children[new_cl].append(cl_ref)
                    else:
                        leaf = next_cluster_id
                        next_cluster_id += 1
                        cluster_birth[leaf] = lam
                        cluster_children[leaf] = []
                        cluster_stability[leaf] = 0.0
                        cluster_points[leaf] = set()
                        cluster_children[new_cl].append(leaf)

                uf_parent[rv] = ru
                uf_size[ru] += uf_size[rv]
                root_to_cluster[ru] = new_cl
            else:
                if cl_u is not None:
                    cluster_stability[cl_u] += sz_v * (lam - cluster_birth.get(cl_u, lam))

                uf_parent[rv] = ru
                uf_size[ru] += uf_size[rv]
                if cl_u is not None:
                    root_to_cluster[ru] = cl_u
                elif cl_v is not None:
                    root_to_cluster[ru] = cl_v

        all_clusters = [c for c in cluster_birth if c >= n]

        # Bottom-up stability selection
        selected = set()
        propagated_stability = dict(cluster_stability)

        for cl in all_clusters:
            child_clusters = [c for c in cluster_children.get(cl, []) if c in cluster_birth and c >= n]
            if not child_clusters:
                selected.add(cl)

        internal_sorted = sorted(
            [cl for cl in all_clusters if cluster_children.get(cl, [])],
            key=lambda c: -cluster_birth.get(c, 0),
        )

        for cl in internal_sorted:
            child_clusters = [c for c in cluster_children.get(cl, []) if c in cluster_birth and c >= n]
            if not child_clusters:
                continue

            child_stab_sum = sum(propagated_stability.get(c, 0) for c in child_clusters)
            own_stab = cluster_stability.get(cl, 0)

            if own_stab >= child_stab_sum:
                for c in child_clusters:
                    selected.discard(c)
                selected.add(cl)
                propagated_stability[cl] = own_stab
            else:
                propagated_stability[cl] = child_stab_sum

        # Assign points to selected clusters
        if not selected:
            self._fallback_largest_gap(edges_sorted, n, mcs)
            return

        self._assign_by_cut(edges_sorted, n, selected, cluster_birth, mcs)

    def _fallback_largest_gap(self, edges_sorted, n, mcs):
        weights = [e[0] for e in edges_sorted]
        cut_dist = weights[np.argmax(np.diff(weights))] if len(weights) > 1 else 0

        uf = np.arange(n)
        uf_sz = np.ones(n, dtype=int)

        def find(x):
            while uf[x] != x:
                uf[x] = uf[uf[x]]
                x = uf[x]
            return x

        for w, a, b in edges_sorted:
            if w > cut_dist:
                break
            ra, rb = find(a), find(b)
            if ra != rb:
                if uf_sz[ra] < uf_sz[rb]:
                    ra, rb = rb, ra
                uf[rb] = ra
                uf_sz[ra] += uf_sz[rb]

        raw = np.array([find(i) for i in range(n)])
        lbl_map, cid = {}, 0
        for root in np.unique(raw):
            lbl_map[root] = cid if np.sum(raw == root) >= mcs else -1
            if lbl_map[root] >= 0:
                cid += 1
        self.labels = np.array([lbl_map[r] for r in raw])
        self.n_clusters = cid

    def _assign_by_cut(self, edges_sorted, n, selected, cluster_birth, mcs):
        uf = np.arange(n)
        uf_sz = np.ones(n, dtype=int)

        def find(x):
            while uf[x] != x:
                uf[x] = uf[uf[x]]
                x = uf[x]
            return x

        min_birth_lambda = min(cluster_birth.get(c, 0) for c in selected)
        cut_distance = 1.0 / min_birth_lambda if min_birth_lambda > 0 else float("inf")

        for w, a, b in edges_sorted:
            if w > cut_distance:
                break
            ra, rb = find(a), find(b)
            if ra != rb:
                if uf_sz[ra] < uf_sz[rb]:
                    ra, rb = rb, ra
                uf[rb] = ra
                uf_sz[ra] += uf_sz[rb]

        comp_labels = np.array([find(i) for i in range(n)])
        lbl_map, cid = {}, 0
        for root in np.unique(comp_labels):
            lbl_map[root] = cid if np.sum(comp_labels == root) >= mcs else -1
            if lbl_map[root] >= 0:
                cid += 1

        self.labels = np.array([lbl_map[r] for r in comp_labels])
        self.n_clusters = cid

    def fit(self, X):
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
