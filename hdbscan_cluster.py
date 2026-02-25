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

        tree = []
        uf_parent = {i: i for i in range(n)}
        
        def find(x):
            while uf_parent[x] != x:
                uf_parent[x] = uf_parent[uf_parent[x]]
                x = uf_parent[x]
            return x

        cluster_size = {i: 1 for i in range(n)}
        node_id_map = {i: i for i in range(n)}
        current_cluster = n

        # 1. Build Single Linkage Dendrogram
        for w, a, b in edges_sorted:
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            
            id_a, id_b = node_id_map[ra], node_id_map[rb]
            sz_a, sz_b = cluster_size[id_a], cluster_size[id_b]
            new_sz = sz_a + sz_b
            
            tree.append({
                'id': current_cluster,
                'left': id_a,
                'right': id_b,
                'dist': w,
                'size': new_sz
            })
            
            cluster_size[current_cluster] = new_sz
            uf_parent[rb] = ra
            node_id_map[ra] = current_cluster
            current_cluster += 1

        children_dict = {n_node['id']: (n_node['left'], n_node['right'], n_node['dist']) for n_node in tree}
        size_dict = {n_node['id']: n_node['size'] for n_node in tree}
        for i in range(n): 
            size_dict[i] = 1

        def get_leaves(n_id):
            stack = [n_id]
            leaves = []
            while stack:
                curr = stack.pop()
                if curr < n:
                    leaves.append(curr)
                else:
                    left, right, _ = children_dict[curr]
                    stack.append(left)
                    stack.append(right)
            return leaves

        # 2. Build Condensed Tree
        root_node_id = 2 * n - 2
        cluster_info = {0: {'birth': 0.0, 'points': [], 'children': []}}
        next_cluster_id = 1
        node_to_cluster = {root_node_id: 0}

        stack = [root_node_id]
        while stack:
            node = stack.pop()
            if node < n: 
                continue
            
            left, right, dist = children_dict[node]
            lam = 1.0 / dist if dist > 0 else float("inf")
            
            sz_left, sz_right = size_dict[left], size_dict[right]
            cl_id = node_to_cluster[node]
            
            if sz_left >= mcs and sz_right >= mcs:
                cl_left, cl_right = next_cluster_id, next_cluster_id + 1
                next_cluster_id += 2
                
                cluster_info[cl_id]['children'].extend([cl_left, cl_right])
                cluster_info[cl_id]['death'] = lam
                
                cluster_info[cl_left] = {'birth': lam, 'points': [], 'children': []}
                cluster_info[cl_right] = {'birth': lam, 'points': [], 'children': []}
                
                node_to_cluster[left] = cl_left
                node_to_cluster[right] = cl_right
                
                stack.extend([left, right])
                
            elif sz_left >= mcs and sz_right < mcs:
                node_to_cluster[left] = cl_id
                for p in get_leaves(right):
                    cluster_info[cl_id]['points'].append((p, lam))
                stack.append(left)
                
            elif sz_right >= mcs and sz_left < mcs:
                node_to_cluster[right] = cl_id
                for p in get_leaves(left):
                    cluster_info[cl_id]['points'].append((p, lam))
                stack.append(right)
                
            else:
                cluster_info[cl_id]['death'] = lam
                for p in get_leaves(left):
                    cluster_info[cl_id]['points'].append((p, lam))
                for p in get_leaves(right):
                    cluster_info[cl_id]['points'].append((p, lam))

        # 3. Compute Stability
        def compute_stability(c):
            stab = 0.0
            for p, lam in cluster_info[c]['points']:
                stab += (lam - cluster_info[c]['birth'])
            if 'death' in cluster_info[c]:
                death_lam = cluster_info[c]['death']
                size_at_death = 0
                child_stack = list(cluster_info[c]['children'])
                while child_stack:
                    curr = child_stack.pop()
                    size_at_death += len(cluster_info[curr]['points'])
                    child_stack.extend(cluster_info[curr]['children'])
                stab += size_at_death * (death_lam - cluster_info[c]['birth'])
            return stab

        clusters = list(cluster_info.keys())
        clusters.sort(key=lambda c: cluster_info[c]['birth'], reverse=True)
        
        S = {c: compute_stability(c) for c in clusters}
        S_prop = {}
        is_selected = {c: False for c in clusters}

        for c in clusters:
            child_stab_sum = sum(S_prop[child] for child in cluster_info[c]['children'])
            
            # allow_single_cluster = False equivalent: don't let root be selected
            if c == 0:
                S_prop[c] = child_stab_sum
                is_selected[c] = False
                continue
                
            if S[c] >= child_stab_sum:
                S_prop[c] = S[c]
                is_selected[c] = True
            else:
                S_prop[c] = child_stab_sum
                is_selected[c] = False

        # 4. Top-Down Final Pass
        final_clusters = []
        stack = [0]
        while stack:
            c = stack.pop()
            if is_selected[c]:
                final_clusters.append(c)
            else:
                stack.extend(cluster_info[c]['children'])

        # 5. Assign exactly which points are in which cluster
        labels = np.full(n, -1, dtype=int)
        cid = 0
        for c in final_clusters:
            pts = []
            subtree = [c]
            while subtree:
                curr = subtree.pop()
                pts.extend(p for p, _ in cluster_info[curr]['points'])
                subtree.extend(cluster_info[curr]['children'])
            
            labels[pts] = cid
            cid += 1
        
        self.labels = labels
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
