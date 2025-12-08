"""
Pure Python implementation of HNSW (Hierarchical Navigable Small World).

This serves as a robust algo-to-algo baseline for benchmarking GMM.
It implements the HNSW graph construction and search algorithms as described
in Malkov & Yashunin (2018), without relying on C++ bindings, to provide
a fair "Python overhead" comparison with the Python GMM prototype.
"""

import numpy as np
import random
import heapq
from typing import List, Dict, Tuple, Optional

class HNSWIndex:
    def __init__(self, embedding_dim: int, m: int = 16, ef_construction: int = 200):
        """
        Initialize HNSW index.
        
        Args:
            embedding_dim: Dimension of vectors
            m: Max number of connections per element
            ef_construction: Size of dynamic candidate list during construction
        """
        self.d = embedding_dim
        self.m = m
        self.m_max = m
        self.m_max0 = m * 2
        self.ef_construction = ef_construction
        self.ef_search = 50
        
        # Scaling factor for level generation
        self.ml = 1.0 / np.log(m)
        
        # Data storage
        self.data: Dict[int, np.ndarray] = {}
        self.graphs: List[Dict[int, List[int]]] = []
        self.entry_point: Optional[int] = None
        self.max_level = -1
        
        # Stats
        self.comparisons = 0

    def add_item(self, vector: np.ndarray, idx: int):
        """Add item to the index."""
        # Normalize
        vector = vector / (np.linalg.norm(vector) + 1e-8)
        self.data[idx] = vector
        
        level = int(-np.log(random.random()) * self.ml)
        
        # Initialize graphs for new levels
        while len(self.graphs) <= level:
            self.graphs.append({})
            
        if self.entry_point is None:
            self.entry_point = idx
            self.max_level = level
            for l in range(level + 1):
                self.graphs[l][idx] = []
            return

        # Search from top to insertion level
        curr_obj = self.entry_point
        for l in range(self.max_level, level, -1):
            results, _ = self._search_layer(curr_obj, vector, ef=1, layer=l)
            if results:
                curr_obj = results[0][0]
            else:
                 break # Should not happen if graph connected
            
        # Insert from level down to 0
        for l in range(min(level, self.max_level), -1, -1):
            # perform search to find neighbors
            neighbors, _ = self._search_layer(curr_obj, vector, ef=self.ef_construction, layer=l)
            
            # Select simpler heuristics for neighbors (basic top-m)
            # In full HNSW implementation, we'd use robust pruning here.
            # For this baseline, simple closest neighbors is sufficient.
            selected_neighbors = [n for n, dist in neighbors[:self.m]]
            
            self.graphs[l][idx] = selected_neighbors
            
            # Add reverse connections
            for neighbor in selected_neighbors:
                if neighbor not in self.graphs[l]:
                    self.graphs[l][neighbor] = []
                self.graphs[l][neighbor].append(idx)
                
                # Prune if too many connections
                max_conn = self.m_max0 if l == 0 else self.m_max
                if len(self.graphs[l][neighbor]) > max_conn:
                    # Keep closest
                    nb_vectors = [self.data[n] for n in self.graphs[l][neighbor]]
                    nb_dists = [self._dist(self.data[neighbor], v) for v in nb_vectors]
                    sorted_nb = sorted(zip(self.graphs[l][neighbor], nb_dists), key=lambda x: x[1])
                    self.graphs[l][neighbor] = [n for n, d in sorted_nb[:max_conn]]
            
            curr_obj = self.entry_point # Reset for next search? No, HNSW continues from closest found.
            # Actually for standard insert: entry point for next layer is the closest found in this layer.
            if neighbors:
                 curr_obj = neighbors[0][0]

        if level > self.max_level:
            self.max_level = level
            self.entry_point = idx

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """Search for k nearest neighbors."""
        # Normalize
        query = query / (np.linalg.norm(query) + 1e-8)
        
        curr_obj = self.entry_point
        if curr_obj is None:
            return []
            
        # 1. Search from top to level 1
        for l in range(self.max_level, 0, -1):
            results, _ = self._search_layer(curr_obj, query, ef=1, layer=l)
            if results:
                curr_obj = results[0][0]
            
        # 2. Search level 0 with ef_search
        neighbors, _ = self._search_layer(curr_obj, query, ef=self.ef_search, layer=0)
        
        # 3. Sort and return top-k
        return neighbors[:k]

    def _search_layer(self, entry_point: int, query: np.ndarray, ef: int, layer: int):
        """Greedy search within a layer."""
        candidates = [(self._dist(self.data[entry_point], query), entry_point)]
        heapq.heapify(candidates)
        
        visited = {entry_point}
        results = [(self._dist(self.data[entry_point], query), entry_point)] # Max heap for results? 
        # Actually HNSW paper uses dynamic candidate list.
        # Simplified:
        
        # Priority queue for candidates (min-distance first)
        candidates_q = [(self._dist(self.data[entry_point], query), entry_point)]
        
        # Set for results found so far
        top_results = [(self._dist(self.data[entry_point], query), entry_point)] 
        
        while candidates_q:
            dist_c, c = heapq.heappop(candidates_q)
            
            # If closest candidate is further than furthest result (and we have enough results), stop
            # Note: results list is usually kept sorted or max-heap.
            # Simplified Logic for Python limitation speed:
            
            furthest_res_dist = max(r[0] for r in top_results)
            if dist_c > furthest_res_dist and len(top_results) >= ef:
                break
                
            # Explore neighbors
            # Explore neighbors
            if c in self.graphs[layer]:
                neighbors = self.graphs[layer][c]
                unvisited = [n for n in neighbors if n not in visited]
                
                if unvisited:
                    # Batch distance calculation
                    # 1. Gather vectors
                    # Note: We can't pre-stack whole matrix efficiently because HNSW structure is dynamic/graph-based
                    # But we can stack small batches of neighbors
                    nb_vectors = np.array([self.data[n] for n in unvisited])
                    
                    # 2. Compute distances (cosine distance = 1 - dot)
                    # Query is (D,), nb_vectors is (M, D)
                    # dot is (M,)
                    dots = np.dot(nb_vectors, query)
                    dists = 1.0 - dots
                    
                    for i, neighbor in enumerate(unvisited):
                        visited.add(neighbor)
                        dist_n = dists[i]
                        
                        if dist_n < furthest_res_dist or len(top_results) < ef:
                            heapq.heappush(candidates_q, (dist_n, neighbor))
                            top_results.append((dist_n, neighbor))
                            
                    # Sort and prune (batch style)
                    top_results.sort(key=lambda x: x[0]) 
                    while len(top_results) > ef:
                        top_results.pop()
                        
                    if top_results:
                        furthest_res_dist = top_results[-1][0]
                                
        # Return sorted by distance (closest first), convert distance to similarity if needed
        # dist is cosine distance (1 - similarity).
        results_formatted = []
        for d, idx in top_results:
             # Convert cosine dist back to sim (approx)
             sim = 1.0 - d
             results_formatted.append((idx, sim))
             
        return results_formatted, None

    def _dist(self, v1, v2):
        """Cosine distance (0 to 2)."""
        # Assumes normalized
        return 1.0 - np.dot(v1, v2)
