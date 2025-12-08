"""
Query and retrieval logic for the Geometric Mnemic Manifold.

Implements the hierarchical search algorithm with foveal access
and layer-based drill-down as described in the paper.
"""

import time
import numpy as np
from typing import List, Tuple, Optional, Set, Dict
from ..core.engram import Engram


class ManifoldRetriever:
    """
    Handles querying and retrieval from the hierarchical engram structure.

    Implements the multi-layer search algorithm:
    1. Broadcast to Layer 2 (abstract concepts)
    2. Simultaneously check foveal ring (working memory)
    3. Drill down through hierarchy to Layer 0
    4. Return top-k results by similarity
    """

    def __init__(self):
        """
        Initialize the retriever.
        """
        self.query_stats = {
            'queries_executed': 0,
            'average_query_time': 0.0,
            'total_query_time': 0.0
        }

    def query(self,
              query_embedding: np.ndarray,
              layer0: List[Engram],
              layer1: List[Engram],
              layer2: List[Engram],
              matrices: Dict[int, Optional[np.ndarray]] = None,
              k: int = 5,
              layer_preference: Optional[int] = None,
              fovea: List[Engram] = None) -> List[Tuple[Engram, float]]:
        """
        Query the manifold for relevant engrams.

        Implements the hierarchical search algorithm from the paper.

        Args:
            query_embedding: Query vector
            layer0: Raw engram layer
            layer1: Pattern engram layer
            layer2: Abstract engram layer
            matrices: Cached embedding matrices (Optional)
            k: Number of results to return
            layer_preference: Prefer specific layer (None = auto)
            fovea: Explicit list of foveal engrams to check (Optional)

        Returns:
            List of (engram, similarity_score) tuples sorted by relevance
        """
        start_time = time.time()
        candidates = []
        
        # Helper to get matrix if available
        def get_matrix(layer_idx):
            if matrices and layer_idx in matrices:
                return matrices[layer_idx]
            return None

        # Step 1: Broadcast to Layer 2 (if exists)
        if len(layer2) > 0 and layer_preference != 0:
            # Search abstract layer first
            layer2_results = self._search_layer(layer2, query_embedding, k=20, matrix=get_matrix(2))

            # Drill down to Layer 1
            for abstract, score in layer2_results:
                children_ids = abstract.metadata.get('children', [])
                if not children_ids: continue
                
                # Optimized slice if matrix available
                matrix_l1 = get_matrix(1)
                if matrix_l1 is not None:
                     # Map children IDs to array indices (assuming sorted implementation where ID == Index)
                     # In GMM implementation, ID corresponds to list index.
                     # We can use numpy fancy indexing.
                     indices = [e_id for e_id in children_ids if e_id < len(layer1)]
                     if indices:
                         sub_matrix = matrix_l1[indices]
                         sub_layer = [layer1[i] for i in indices]
                         candidates.extend(
                             self._search_layer(sub_layer, query_embedding, k=20, matrix=sub_matrix)
                         )
                else:
                    layer1_subset = [e for e in layer1 if e.id in children_ids]
                    candidates.extend(
                        self._search_layer(layer1_subset, query_embedding, k=20)
                    )

        elif len(layer1) > 0 and layer_preference != 0:
            # Search pattern layer directly
            layer1_results = self._search_layer(layer1, query_embedding, k=20, matrix=get_matrix(1))
            candidates.extend(layer1_results)

        # Step 2: Simultaneous foveal check (Distributed Fovea)
        if fovea:
             # We rely on the caller to provide the cached matrix for these indices if possible
             # For now, just search the list (batch size is small, ~log N)
             foveal_results = self._search_layer(fovea, query_embedding, k=k)
             candidates.extend(foveal_results)

        # Step 3: Drill down to Layer 0 for final results
        final_results = self._drill_down_to_layer0(
            candidates,
            layer0,
            query_embedding,
            matrix=get_matrix(0)
        )

        # Sort by score and return top-k
        final_results = sorted(
            final_results,
            key=lambda x: x[1],
            reverse=True
        )[:k]

        # Update statistics
        self._update_query_stats(start_time)

        return final_results

    def get_foveal_ring(self, layer0: List[Engram]) -> List[Engram]:
        """
        Get the foveal ring (most recent N engrams = working memory).

        Args:
            layer0: Raw engram layer

        Returns:
            Last N engrams representing working memory
        """
        return self._get_foveal_ring(layer0)

    def get_statistics(self) -> dict:
        """Get retrieval statistics."""
        return self.query_stats.copy()

    def reset_statistics(self):
        """Reset query statistics."""
        self.query_stats = {
            'queries_executed': 0,
            'average_query_time': 0.0,
            'total_query_time': 0.0
        }

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _search_layer(self,
                      layer: List[Engram],
                      query_embedding: np.ndarray,
                      k: int,
                      matrix: Optional[np.ndarray] = None) -> List[Tuple[Engram, float]]:
        """
        Search a specific layer for similar engrams.

        Args:
            layer: List of engrams to search
            query_embedding: Query vector
            k: Number of top results to return
            matrix: Pre-computed embedding matrix (N, D) - Optional

        Returns:
            List of (engram, similarity_score) tuples
        """
        if not layer:
            return []

        # Vectorized cosine similarity
        # 1. Stack all embeddings into a matrix (N x D) IF NOT PROVIDED
        if matrix is not None:
             embeddings = matrix
        else:
             embeddings = np.array([e.embedding for e in layer])
        
        # 2. Compute dot products (N,)
        # Note: Embeddings are assumed to be normalized. If not, we'd need:
        # scores = np.dot(embeddings, query_embedding) / (norm(embeddings) * norm(query))
        # But GMM embeddings are typically normalized on creation.
        # Adding safety normalization just in case:
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Safe normalization for batch
        e_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings_norm = embeddings / e_norms
        
        scores = np.dot(embeddings_norm, q_norm)
        
        # 3. Pair with engrams and sort
        # We can use argpartition for faster top-k if k << N
        if k < len(layer):
            top_k_indices = np.argpartition(scores, -k)[-k:]
            # Initial filter, then sort exactly
            top_candidates = [(layer[i], scores[i]) for i in top_k_indices]
            return sorted(top_candidates, key=lambda x: x[1], reverse=True)
        else:
            # Sort all
            results = [(layer[i], scores[i]) for i in range(len(layer))]
            return sorted(results, key=lambda x: x[1], reverse=True)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Similarity score in [0, 1] range
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    def _get_foveal_ring(self, layer0: List[Engram]) -> List[Engram]:
        """Get the foveal ring (most recent N engrams)."""
        if len(layer0) >= self.foveal_ring_size:
            return layer0[-self.foveal_ring_size:]
        return layer0

    def _drill_down_to_layer0(self,
                              candidates: List[Tuple[Engram, float]],
                              layer0: List[Engram],
                              query_embedding: np.ndarray,
                              matrix: Optional[np.ndarray] = None) -> List[Tuple[Engram, float]]:
        """
        Drill down from higher layers to Layer 0 engrams.

        Args:
            candidates: Higher-layer candidate engrams
            layer0: Raw engram layer
            query_embedding: Query vector
            matrix: Pre-computed Layer 0 matrix (Optional)

        Returns:
            List of Layer 0 engrams with scores
        """
        final_results = []
        seen_ids: Set[int] = set()

        # Vectorized drill-down
        final_results = []
        children_to_check = set()
        
        # 1. Collect all valid children IDs
        for engram, score in candidates:
            if engram.layer == 0:
                if engram.id not in seen_ids:
                    final_results.append((engram, score))
                    seen_ids.add(engram.id)
            else:
                children_ids = engram.metadata.get('children', [])
                for child_id in children_ids:
                    if child_id < len(layer0) and child_id not in seen_ids:
                        children_to_check.add(child_id)
                        seen_ids.add(child_id)
                        
        # 2. Batch score calculation for all children
        if children_to_check:
            child_indices = sorted(list(children_to_check))
            
            # Map indices to engram objects
            batch_engrams = [layer0[i] for i in child_indices]
            
            # Slice matrix if available
            sub_matrix = None
            if matrix is not None and len(matrix) >= len(layer0): 
                 # Assuming matrix index aligns with list index (strictly true for this impl)
                 sub_matrix = matrix[child_indices]

            # Use vectorized search on the subset of children
            # k=len(batch_engrams) to return ALL valid children scores
            child_results = self._search_layer(
                batch_engrams, 
                query_embedding, 
                k=len(batch_engrams), 
                matrix=sub_matrix
            )
            
            final_results.extend(child_results)

        return final_results

    def _update_query_stats(self, start_time: float):
        """Update query statistics after a query."""
        query_time = time.time() - start_time
        self.query_stats['queries_executed'] += 1
        self.query_stats['total_query_time'] += query_time
        self.query_stats['average_query_time'] = (
            self.query_stats['total_query_time'] /
            self.query_stats['queries_executed']
        )
