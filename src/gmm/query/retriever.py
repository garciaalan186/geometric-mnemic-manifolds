"""
Query and retrieval logic for the Geometric Mnemic Manifold.

Implements the hierarchical search algorithm with foveal access
and layer-based drill-down as described in the paper.
"""

import time
import numpy as np
from typing import List, Tuple, Optional, Set
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

    def __init__(self, foveal_ring_size: int = 10):
        """
        Initialize the retriever.

        Args:
            foveal_ring_size: Size of the foveal ring (working memory)
        """
        self.foveal_ring_size = foveal_ring_size
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
              k: int = 5,
              layer_preference: Optional[int] = None) -> List[Tuple[Engram, float]]:
        """
        Query the manifold for relevant engrams.

        Implements the hierarchical search algorithm from the paper.

        Args:
            query_embedding: Query vector
            layer0: Raw engram layer
            layer1: Pattern engram layer
            layer2: Abstract engram layer
            k: Number of results to return
            layer_preference: Prefer specific layer (None = auto)

        Returns:
            List of (engram, similarity_score) tuples sorted by relevance
        """
        start_time = time.time()
        candidates = []

        # Step 1: Broadcast to Layer 2 (if exists)
        if len(layer2) > 0 and layer_preference != 0:
            # Search abstract layer first
            layer2_results = self._search_layer(layer2, query_embedding, k=3)

            # Drill down to Layer 1
            for abstract, score in layer2_results:
                children_ids = abstract.metadata.get('children', [])
                layer1_subset = [e for e in layer1 if e.id in children_ids]
                candidates.extend(
                    self._search_layer(layer1_subset, query_embedding, k=2)
                )

        elif len(layer1) > 0 and layer_preference != 0:
            # Search pattern layer directly
            layer1_results = self._search_layer(layer1, query_embedding, k=5)
            candidates.extend(layer1_results)

        # Step 2: Simultaneous foveal check (last N engrams = working memory)
        fovea = self._get_foveal_ring(layer0)
        foveal_results = self._search_layer(fovea, query_embedding, k=k)
        candidates.extend(foveal_results)

        # Step 3: Drill down to Layer 0 for final results
        final_results = self._drill_down_to_layer0(
            candidates,
            layer0,
            query_embedding
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
                      k: int) -> List[Tuple[Engram, float]]:
        """
        Search a specific layer for similar engrams.

        Args:
            layer: List of engrams to search
            query_embedding: Query vector
            k: Number of top results to return

        Returns:
            List of (engram, similarity_score) tuples
        """
        results = []

        for engram in layer:
            similarity = self._cosine_similarity(query_embedding, engram.embedding)
            results.append((engram, similarity))

        return sorted(results, key=lambda x: x[1], reverse=True)[:k]

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
                              query_embedding: np.ndarray) -> List[Tuple[Engram, float]]:
        """
        Drill down from higher layers to Layer 0 engrams.

        Args:
            candidates: Higher-layer candidate engrams
            layer0: Raw engram layer
            query_embedding: Query vector

        Returns:
            List of Layer 0 engrams with scores
        """
        final_results = []
        seen_ids: Set[int] = set()

        for engram, score in sorted(candidates, key=lambda x: x[1], reverse=True):
            if engram.layer == 0:
                # Already a Layer 0 engram
                if engram.id not in seen_ids:
                    final_results.append((engram, score))
                    seen_ids.add(engram.id)
            else:
                # Get children from lower layers
                children_ids = engram.metadata.get('children', [])
                for child_id in children_ids:
                    if child_id < len(layer0) and child_id not in seen_ids:
                        child = layer0[child_id]
                        child_score = self._cosine_similarity(
                            query_embedding,
                            child.embedding
                        )
                        final_results.append((child, child_score))
                        seen_ids.add(child_id)

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
