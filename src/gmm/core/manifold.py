"""
Main Geometric Mnemic Manifold orchestrator.

Coordinates all subsystems: geometry, hierarchy, storage, and retrieval.
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

from .engram import Engram
from ..geometry.spiral import KroneckerSpiral
from ..hierarchy.compressor import HierarchicalCompressor
from ..storage.serializer import EngramSerializer
from ..query.retriever import ManifoldRetriever


class GeometricMnemicManifold:
    """
    Main GMM system implementing the full architecture.

    Orchestrates:
    - Geometric positioning via Kronecker sequences
    - Hierarchical compression (3-layer skip-list)
    - Persistent storage
    - Foveated retrieval

    This class follows the Facade pattern, providing a simple interface
    to the complex subsystems while delegating specific responsibilities
    to specialized components.
    """

    def __init__(self,
                 embedding_dim: int = 384,
                 lambda_decay: float = 0.01,
                 beta1: int = 64,
                 beta2: int = 16,
                 storage_path: Optional[Path] = None,
                 foveal_ring_size: int = 10):
        """
        Initialize the Geometric Mnemic Manifold.

        Args:
            embedding_dim: Dimensionality of embeddings
            lambda_decay: Exponential decay for spiral expansion
            beta1: Layer 1 compression ratio (patterns)
            beta2: Layer 2 compression ratio (abstracts)
            storage_path: Path to persist engrams
            foveal_ring_size: Size of working memory ring
        """
        # Configuration
        self.embedding_dim = embedding_dim
        self.beta1 = beta1
        self.beta2 = beta2

        # Subsystems (Dependency Injection)
        self.spiral = KroneckerSpiral(dimensions=2, lambda_decay=lambda_decay)
        self.compressor = HierarchicalCompressor(beta1=beta1, beta2=beta2)
        self.serializer = EngramSerializer(storage_path=storage_path)
        self.retriever = ManifoldRetriever(foveal_ring_size=foveal_ring_size)

        # Engram layers (in-memory state)
        self.layer0: List[Engram] = []  # Raw memories
        self.layer1: List[Engram] = []  # Patterns
        self.layer2: List[Engram] = []  # Abstracts

        # Spatial index for visualization
        self.positions: Dict[int, Any] = {}

        # Global statistics
        self.stats = {
            'total_engrams': 0,
            'rebuild_count': 0
        }

    def add_engram(self,
                   context_window: str,
                   embedding: np.ndarray,
                   metadata: Optional[Dict] = None) -> Engram:
        """
        Add a new engram to the manifold.

        This is the primary write operation. Creates an engram,
        positions it on the spiral, persists it, and triggers
        hierarchical rebuilding if needed.

        Args:
            context_window: Text/token content
            embedding: Vector embedding
            metadata: Additional metadata

        Returns:
            Created engram
        """
        engram_id = len(self.layer0)

        # Create engram
        engram = Engram(
            id=engram_id,
            timestamp=time.time(),
            context_window=context_window,
            embedding=embedding,
            metadata=metadata or {},
            layer=0
        )

        # Add to Layer 0
        self.layer0.append(engram)

        # Calculate geometric position
        position = self.spiral.position(engram_id)
        self.positions[engram_id] = position

        # Persist to disk
        self.serializer.save(engram)

        # Rebuild hierarchy if needed (every beta1 engrams)
        if len(self.layer0) % self.beta1 == 0:
            self._rebuild_hierarchy()

        # Update statistics
        self.stats['total_engrams'] += 1

        return engram

    def query(self,
              query_embedding: np.ndarray,
              k: int = 5,
              layer_preference: Optional[int] = None) -> List[Tuple[Engram, float]]:
        """
        Query the manifold for relevant engrams.

        Delegates to the ManifoldRetriever which implements the
        hierarchical search algorithm.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            layer_preference: Prefer specific layer (None = auto)

        Returns:
            List of (engram, similarity_score) tuples
        """
        return self.retriever.query(
            query_embedding=query_embedding,
            layer0=self.layer0,
            layer1=self.layer1,
            layer2=self.layer2,
            k=k,
            layer_preference=layer_preference
        )

    def get_foveal_ring(self) -> List[Engram]:
        """
        Get the foveal ring (most recent engrams = working memory).

        Returns:
            List of most recent engrams
        """
        return self.retriever.get_foveal_ring(self.layer0)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.

        Returns:
            Dictionary with layer sizes, compression ratios, and performance metrics
        """
        retriever_stats = self.retriever.get_statistics()

        return {
            **self.stats,
            'layer0_size': len(self.layer0),
            'layer1_size': len(self.layer1),
            'layer2_size': len(self.layer2),
            'compression_ratio': len(self.layer0) / max(len(self.layer1), 1),
            'storage_size_mb': self.serializer.get_storage_size(),
            'queries_executed': retriever_stats['queries_executed'],
            'average_query_time': retriever_stats['average_query_time']
        }

    def visualize_manifold(self) -> Dict[str, Any]:
        """
        Generate visualization data for the manifold.

        Produces JSON-serializable data for web-based visualization
        of the spiral geometry and hierarchical structure.

        Returns:
            Dictionary suitable for JSON serialization
        """
        viz_data = {
            'spiral_positions': [],
            'layers': {
                0: [],
                1: [],
                2: []
            },
            'stats': self.get_statistics()
        }

        # Spiral positions
        for engram_id, position in self.positions.items():
            viz_data['spiral_positions'].append({
                'id': engram_id,
                'x': position.x,
                'y': position.y,
                'radius': position.radius,
                'theta': position.theta,
                'layer': position.layer
            })

        # Layer contents
        for layer_idx, layer in enumerate([self.layer0, self.layer1, self.layer2]):
            for engram in layer:
                viz_data['layers'][layer_idx].append({
                    'id': engram.id,
                    'timestamp': engram.timestamp,
                    'context_preview': engram.context_window[:100],
                    'metadata': engram.metadata
                })

        return viz_data

    def save_visualization(self, filepath: Path):
        """
        Save visualization data to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        viz_data = self.visualize_manifold()
        with open(filepath, 'w') as f:
            json.dump(viz_data, f, indent=2, default=str)

    def clear_storage(self) -> int:
        """
        Clear all persisted engrams.

        Returns:
            Number of files deleted
        """
        return self.serializer.clear_storage()

    def load_engram(self, engram_id: int, layer: int = 0) -> Optional[Engram]:
        """
        Load a specific engram from disk.

        Args:
            engram_id: ID of the engram
            layer: Layer of the engram (0, 1, or 2)

        Returns:
            Loaded Engram or None if not found
        """
        return self.serializer.load(engram_id, layer)

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _rebuild_hierarchy(self):
        """
        Rebuild the hierarchical layers.

        Compresses Layer 0 into Layer 1, and Layer 1 into Layer 2
        when sufficient engrams exist.
        """
        # Compress Layer 0 -> Layer 1
        self.layer1 = self.compressor.compress_to_layer1(self.layer0)

        # Compress Layer 1 -> Layer 2 (if enough patterns exist)
        if len(self.layer1) >= self.compressor.beta2:
            self.layer2 = self.compressor.compress_to_layer2(self.layer1)

        self.stats['rebuild_count'] += 1
