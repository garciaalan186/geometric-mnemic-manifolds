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
                 beta1: int = 32,
                 beta2: int = 8,
                 storage_path: Optional[Path] = None,
                 foveal_beta: float = 1.2):
        """
        Initialize the Geometric Mnemic Manifold.

        Args:
            embedding_dim: Dimensionality of embeddings
            lambda_decay: Exponential decay for spiral expansion
            beta1: Layer 1 compression ratio (patterns)
            beta2: Layer 2 compression ratio (abstracts)
            storage_path: Path to persist engrams
            foveal_beta: Geometric decay factor for distributed fovea (default 1.2)
        """
        # Configuration
        self.embedding_dim = embedding_dim
        self.beta1 = beta1
        self.beta2 = beta2
        self.foveal_beta = foveal_beta

        # Subsystems (Dependency Injection)
        self.spiral = KroneckerSpiral(dimensions=2, lambda_decay=lambda_decay)
        self.compressor = HierarchicalCompressor(beta1=beta1, beta2=beta2)
        self.serializer = EngramSerializer(storage_path=storage_path)
        self.retriever = ManifoldRetriever() # No longer takes ring size

        # State
        self.layer0: List[Engram] = [] # Raw Observations
        self.layer1: List[Engram] = [] # Patterns
        self.layer2: List[Engram] = [] # Abstracts
        
        # Distributed Fovea State
        self.fovea_indices: List[int] = []

        # Spatial index for visualization
        self.positions: Dict[int, Any] = {}

        # Cache for embedding matrices
        # Key: Layer Index (0, 1, 2) -> Value: np.ndarray or None
        self._layer_matrices: Dict[int, Optional[np.ndarray]] = {
            0: None,
            1: None,
            2: None
        }
        self._is_dirty: Dict[int, bool] = {
            0: False,
            1: False,
            2: False
        }

        # Global statistics
        self.stats = {
            'total_engrams': 0,
            'rebuild_count': 0
        }

    def add_engram(self, context_window: str, embedding: np.ndarray, metadata: dict = None) -> Engram:
        """
        Add a new memory engram to the manifold.

        Args:
            context_window: Text content
            embedding: Vector representation
            metadata: Additional info

        Returns:
            The created Engram
        """
        # 1. Create Engram
        engram_id = len(self.layer0)
        # Calculate spiral coordinates
        coords = self.spiral.position(engram_id)
        
        # Store coordinates in visualization index
        self.positions[engram_id] = coords

        # Daisy Chaining: Mix in summary of previous engram
        final_context = context_window
        metadata = metadata or {}
        
        if len(self.layer0) > 0:
            prev_engram = self.layer0[-1]
            seed = self.compressor.generate_seed_context(prev_engram.context_window)
            final_context = f"[PREV: {seed}] {context_window}"
            metadata['previous_id'] = prev_engram.id
            metadata['seed_context'] = seed
        
        engram = Engram(
            id=engram_id,
            timestamp=time.time(),
            layer=0,
            embedding=embedding,
            context_window=final_context,
            metadata=metadata
        )
        
        self.layer0.append(engram)
        self._is_dirty[0] = True
        
        # 2. Rebuild Distributed Fovea
        self._rebuild_distributed_fovea()
        
        # 3. Check for Compression (Hierarchical Growth)
        self._process_compression()
        
        # 4. Auto-persist
        if self.serializer.storage_path:
            self.serializer.save(engram)
            
        # Update statistics
        self.stats['total_engrams'] += 1
            
        return engram

    def query(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Engram, float]]:
        """
        Query the manifold.

        Args:
            query_embedding: Query embedding
            k: Number of results

        Returns:
            Ranked list of results
        """
        # Ensure matrices are up to date
        self._update_matrices()
        
        # Gather foveal engrams
        fovea_list = [self.layer0[i] for i in self.fovea_indices if i < len(self.layer0)]
        
        return self.retriever.query(
            query_embedding=query_embedding,
            layer0=self.layer0,
            layer1=self.layer1,
            layer2=self.layer2,
            matrices=self._layer_matrices,
            k=k,
            fovea=fovea_list
        )

    def _rebuild_distributed_fovea(self):
        """
        Recalculate the distributed fovea indices based on geometric progression.
        Hybrid Mode: Enforces a dense ring for the high-density start of the curve.
        """
        N = len(self.layer0)
        if N == 0:
            self.fovea_indices = []
            return
            
        if N < 10:
             # Just include everything for very small N
             self.fovea_indices = list(range(N))
             return

        beta = self.foveal_beta
        if beta <= 1.0: beta = 1.001 # Safety

        # Calculate M (number of steps)
        M = int(np.log(N) / np.log(beta))
        
        # Scaling factor
        denominator = (beta**M - 1)
        if denominator == 0: denominator = 0.0001
        c = (N - 1) / denominator
        
        indices = set()
        # Always include endpoints
        indices.add(0)
        indices.add(N-1)
        
        # Hybrid Logic: Track density to identify the "elbow"
        last_idx = N - 1
        elbow_found = False
        elbow_idx = N - 1
        
        for k in range(1, M + 1): # Start from 1 as 0 is N-1
            y = c * (beta**k - 1)
            idx = (N - 1) - int(y)
            
            # Clamp
            if idx < 0: idx = 0
            if idx >= N: idx = N - 1
            
            # Detect Elbow (first time gap > 1)
            # We are moving backwards: N-1, N-2, ...
            # If idx is significantly smaller than last_idx, we have a gap.
            if not elbow_found:
                if (last_idx - idx) > 1:
                    elbow_found = True
                    # The elbow entails the end of the dense region.
                    # The dense region ended at last_idx.
                    elbow_idx = last_idx
            
            indices.add(idx)
            last_idx = idx
            
        # Enforce Ring Density "before the elbow"
        # Everything from elbow_idx to N-1 should be included
        if elbow_idx < N - 1:
            indices.update(range(elbow_idx, N))
            
        self.fovea_indices = sorted(list(indices))
        
    def _update_matrices(self):
        """Update cached embedding matrices if dirty."""
        # Layer 0
        if self._is_dirty[0] or (self._layer_matrices[0] is None and self.layer0):
            # Optimization: If only appending, vstack new items?
            # For now, full rebuild is safer and fast enough for prototype batching vs query ratio
            if self.layer0:
                 self._layer_matrices[0] = np.array([e.embedding for e in self.layer0])
            self._is_dirty[0] = False
            
        # Layer 1
        if self._is_dirty[1] or (self._layer_matrices[1] is None and self.layer1):
            if self.layer1:
                self._layer_matrices[1] = np.array([e.embedding for e in self.layer1])
            self._is_dirty[1] = False
            
        # Layer 2
        if self._is_dirty[2] or (self._layer_matrices[2] is None and self.layer2):
            if self.layer2:
                self._layer_matrices[2] = np.array([e.embedding for e in self.layer2])
            self._is_dirty[2] = False

    def get_foveal_ring(self) -> List[Engram]:
        """
        Get the foveal ring (distributed sample of history).

        Returns:
            List of foveal engrams
        """
        return [self.layer0[i] for i in self.fovea_indices if i < len(self.layer0)]

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

    def _process_compression(self):
        """Check if compression is needed and execute."""
        # Rebuild hierarchy if needed (every beta1 engrams)
        if len(self.layer0) > 0 and len(self.layer0) % self.beta1 == 0:
            self._rebuild_hierarchy()
            self._is_dirty[1] = True
            self._is_dirty[2] = True

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
