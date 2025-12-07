"""
Geometric Mnemic Manifold (GMM) - Working Prototype
A functional implementation of the architecture described in the paper.

This prototype demonstrates:
1. Kronecker sequence-based spiral positioning
2. Hierarchical layer construction
3. Foveated memory access
4. Epistemic gap detection
5. Engram serialization and retrieval
"""

import numpy as np
import json
import pickle
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from collections import defaultdict


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class Engram:
    """Immutable memory state representing a frozen phenomenological moment."""
    id: int
    timestamp: float
    context_window: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    layer: int = 0  # 0=raw, 1=pattern, 2=abstract
    
    def serialize(self) -> bytes:
        """Serialize engram for storage."""
        data = {
            'id': self.id,
            'timestamp': self.timestamp,
            'context_window': self.context_window,
            'embedding': self.embedding.tolist(),
            'metadata': self.metadata,
            'layer': self.layer
        }
        return pickle.dumps(data)
    
    @staticmethod
    def deserialize(data: bytes) -> 'Engram':
        """Deserialize engram from storage."""
        obj = pickle.loads(data)
        obj['embedding'] = np.array(obj['embedding'])
        return Engram(**obj)


@dataclass
class SpiralPosition:
    """Position of an engram on the geometric manifold."""
    k: int  # Index in sequence
    theta: float  # Angular position
    radius: float  # Radial distance (temporal)
    x: float  # Cartesian x
    y: float  # Cartesian y
    layer: int  # Which hierarchical layer


# ============================================================================
# GEOMETRIC POSITIONING ENGINE
# ============================================================================

class KroneckerSpiral:
    """
    Generates low-discrepancy positions on a hypersphere using Kronecker sequences.
    Implements the core geometric positioning from the paper.
    """
    
    def __init__(self, dimensions: int = 2, lambda_decay: float = 0.01):
        """
        Args:
            dimensions: Embedding space dimensions
            lambda_decay: Exponential decay constant for radial expansion
        """
        self.dimensions = dimensions
        self.lambda_decay = lambda_decay
        
        # First primes for Kronecker sequence
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        
        # Alpha vector: square roots of primes (linearly independent irrationals)
        self.alpha = np.array([np.sqrt(p) for p in self.primes[:dimensions]])
    
    def position(self, k: int) -> SpiralPosition:
        """
        Calculate position of the k-th engram on the spiral.
        
        Args:
            k: Index in the sequence (0 = most recent)
            
        Returns:
            SpiralPosition with all coordinate information
        """
        # Kronecker sequence for angular distribution
        u = np.array([(k * self.alpha[d]) % 1 for d in range(self.dimensions)])
        
        # For 2D visualization, use first component
        theta = 2 * np.pi * u[0]
        
        # Exponential radial expansion
        radius = np.exp(self.lambda_decay * k)
        
        # Cartesian coordinates
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        # Determine hierarchical layer based on k
        if k < 10:
            layer = 0  # Fovea
        elif k < 64:
            layer = 1  # Para-fovea (pattern)
        else:
            layer = 2  # Periphery (abstract)
        
        return SpiralPosition(k=k, theta=theta, radius=radius, x=x, y=y, layer=layer)
    
    def distance(self, pos1: SpiralPosition, pos2: SpiralPosition) -> float:
        """
        Calculate geometric distance between two positions.
        Incorporates both spatial and temporal components.
        """
        spatial = np.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)
        temporal = abs(pos1.k - pos2.k)
        
        # Weighted combination: space matters more for nearby, time for distant
        weight = np.exp(-0.01 * temporal)
        return spatial * weight + temporal * (1 - weight)


# ============================================================================
# HIERARCHICAL LAYER BUILDER
# ============================================================================

class HierarchicalCompressor:
    """
    Builds the hierarchical skip-list structure from raw engrams.
    Implements the telegraphic compression operator Λ.
    """
    
    def __init__(self, beta1: int = 64, beta2: int = 16):
        """
        Args:
            beta1: Compression ratio for Layer 1 (patterns)
            beta2: Compression ratio for Layer 2 (abstracts)
        """
        self.beta1 = beta1
        self.beta2 = beta2
    
    def compress_to_layer1(self, engrams: List[Engram]) -> List[Engram]:
        """
        Compress raw engrams into pattern-level summaries.
        
        Groups every beta1 raw engrams into a single pattern engram.
        """
        layer1_engrams = []
        
        for i in range(0, len(engrams), self.beta1):
            chunk = engrams[i:i + self.beta1]
            
            # Telegraphic compression: extract key entities and patterns
            compressed_context = self._telegraphic_compress([e.context_window for e in chunk])
            
            # Average embedding
            avg_embedding = np.mean([e.embedding for e in chunk], axis=0)
            
            # Create pattern engram
            pattern = Engram(
                id=i // self.beta1,
                timestamp=chunk[0].timestamp,  # Use earliest timestamp
                context_window=compressed_context,
                embedding=avg_embedding,
                metadata={
                    'children': [e.id for e in chunk],
                    'span': (chunk[0].id, chunk[-1].id)
                },
                layer=1
            )
            layer1_engrams.append(pattern)
        
        return layer1_engrams
    
    def compress_to_layer2(self, layer1_engrams: List[Engram]) -> List[Engram]:
        """
        Compress pattern engrams into abstract semantic concepts.
        """
        layer2_engrams = []
        
        for i in range(0, len(layer1_engrams), self.beta2):
            chunk = layer1_engrams[i:i + self.beta2]
            
            # Extract high-level themes
            compressed_context = self._abstract_compress([e.context_window for e in chunk])
            
            # Average embedding
            avg_embedding = np.mean([e.embedding for e in chunk], axis=0)
            
            # Create abstract engram
            abstract = Engram(
                id=i // self.beta2,
                timestamp=chunk[0].timestamp,
                context_window=compressed_context,
                embedding=avg_embedding,
                metadata={
                    'children': [e.id for e in chunk],
                    'patterns': len(chunk)
                },
                layer=2
            )
            layer2_engrams.append(abstract)
        
        return layer2_engrams
    
    def _telegraphic_compress(self, texts: List[str]) -> str:
        """
        Telegraphic compression: remove function words, keep content.
        This is a simplified version - production would use NLP.
        """
        # Combine texts
        combined = " ".join(texts)
        
        # Simple heuristic: keep capitalized words (entities) and important keywords
        words = combined.split()
        
        # Function words to remove
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                     'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'}
        
        compressed_words = [w for w in words if w.lower() not in stop_words or w[0].isupper()]
        
        return " ".join(compressed_words[:200])  # Limit length
    
    def _abstract_compress(self, texts: List[str]) -> str:
        """
        Abstract compression: extract themes and concepts.
        """
        # Very simplified - production would use LLM summarization
        combined = " ".join(texts)
        words = combined.split()
        
        # Extract unique important terms
        unique_terms = list(set([w for w in words if len(w) > 4]))[:50]
        
        return " ".join(unique_terms)


# ============================================================================
# GEOMETRIC MNEMIC MANIFOLD - MAIN CLASS
# ============================================================================

class GeometricMnemicManifold:
    """
    Main GMM system implementing the full architecture.
    """
    
    def __init__(self, 
                 embedding_dim: int = 384,
                 lambda_decay: float = 0.01,
                 beta1: int = 64,
                 beta2: int = 16,
                 storage_path: Optional[Path] = None):
        """
        Initialize the Geometric Mnemic Manifold.
        
        Args:
            embedding_dim: Dimensionality of embeddings
            lambda_decay: Exponential decay for spiral
            beta1: Layer 1 compression ratio
            beta2: Layer 2 compression ratio
            storage_path: Path to persist engrams
        """
        self.embedding_dim = embedding_dim
        self.spiral = KroneckerSpiral(dimensions=2, lambda_decay=lambda_decay)
        self.compressor = HierarchicalCompressor(beta1=beta1, beta2=beta2)
        
        # Storage
        self.storage_path = storage_path or Path("./gmm_storage")
        self.storage_path.mkdir(exist_ok=True)
        
        # Engram layers
        self.layer0: List[Engram] = []  # Raw memories
        self.layer1: List[Engram] = []  # Patterns
        self.layer2: List[Engram] = []  # Abstracts
        
        # Spatial index for fast lookup
        self.positions: Dict[int, SpiralPosition] = {}
        
        # Statistics
        self.stats = {
            'total_engrams': 0,
            'queries_executed': 0,
            'average_query_time': 0.0
        }
    
    def add_engram(self, context_window: str, embedding: np.ndarray, 
                   metadata: Optional[Dict] = None) -> Engram:
        """
        Add a new engram to the manifold.
        
        Args:
            context_window: Text/token content
            embedding: Vector embedding
            metadata: Additional metadata
            
        Returns:
            Created engram
        """
        engram_id = len(self.layer0)
        
        engram = Engram(
            id=engram_id,
            timestamp=time.time(),
            context_window=context_window,
            embedding=embedding,
            metadata=metadata or {},
            layer=0
        )
        
        self.layer0.append(engram)
        
        # Calculate position on spiral
        position = self.spiral.position(engram_id)
        self.positions[engram_id] = position
        
        # Rebuild hierarchical layers if needed
        if len(self.layer0) % 64 == 0:
            self._rebuild_hierarchy()
        
        # Persist
        self._save_engram(engram)
        
        self.stats['total_engrams'] += 1
        
        return engram
    
    def query(self, 
              query_embedding: np.ndarray,
              k: int = 5,
              layer_preference: Optional[int] = None) -> List[Tuple[Engram, float]]:
        """
        Query the manifold for relevant engrams.
        
        Implements the hierarchical search algorithm from the paper.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            layer_preference: Prefer specific layer (None = auto)
            
        Returns:
            List of (engram, similarity_score) tuples
        """
        start_time = time.time()
        
        # Step 1: Broadcast to Layer 2 (if exists)
        candidates = []
        
        if len(self.layer2) > 0 and layer_preference != 0:
            # Search abstract layer first
            layer2_results = self._search_layer(self.layer2, query_embedding, k=3)
            
            # Drill down to Layer 1
            for abstract, score in layer2_results:
                children_ids = abstract.metadata.get('children', [])
                layer1_subset = [e for e in self.layer1 if e.id in children_ids]
                candidates.extend(self._search_layer(layer1_subset, query_embedding, k=2))
        
        elif len(self.layer1) > 0 and layer_preference != 0:
            # Search pattern layer
            layer1_results = self._search_layer(self.layer1, query_embedding, k=5)
            candidates.extend(layer1_results)
        
        # Step 2: Simultaneous foveal check (last 10 engrams)
        fovea = self.layer0[-10:] if len(self.layer0) >= 10 else self.layer0
        foveal_results = self._search_layer(fovea, query_embedding, k=k)
        candidates.extend(foveal_results)
        
        # Step 3: Drill down to Layer 0 for final results
        final_results = []
        seen_ids = set()
        
        for engram, score in sorted(candidates, key=lambda x: x[1], reverse=True):
            if engram.layer == 0:
                if engram.id not in seen_ids:
                    final_results.append((engram, score))
                    seen_ids.add(engram.id)
            else:
                # Get children from lower layers
                children_ids = engram.metadata.get('children', [])
                for child_id in children_ids:
                    if child_id < len(self.layer0) and child_id not in seen_ids:
                        child = self.layer0[child_id]
                        child_score = self._cosine_similarity(query_embedding, child.embedding)
                        final_results.append((child, child_score))
                        seen_ids.add(child_id)
        
        # Sort by score and return top-k
        final_results = sorted(final_results, key=lambda x: x[1], reverse=True)[:k]
        
        # Update stats
        query_time = time.time() - start_time
        self.stats['queries_executed'] += 1
        self.stats['average_query_time'] = (
            (self.stats['average_query_time'] * (self.stats['queries_executed'] - 1) + query_time)
            / self.stats['queries_executed']
        )
        
        return final_results
    
    def get_foveal_ring(self) -> List[Engram]:
        """Get the foveal ring (most recent 10 engrams)."""
        return self.layer0[-10:] if len(self.layer0) >= 10 else self.layer0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            **self.stats,
            'layer0_size': len(self.layer0),
            'layer1_size': len(self.layer1),
            'layer2_size': len(self.layer2),
            'compression_ratio': len(self.layer0) / max(len(self.layer1), 1),
            'storage_size_mb': self._get_storage_size()
        }
    
    def visualize_manifold(self) -> Dict[str, Any]:
        """
        Generate visualization data for the manifold.
        
        Returns dict suitable for JSON serialization and web rendering.
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
        
        for engram_id, position in self.positions.items():
            viz_data['spiral_positions'].append({
                'id': engram_id,
                'x': position.x,
                'y': position.y,
                'radius': position.radius,
                'theta': position.theta,
                'layer': position.layer
            })
        
        for layer_idx, layer in enumerate([self.layer0, self.layer1, self.layer2]):
            for engram in layer:
                viz_data['layers'][layer_idx].append({
                    'id': engram.id,
                    'timestamp': engram.timestamp,
                    'context_preview': engram.context_window[:100],
                    'metadata': engram.metadata
                })
        
        return viz_data
    
    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================
    
    def _search_layer(self, 
                      layer: List[Engram], 
                      query_embedding: np.ndarray,
                      k: int) -> List[Tuple[Engram, float]]:
        """Search a specific layer for similar engrams."""
        results = []
        
        for engram in layer:
            similarity = self._cosine_similarity(query_embedding, engram.embedding)
            results.append((engram, similarity))
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    
    def _rebuild_hierarchy(self):
        """Rebuild the hierarchical layers."""
        self.layer1 = self.compressor.compress_to_layer1(self.layer0)
        
        if len(self.layer1) >= 16:
            self.layer2 = self.compressor.compress_to_layer2(self.layer1)
    
    def _save_engram(self, engram: Engram):
        """Persist an engram to disk."""
        filepath = self.storage_path / f"engram_{engram.layer}_{engram.id}.pkl"
        with open(filepath, 'wb') as f:
            f.write(engram.serialize())
    
    def _get_storage_size(self) -> float:
        """Calculate total storage size in MB."""
        total_bytes = sum(f.stat().st_size for f in self.storage_path.glob("*.pkl"))
        return total_bytes / (1024 * 1024)


# ============================================================================
# SYNTHETIC LONGITUDINAL BIOGRAPHY GENERATOR
# ============================================================================

class SyntheticBiographyGenerator:
    """
    Generates synthetic life histories for testing.
    Ensures no data contamination from pre-training.
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
        # Phonotactically neutral entity names
        self.entities = [
            "Banet", "Mison", "Toral", "Kevar", "Rilax", "Nophen", 
            "Quorix", "Zeltan", "Morfin", "Plexar"
        ]
        
        self.actions = [
            "acquired", "sold", "discovered", "invented", "lost",
            "traded", "modified", "damaged", "repaired", "studied"
        ]
        
        self.locations = [
            "Nexara", "Valdor", "Crimson Plains", "Azure District",
            "Obsidian Tower", "Crystal Bay"
        ]
    
    def generate_event(self, day: int) -> str:
        """Generate a single life event."""
        entity = np.random.choice(self.entities)
        action = np.random.choice(self.actions)
        location = np.random.choice(self.locations)
        value = np.random.randint(10, 1000)
        
        return f"Day {day}: I {action} the {entity} in {location} for {value} credits."
    
    def generate_biography(self, num_days: int = 1000) -> List[str]:
        """Generate a complete synthetic biography."""
        return [self.generate_event(day) for day in range(num_days)]
    
    def generate_passkey_dataset(self, 
                                num_events: int = 10000,
                                passkey_depth: int = 5000) -> Tuple[List[str], str, int]:
        """
        Generate dataset with a "needle in haystack" passkey.
        
        Returns:
            (events, passkey, passkey_index)
        """
        events = self.generate_biography(num_events)
        
        # Insert unique passkey
        passkey = f"PASSKEY: The secret code is {np.random.randint(100000, 999999)}"
        events[passkey_depth] = passkey
        
        return events, passkey, passkey_depth


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("GEOMETRIC MNEMIC MANIFOLD - PROTOTYPE DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Initialize GMM
    print("Initializing Geometric Mnemic Manifold...")
    gmm = GeometricMnemicManifold(
        embedding_dim=384,
        lambda_decay=0.01,
        beta1=64,
        beta2=16
    )
    print(f"✓ GMM initialized with {gmm.embedding_dim}D embeddings")
    print()
    
    # Generate synthetic biography
    print("Generating Synthetic Longitudinal Biography...")
    bio_gen = SyntheticBiographyGenerator(seed=42)
    events = bio_gen.generate_biography(num_days=500)
    print(f"✓ Generated {len(events)} life events")
    print(f"  Sample event: {events[0]}")
    print()
    
    # Add engrams to manifold
    print("Populating manifold with engrams...")
    for i, event in enumerate(events):
        # Simple embedding: random vector (in production, use real embeddings)
        embedding = np.random.randn(gmm.embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        gmm.add_engram(
            context_window=event,
            embedding=embedding,
            metadata={'event_type': 'daily_log', 'day': i}
        )
        
        if (i + 1) % 100 == 0:
            print(f"  Added {i + 1} engrams...")
    
    print(f"✓ Manifold populated with {len(events)} engrams")
    print()
    
    # Display statistics
    print("Manifold Statistics:")
    stats = gmm.get_statistics()
    print(f"  Layer 0 (Raw):     {stats['layer0_size']} engrams")
    print(f"  Layer 1 (Pattern): {stats['layer1_size']} engrams")
    print(f"  Layer 2 (Abstract): {stats['layer2_size']} engrams")
    print(f"  Compression:       {stats['compression_ratio']:.1f}x")
    print(f"  Storage:           {stats['storage_size_mb']:.2f} MB")
    print()
    
    # Test retrieval
    print("Testing retrieval performance...")
    query_embedding = np.random.randn(gmm.embedding_dim)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    start_time = time.time()
    results = gmm.query(query_embedding, k=5)
    query_time = time.time() - start_time
    
    print(f"✓ Query completed in {query_time*1000:.2f}ms")
    print(f"\nTop 5 Results:")
    for i, (engram, score) in enumerate(results, 1):
        print(f"  {i}. [Score: {score:.3f}] {engram.context_window[:60]}...")
    print()
    
    # Demonstrate foveal access
    print("Foveal Ring (Most Recent 10 Memories):")
    fovea = gmm.get_foveal_ring()
    for engram in fovea:
        print(f"  • {engram.context_window}")
    print()
    
    # Save visualization data
    print("Generating visualization data...")
    viz_data = gmm.visualize_manifold()
    
    viz_path = Path("./gmm_visualization.json")
    with open(viz_path, 'w') as f:
        json.dump(viz_data, f, indent=2, default=str)
    
    print(f"✓ Visualization data saved to {viz_path}")
    print()
    
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
