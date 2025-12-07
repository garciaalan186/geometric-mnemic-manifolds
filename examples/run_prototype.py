#!/usr/bin/env python3
"""
Geometric Mnemic Manifold - Prototype Demonstration

A functional demonstration of the GMM architecture using the modular
refactored codebase. This script mirrors the original gmm_prototype.py
but uses the new SOLID-based package structure.

Demonstrates:
1. Kronecker sequence-based spiral positioning
2. Hierarchical layer construction
3. Foveated memory access
4. Engram serialization and retrieval
5. Visualization data generation
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gmm import (
    GeometricMnemicManifold,
    SyntheticBiographyGenerator
)


def main():
    """Run the prototype demonstration."""
    print("=" * 80)
    print("GEOMETRIC MNEMIC MANIFOLD - PROTOTYPE DEMONSTRATION")
    print("=" * 80)
    print()

    # ========================================================================
    # 1. INITIALIZATION
    # ========================================================================
    print("Initializing Geometric Mnemic Manifold...")
    gmm = GeometricMnemicManifold(
        embedding_dim=384,
        lambda_decay=0.01,
        beta1=64,
        beta2=16
    )
    print(f"✓ GMM initialized with {gmm.embedding_dim}D embeddings")
    print()

    # ========================================================================
    # 2. SYNTHETIC BIOGRAPHY GENERATION
    # ========================================================================
    print("Generating Synthetic Longitudinal Biography...")
    bio_gen = SyntheticBiographyGenerator(seed=42)
    events = bio_gen.generate_biography(num_days=500)
    print(f"✓ Generated {len(events)} life events")
    print(f"  Sample event: {events[0]}")
    print()

    # ========================================================================
    # 3. POPULATE MANIFOLD
    # ========================================================================
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

    # ========================================================================
    # 4. STATISTICS
    # ========================================================================
    print("Manifold Statistics:")
    stats = gmm.get_statistics()
    print(f"  Layer 0 (Raw):      {stats['layer0_size']} engrams")
    print(f"  Layer 1 (Pattern):  {stats['layer1_size']} engrams")
    print(f"  Layer 2 (Abstract): {stats['layer2_size']} engrams")
    print(f"  Compression:        {stats['compression_ratio']:.1f}x")
    print(f"  Storage:            {stats['storage_size_mb']:.2f} MB")
    print()

    # ========================================================================
    # 5. RETRIEVAL TEST
    # ========================================================================
    print("Testing retrieval performance...")
    query_embedding = np.random.randn(gmm.embedding_dim)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    start_time = time.time()
    results = gmm.query(query_embedding, k=5)
    query_time = time.time() - start_time

    print(f"✓ Query completed in {query_time*1000:.2f}ms")
    print(f"\nTop 5 Results:")
    for i, (engram, score) in enumerate(results, 1):
        preview = engram.context_window[:60]
        print(f"  {i}. [Score: {score:.3f}] {preview}...")
    print()

    # ========================================================================
    # 6. FOVEAL ACCESS
    # ========================================================================
    print("Foveal Ring (Most Recent 10 Memories):")
    fovea = gmm.get_foveal_ring()
    for engram in fovea:
        print(f"  • {engram.context_window}")
    print()

    # ========================================================================
    # 7. VISUALIZATION
    # ========================================================================
    print("Generating visualization data...")
    viz_path = Path("./gmm_visualization.json")
    gmm.save_visualization(viz_path)

    print(f"✓ Visualization data saved to {viz_path}")
    print()

    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  • Open gmm_visualization_viewer.html to explore the manifold")
    print("  • Run benchmarks: python benchmarks/run_benchmarks.py")
    print("  • Try demos: python examples/demos/run_all_demos.py")


if __name__ == "__main__":
    main()
