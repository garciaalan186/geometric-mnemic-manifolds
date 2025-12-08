"""
Needle in the Spiral Benchmark

Implements the passkey retrieval experimental protocol described in the paper.
Tests GMM retrieval performance across various memory depths.
"""

import sys
import numpy as np
import time
import json
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gmm.core.manifold import GeometricMnemicManifold
from src.gmm.core.engram import Engram
from src.gmm import SyntheticBiographyGenerator
from baselines.hnsw import HNSWIndex


class NeedleInSpiralBenchmark:
    """
    Benchmark suite for evaluating GMM retrieval performance.

    Tests:
    1. Recall@1 across different memory depths
    2. Time-to-First-Token (TTFT) scaling
    3. Comparison with simulated HNSW baseline
    """

    def __init__(self, output_dir: Path = Path("./benchmark_results")):
        """
        Initialize benchmark.

        Args:
            output_dir: Directory for saving results
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.results = {
            'gmm_results': [],
            'hnsw_baseline': [],
            'metadata': {}
        }

    def run_passkey_retrieval_test(self,
                                   depths: List[int] = [100, 1000, 10000],
                                   num_trials: int = 10,
                                   embedding_dim: int = 128,
                                   gmm_params: Dict[str, Any] = None) -> Dict:
        """
        Test passkey retrieval at various memory depths.

        Args:
            depths: List of memory depths to test
            num_trials: Number of trials per depth
            embedding_dim: Embedding dimensionality

        Returns:
            Dictionary of results
        """
        print("=" * 80)
        print("NEEDLE IN THE SPIRAL BENCHMARK")
        print("=" * 80)
        print()

        bio_gen = SyntheticBiographyGenerator(seed=42)

        for depth in depths:
            print(f"\nTesting at depth: {depth:,} memories")
            print("-" * 40)

            depth_results = {
                'depth': depth,
                'gmm_times': [],
                'gmm_recall': [],
                'hnsw_times': [],
                'hnsw_recall': []
            }

            for trial in range(num_trials):
                # Generate dataset with passkey
                events, passkey, passkey_idx = bio_gen.generate_passkey_dataset(
                    num_events=depth,
                    passkey_depth=int(depth * 0.5)  # Middle of history
                )

                # Create GMM and HNSW
                gmm_kwargs = {
                    'embedding_dim': embedding_dim,
                    'lambda_decay': 0.01 / np.log(depth + 1),
                    'beta1': 64,
                    'beta2': 16,
                    'foveal_beta': 1.1  # Distributed fovea (10% decay)
                }
                if gmm_params:
                    gmm_kwargs.update(gmm_params)
                    
                gmm = GeometricMnemicManifold(**gmm_kwargs)
                
                hnsw = HNSWIndex(
                    embedding_dim=embedding_dim,
                    m=16,
                    ef_construction=200
                )

                # Add all events
                for i, event in enumerate(events):
                    emb = self._simple_embedding(event, dim=embedding_dim)
                    # Add to GMM
                    gmm.add_engram(
                        context_window=event,
                        embedding=emb,
                        metadata={'original': event}
                    )
                    # Add to HNSW (requires explicit index)
                    hnsw.add_item(emb, i)

                # Query for passkey
                passkey_embedding = self._simple_embedding(passkey, dim=embedding_dim)

                # Pre-warm GMM matrices (measure pure retrieval speed)
                if hasattr(gmm, '_update_matrices'):
                    gmm._update_matrices()

                # Test GMM
                gmm_start = time.time()
                gmm_results = gmm.query(passkey_embedding, k=1)
                gmm_time = (time.time() - gmm_start) * 1000  # ms

                gmm_found = (len(gmm_results) > 0 and
                            passkey in gmm_results[0][0].context_window)

                depth_results['gmm_times'].append(gmm_time)
                depth_results['gmm_recall'].append(1.0 if gmm_found else 0.0)

                # Test HNSW (Real retrieval)
                hnsw_start = time.time()
                hnsw_results = hnsw.search(passkey_embedding, k=1) # Returns list of (idx, score)
                hnsw_time = (time.time() - hnsw_start) * 1000 # ms
                
                # Check recall (did we get the index of the passkey?)
                # passkey_idx is the index of the passkey in the 'events' list
                hnsw_found = False
                if hnsw_results:
                    found_idx = hnsw_results[0][0]
                    if found_idx == passkey_idx:
                         hnsw_found = True
                    # Double check if events[found_idx] == passkey
                    elif events[found_idx] == passkey: # Should match by index, but content safety check
                         hnsw_found = True

                depth_results['hnsw_times'].append(hnsw_time)
                depth_results['hnsw_recall'].append(1.0 if hnsw_found else 0.0)

                if (trial + 1) % max(1, num_trials // 5) == 0:
                    print(f"  Trial {trial + 1}/{num_trials}: "
                          f"GMM={gmm_time:.1f}ms (recall={gmm_found}), "
                          f"HNSW={hnsw_time:.1f}ms (recall={hnsw_found})")

            # Aggregate results
            # Handle division by zero for speedup if GMM is super fast (0.0ms)
            mean_gmm = np.mean(depth_results['gmm_times'])
            mean_hnsw = np.mean(depth_results['hnsw_times'])
            speedup = mean_hnsw / mean_gmm if mean_gmm > 1e-9 else 0.0

            result_summary = {
                'depth': depth,
                'gmm_mean_time': mean_gmm,
                'gmm_std_time': np.std(depth_results['gmm_times']),
                'gmm_recall': np.mean(depth_results['gmm_recall']),
                'hnsw_mean_time': mean_hnsw,
                'hnsw_std_time': np.std(depth_results['hnsw_times']),
                'hnsw_recall': np.mean(depth_results['hnsw_recall']),
                'speedup': speedup
            }

            self.results['gmm_results'].append(depth_results)

            print(f"\n  Results for depth {depth:,}:")
            print(f"    GMM:  {result_summary['gmm_mean_time']:.1f}ms ± "
                  f"{result_summary['gmm_std_time']:.1f}ms "
                  f"(recall: {result_summary['gmm_recall']*100:.1f}%)")
            print(f"    HNSW: {result_summary['hnsw_mean_time']:.1f}ms ± "
                  f"{result_summary['hnsw_std_time']:.1f}ms "
                  f"(recall: {result_summary['hnsw_recall']*100:.1f}%)")
            print(f"    Speedup: {result_summary['speedup']:.2f}x")

        self.results['metadata'] = {
            'depths_tested': depths,
            'trials_per_depth': num_trials,
            'embedding_dim': embedding_dim,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        return self.results

    def get_results(self) -> Dict:
        """Get benchmark results."""
        return self.results

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _simple_embedding(self, text: str, dim: int = 128) -> np.ndarray:
        """
        Simple deterministic embedding for testing.

        In production, use actual sentence transformers.
        Uses hash-based seeding for consistency across runs.

        Args:
            text: Input text
            dim: Embedding dimension

        Returns:
            Normalized embedding vector
        """
        hash_val = hash(text)
        np.random.seed(hash_val % (2**32))
        embedding = np.random.randn(dim)
        return embedding / np.linalg.norm(embedding)

    def _simulate_hnsw_retrieval(self, depth: int) -> float:
        """
        Simulate HNSW retrieval time.

        Assumes O(log N) graph traversal + index overhead.
        Based on empirical HNSW performance characteristics.

        Args:
            depth: Number of vectors in index

        Returns:
            Simulated retrieval time in milliseconds
        """
        base_time = 5.0  # Base latency
        traversal_time = 2.0 * np.log2(max(depth, 2))  # O(log N) component
        index_overhead = 0.001 * depth  # Linear index maintenance cost

        # Add some realistic noise
        noise = np.random.normal(0, 0.5)

        return base_time + traversal_time + index_overhead + noise
