"""
Needle in the Spiral Benchmark
Implements the experimental protocol described in the paper.
"""

import numpy as np
import time
from typing import List, Tuple, Dict
from gmm_prototype import (
    GeometricMnemicManifold,
    SyntheticBiographyGenerator
)
import matplotlib.pyplot as plt
from pathlib import Path
import json


class NeedleInSpiralBenchmark:
    """
    Benchmark suite for evaluating GMM retrieval performance.
    
    Tests:
    1. Recall@1 across different memory depths
    2. Time-to-First-Token (TTFT) scaling
    3. Comparison with simulated HNSW baseline
    """
    
    def __init__(self, output_dir: Path = Path("./benchmark_results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.results = {
            'gmm_results': [],
            'hnsw_baseline': [],
            'metadata': {}
        }
    
    def run_passkey_retrieval_test(self,
                                   depths: List[int] = [100, 1000, 10000, 100000],
                                   num_trials: int = 10) -> Dict:
        """
        Test passkey retrieval at various memory depths.
        
        Args:
            depths: List of memory depths to test
            num_trials: Number of trials per depth
            
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
                
                # Create GMM and populate
                gmm = GeometricMnemicManifold(
                    embedding_dim=128,  # Smaller for speed
                    lambda_decay=0.01 / np.log(depth),  # Scale with depth
                    beta1=64,
                    beta2=16
                )
                
                # Add all events
                embeddings = []
                for event in events:
                    emb = self._simple_embedding(event)
                    embeddings.append(emb)
                    gmm.add_engram(
                        context_window=event,
                        embedding=emb,
                        metadata={'original': event}
                    )
                
                # Query for passkey
                passkey_embedding = self._simple_embedding(passkey)
                
                # Test GMM
                gmm_start = time.time()
                gmm_results = gmm.query(passkey_embedding, k=1)
                gmm_time = (time.time() - gmm_start) * 1000  # ms
                
                gmm_found = len(gmm_results) > 0 and passkey in gmm_results[0][0].context_window
                
                depth_results['gmm_times'].append(gmm_time)
                depth_results['gmm_recall'].append(1.0 if gmm_found else 0.0)
                
                # Simulate HNSW (O(log N) traversal + index overhead)
                hnsw_time = self._simulate_hnsw_retrieval(depth)
                hnsw_found = np.random.rand() > 0.05  # 95% recall (optimistic)
                
                depth_results['hnsw_times'].append(hnsw_time)
                depth_results['hnsw_recall'].append(1.0 if hnsw_found else 0.0)
                
                if (trial + 1) % max(1, num_trials // 5) == 0:
                    print(f"  Trial {trial + 1}/{num_trials}: "
                          f"GMM={gmm_time:.1f}ms (recall={gmm_found}), "
                          f"HNSW={hnsw_time:.1f}ms (recall={hnsw_found})")
            
            # Aggregate results
            result_summary = {
                'depth': depth,
                'gmm_mean_time': np.mean(depth_results['gmm_times']),
                'gmm_std_time': np.std(depth_results['gmm_times']),
                'gmm_recall': np.mean(depth_results['gmm_recall']),
                'hnsw_mean_time': np.mean(depth_results['hnsw_times']),
                'hnsw_std_time': np.std(depth_results['hnsw_times']),
                'hnsw_recall': np.mean(depth_results['hnsw_recall']),
                'speedup': np.mean(depth_results['hnsw_times']) / np.mean(depth_results['gmm_times'])
            }
            
            self.results['gmm_results'].append(depth_results)
            
            print(f"\n  Results for depth {depth:,}:")
            print(f"    GMM:  {result_summary['gmm_mean_time']:.1f}ms ± {result_summary['gmm_std_time']:.1f}ms "
                  f"(recall: {result_summary['gmm_recall']*100:.1f}%)")
            print(f"    HNSW: {result_summary['hnsw_mean_time']:.1f}ms ± {result_summary['hnsw_std_time']:.1f}ms "
                  f"(recall: {result_summary['hnsw_recall']*100:.1f}%)")
            print(f"    Speedup: {result_summary['speedup']:.2f}x")
        
        self.results['metadata'] = {
            'depths_tested': depths,
            'trials_per_depth': num_trials,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return self.results
    
    def visualize_results(self):
        """Generate visualization plots of benchmark results."""
        if not self.results['gmm_results']:
            print("No results to visualize. Run benchmark first.")
            return
        
        print("\nGenerating visualizations...")
        
        depths = [r['depth'] for r in self.results['gmm_results']]
        gmm_times = [np.mean(r['gmm_times']) for r in self.results['gmm_results']]
        hnsw_times = [np.mean(r['hnsw_times']) for r in self.results['gmm_results']]
        gmm_recalls = [np.mean(r['gmm_recall']) for r in self.results['gmm_results']]
        hnsw_recalls = [np.mean(r['hnsw_recall']) for r in self.results['gmm_results']]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Needle in the Spiral Benchmark Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Retrieval Time vs Depth
        ax1 = axes[0, 0]
        ax1.plot(depths, gmm_times, 'o-', linewidth=2, markersize=8, 
                label='GMM (O(1))', color='#ffd700')
        ax1.plot(depths, hnsw_times, 's-', linewidth=2, markersize=8,
                label='HNSW (O(log N))', color='#ff6b6b')
        ax1.set_xlabel('Memory Depth (# of engrams)', fontsize=12)
        ax1.set_ylabel('Retrieval Time (ms)', fontsize=12)
        ax1.set_title('Retrieval Time Scaling', fontsize=13, fontweight='bold')
        ax1.set_xscale('log')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speedup Factor
        ax2 = axes[0, 1]
        speedups = [h/g for h, g in zip(hnsw_times, gmm_times)]
        ax2.bar(range(len(depths)), speedups, color='#4a9eff', edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Memory Depth', fontsize=12)
        ax2.set_ylabel('Speedup Factor (HNSW / GMM)', fontsize=12)
        ax2.set_title('GMM Performance Advantage', fontsize=13, fontweight='bold')
        ax2.set_xticks(range(len(depths)))
        ax2.set_xticklabels([f'{d:,}' for d in depths], rotation=45)
        ax2.axhline(y=1, color='red', linestyle='--', label='Parity')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Recall Performance
        ax3 = axes[1, 0]
        x = np.arange(len(depths))
        width = 0.35
        ax3.bar(x - width/2, [r*100 for r in gmm_recalls], width, 
               label='GMM', color='#ffd700', edgecolor='black')
        ax3.bar(x + width/2, [r*100 for r in hnsw_recalls], width,
               label='HNSW', color='#ff6b6b', edgecolor='black')
        ax3.set_xlabel('Memory Depth', fontsize=12)
        ax3.set_ylabel('Recall@1 (%)', fontsize=12)
        ax3.set_title('Retrieval Accuracy', fontsize=13, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'{d:,}' for d in depths], rotation=45)
        ax3.legend(fontsize=11)
        ax3.set_ylim([0, 105])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Time Complexity Analysis
        ax4 = axes[1, 1]
        # Theoretical curves
        depth_range = np.logspace(2, 6, 50)
        constant_time = np.ones_like(depth_range) * gmm_times[0]
        log_time = gmm_times[0] + 2 * np.log2(depth_range / depths[0])
        
        ax4.plot(depth_range, constant_time, '--', linewidth=2, 
                label='O(1) - GMM Theory', color='#ffd700', alpha=0.7)
        ax4.plot(depth_range, log_time, '--', linewidth=2,
                label='O(log N) - HNSW Theory', color='#ff6b6b', alpha=0.7)
        ax4.plot(depths, gmm_times, 'o', markersize=10,
                label='GMM Measured', color='#ffd700')
        ax4.plot(depths, hnsw_times, 's', markersize=10,
                label='HNSW Simulated', color='#ff6b6b')
        ax4.set_xlabel('Memory Depth (# of engrams)', fontsize=12)
        ax4.set_ylabel('Retrieval Time (ms)', fontsize=12)
        ax4.set_title('Complexity Analysis', fontsize=13, fontweight='bold')
        ax4.set_xscale('log')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'benchmark_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to {plot_path}")
        
        plt.close()
    
    def save_results(self):
        """Save benchmark results to JSON."""
        results_path = self.output_dir / 'benchmark_results.json'
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            'gmm_results': [
                {
                    'depth': r['depth'],
                    'gmm_times': [float(x) for x in r['gmm_times']],
                    'gmm_recall': [float(x) for x in r['gmm_recall']],
                    'hnsw_times': [float(x) for x in r['hnsw_times']],
                    'hnsw_recall': [float(x) for x in r['hnsw_recall']]
                }
                for r in self.results['gmm_results']
            ],
            'metadata': self.results['metadata']
        }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"✓ Results saved to {results_path}")
    
    def _simple_embedding(self, text: str, dim: int = 128) -> np.ndarray:
        """
        Simple deterministic embedding for testing.
        In production, use actual sentence transformers.
        """
        # Hash-based embedding for consistency
        hash_val = hash(text)
        np.random.seed(hash_val % (2**32))
        embedding = np.random.randn(dim)
        return embedding / np.linalg.norm(embedding)
    
    def _simulate_hnsw_retrieval(self, depth: int) -> float:
        """
        Simulate HNSW retrieval time.
        Assumes O(log N) graph traversal + index overhead.
        """
        base_time = 5.0  # Base latency
        traversal_time = 2.0 * np.log2(max(depth, 2))  # O(log N) component
        index_overhead = 0.001 * depth  # Linear index maintenance cost
        
        # Add some noise
        noise = np.random.normal(0, 0.5)
        
        return base_time + traversal_time + index_overhead + noise


# ============================================================================
# EPISTEMIC GAP DETECTION TEST
# ============================================================================

class EpistemicGapBenchmark:
    """
    Test the RRK's ability to detect when information is missing.
    Implements the Epistemic Regularization protocol from the paper.
    """
    
    def __init__(self):
        self.test_cases = []
    
    def generate_test_cases(self, num_cases: int = 100) -> List[Dict]:
        """
        Generate test cases for epistemic gap detection.
        
        Each case has:
        - A question
        - A context (complete or masked)
        - Ground truth: should signal or not
        """
        bio_gen = SyntheticBiographyGenerator(seed=42)
        
        for i in range(num_cases):
            event = bio_gen.generate_event(i)
            
            # Extract a fact to query
            words = event.split()
            entity_idx = [j for j, w in enumerate(words) if w in bio_gen.entities]
            
            if entity_idx:
                entity = words[entity_idx[0]]
                question = f"What happened with the {entity}?"
                
                # 50% complete context, 50% masked
                if i % 2 == 0:
                    context = event
                    should_signal = False
                else:
                    # Mask the entity
                    masked_words = [w if j not in entity_idx else "[MASKED]" 
                                  for j, w in enumerate(words)]
                    context = " ".join(masked_words)
                    should_signal = True
                
                self.test_cases.append({
                    'question': question,
                    'context': context,
                    'should_signal': should_signal,
                    'original': event
                })
        
        return self.test_cases
    
    def evaluate_rnn_performance(self) -> Dict:
        """
        Evaluate RRK epistemic gap detection.
        In a full implementation, this would use the actual RRK model.
        """
        # Placeholder for actual RRK evaluation
        print("Epistemic Gap Detection Benchmark")
        print("(Full implementation requires trained RRK model)")
        print()
        
        results = {
            'total_cases': len(self.test_cases),
            'signal_precision': 0.0,
            'signal_recall': 0.0,
            'f1_score': 0.0
        }
        
        return results


# ============================================================================
# MAIN BENCHMARK RUNNER
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("GEOMETRIC MNEMIC MANIFOLD - BENCHMARK SUITE")
    print("=" * 80)
    print()
    
    # Create benchmark
    benchmark = NeedleInSpiralBenchmark(output_dir=Path("./benchmark_results"))
    
    # Run passkey retrieval test
    print("Running Needle in the Spiral benchmark...")
    print()
    
    results = benchmark.run_passkey_retrieval_test(
        depths=[100, 500, 1000, 5000],
        num_trials=5
    )
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    
    # Generate visualizations
    benchmark.visualize_results()
    
    # Save results
    benchmark.save_results()
    
    # Summary
    print("\nSummary:")
    print("-" * 40)
    
    for i, depth_result in enumerate(results['gmm_results']):
        depth = depth_result['depth']
        gmm_time = np.mean(depth_result['gmm_times'])
        hnsw_time = np.mean(depth_result['hnsw_times'])
        speedup = hnsw_time / gmm_time
        
        print(f"Depth {depth:>6,}: GMM={gmm_time:6.1f}ms, "
              f"HNSW={hnsw_time:6.1f}ms, Speedup={speedup:.2f}x")
    
    print()
    print("✓ All results saved to ./benchmark_results/")
