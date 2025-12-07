"""
Benchmark visualization utilities.

Generates plots and charts for benchmark results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict


class BenchmarkVisualizer:
    """Handles visualization of benchmark results."""

    def __init__(self, output_dir: Path = Path("./benchmark_results")):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory for saving visualizations
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def visualize_needle_results(self, results: Dict):
        """
        Generate visualization plots for Needle in the Spiral benchmark.

        Creates a 2x2 grid of plots:
        1. Retrieval Time vs Depth
        2. Speedup Factor
        3. Recall Performance
        4. Complexity Analysis

        Args:
            results: Results dictionary from NeedleInSpiralBenchmark
        """
        if not results.get('gmm_results'):
            print("No results to visualize. Run benchmark first.")
            return

        print("\nGenerating visualizations...")

        # Extract data
        depths = [r['depth'] for r in results['gmm_results']]
        gmm_times = [np.mean(r['gmm_times']) for r in results['gmm_results']]
        hnsw_times = [np.mean(r['hnsw_times']) for r in results['gmm_results']]
        gmm_recalls = [np.mean(r['gmm_recall']) for r in results['gmm_results']]
        hnsw_recalls = [np.mean(r['hnsw_recall']) for r in results['gmm_results']]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Needle in the Spiral Benchmark Results',
                    fontsize=16, fontweight='bold')

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
        ax2.bar(range(len(depths)), speedups, color='#4a9eff',
               edgecolor='black', linewidth=1.5)
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

    def save_results_json(self, results: Dict):
        """
        Save benchmark results to JSON.

        Args:
            results: Results dictionary to save
        """
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
                for r in results['gmm_results']
            ],
            'metadata': results['metadata']
        }

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"✓ Results saved to {results_path}")
