#!/usr/bin/env python3
"""
Main Benchmark Runner

Executes all GMM benchmarks and generates comprehensive results.
"""

import numpy as np
from pathlib import Path

from needle_benchmark import NeedleInSpiralBenchmark
from epistemic_benchmark import EpistemicGapBenchmark
from visualizer import BenchmarkVisualizer


def main():
    """Run all benchmarks."""
    print("=" * 80)
    print("GEOMETRIC MNEMIC MANIFOLD - BENCHMARK SUITE")
    print("=" * 80)
    print()

    output_dir = Path("./benchmark_results")

    # ========================================================================
    # 1. NEEDLE IN THE SPIRAL BENCHMARK
    # ========================================================================
    print("Running Needle in the Spiral benchmark...")
    print()

    needle_benchmark = NeedleInSpiralBenchmark(output_dir=output_dir)

    results = needle_benchmark.run_passkey_retrieval_test(
        depths=[100, 500, 1000, 5000],
        num_trials=5
    )

    print("\n" + "=" * 80)
    print("NEEDLE BENCHMARK COMPLETE")
    print("=" * 80)

    # Generate visualizations
    visualizer = BenchmarkVisualizer(output_dir=output_dir)
    visualizer.visualize_needle_results(results)
    visualizer.save_results_json(results)

    # Print summary
    print("\nSummary:")
    print("-" * 40)

    for depth_result in results['gmm_results']:
        depth = depth_result['depth']
        gmm_time = np.mean(depth_result['gmm_times'])
        hnsw_time = np.mean(depth_result['hnsw_times'])
        speedup = hnsw_time / gmm_time

        print(f"Depth {depth:>6,}: GMM={gmm_time:6.1f}ms, "
              f"HNSW={hnsw_time:6.1f}ms, Speedup={speedup:.2f}x")

    print()

    # ========================================================================
    # 2. EPISTEMIC GAP BENCHMARK
    # ========================================================================
    print("\n" + "=" * 80)
    print("Epistemic Gap Detection")
    print("=" * 80)
    print()

    epistemic_benchmark = EpistemicGapBenchmark()
    test_cases = epistemic_benchmark.generate_test_cases(num_cases=100)
    epistemic_results = epistemic_benchmark.evaluate_rrk_performance()

    print()

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("=" * 80)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 80)
    print()
    print(f"✓ Results saved to {output_dir}/")
    print(f"✓ Visualizations saved to {output_dir}/benchmark_results.png")
    print()
    print("Next steps:")
    print("  • Review results in ./benchmark_results/")
    print("  • Compare with baselines in the paper")
    print("  • Run with larger depths for extended analysis")


if __name__ == "__main__":
    main()
