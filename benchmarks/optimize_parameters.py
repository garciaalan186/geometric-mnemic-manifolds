"""
Parameter Optimization Grid Search for GMM.

Iterates through combinations of hyperparameters to find optimal
trade-offs between recall and latency.
"""

import itertools
import json
import time
import numpy as np
from pathlib import Path
from needle_benchmark import NeedleInSpiralBenchmark

def run_grid_search():
    print("=" * 80)
    print("GMM PARAMETER OPTIMIZATION GRID SEARCH")
    print("=" * 80)
    
    # Parameter Grid
    # Focused range based on initial findings
    grid = {
        'beta1': [32, 64],
        'beta2': [8, 16],
        'k': [5, 20],  # Retrieval beam width (requires changing retriever default or passing k)
        # Note: 'k' is currently a query-time param, not init param.
        # We need to ensure GMM.query uses this k.
        # But GMM constructor doesn't take 'k'. 
        # Actually gmm.query takes k. Benchmark calls gmm.query with k=1.
        # Wait, the optimization requested "recall maintained but retrieval time doesn't increase".
        # Recall is affected by internal 'k' in retriever, which is hardcoded in query/retriever.py 
        # OR passed via some mechanism.
        # In my recent edit to retriever.py, I hardcoded k=20 for internal drilled down.
        # To make it tunable, I should probably pass it or make it configurable?
        # For now, let's optimize beta1/beta2 and lambda.
        
        'lambda_decay_factor': [0.01], # Fixed based on previous optimization
        'foveal_ring_size': [10, 50, 100] # Optimizing this now
    }
    
    combinations = list(itertools.product(
        grid['beta1'],
        grid['beta2'],
        grid['lambda_decay_factor'],
        grid['foveal_ring_size']
    ))
    
    print(f"Total Combinations: {len(combinations)}")
    
    results = []
    
    benchmark = NeedleInSpiralBenchmark(output_dir=Path("./benchmark_results/optimization"))
    
    for i, (b1, b2, decay, fovea) in enumerate(combinations):
        print(f"\nRunning combination {i+1}/{len(combinations)}:")
        print(f"  beta1={b1}, beta2={b2}, decay={decay}, fovea={fovea}")
        
        # Quick test at depth 1000
        depths = [1000] 
        
        lambda_val = decay / np.log(1001)
        
        params = {
            'beta1': b1,
            'beta2': b2,
            'lambda_decay': lambda_val,
            'foveal_ring_size': fovea
        }
        
        res = benchmark.run_passkey_retrieval_test(
            depths=depths,
            num_trials=3,
            gmm_params=params
        )
        
        # Extract metrics for depth 1000
        gmm_results = res['gmm_results'][-1]
        mean_time = np.mean(gmm_results['gmm_times'])
        mean_recall = np.mean(gmm_results['gmm_recall'])
        
        print(f"  -> Recall: {mean_recall*100:.1f}%, Time: {mean_time:.2f}ms")
        
        results.append({
            'params': params,
            'recall': mean_recall,
            'latency': mean_time
        })
        
    # Find Pareto optimal
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    
    # Sort by Recall (desc), then Time (asc)
    sorted_results = sorted(results, key=lambda x: (-x['recall'], x['latency']))
    
    print(f"{'Beta1':<6} {'Beta2':<6} {'Fovea':<6} | {'Recall':<8} {'Latency':<8}")
    print("-" * 60)
    
    for r in sorted_results:
        p = r['params']
        print(f"{p['beta1']:<6} {p['beta2']:<6} {p['foveal_ring_size']:<6} | {r['recall']*100:<6.1f}% {r['latency']:<6.2f}ms")
        
    best = sorted_results[0]
    print("\nBest Configuration:")
    print(json.dumps(best, indent=2))
    
    # Save 
    with open("benchmark_results/optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_grid_search()
