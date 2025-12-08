import numpy as np

def simulate_distribution(N, alpha=5.0):
    """
    Test a logarithmic distribution of points.
    We want points dense near N, sparse near 0.
    
    Formula idea:
    Map uniform k in [0, 1] to indices via exponential CDF.
    """
    # Number of points M? 
    # User says max sparsity is function of N.
    # Let's say we want max sparsity S_max.
    # The largest gap is at the "sparse" end (near 0).
    # Gap_0 = i_1 - i_0? No, i_last - i_second_last.
    
    # Let's try defining points by "staleness" y = N - 1 - i
    # y_k = y_{k-1} * scalar ? (Geometric series of gaps?)
    # y_0 = 0
    # y_1 = 1
    # y_2 = 1 + beta
    # y_3 = 1 + beta + beta^2 ...
    
    # Geometric positions: y_k = c * (beta^k - 1)
    # We want y_last = N-1 (so index is 0).
    # N-1 = c * (beta^M - 1)
    
    # This guarantees exponential decay of density.
    # The gaps grow as beta^k.
    # Max sparsity (largest gap) is the last gap: approx y_M - y_{M-1}
    # Gap_last = c * beta^{M-1} * (beta - 1)
    
    # Simulating for different N to see the relationship
    print(f"\n--- N = {N} ---")
    
    # Option 1: Fix beta, calculate M
    # Let beta = 1.5 (Gaps grow by 50% each step)
    beta = 1.2
    # We need c * beta^M approx N.
    # Let c = 1 (start with gap 1).
    # beta^M = N => M = log_beta(N)
    
    M = int(np.log(N)/np.log(beta))
    
    # Generate indices
    indices = []
    c = (N-1) / (beta**M - 1) # Adjust c slightly to hit N-1 exactly
    
    for k in range(M + 1):
        y = c * (beta**k - 1)
        idx = (N - 1) - int(y)
        if idx >= 0:
            indices.append(idx)
            
    # Force 0 and N-1 to be included exactly
    indices = sorted(list(set([0, N-1] + indices)))
    
    print(f"Indices: {indices}")
    print(f"Count: {len(indices)}")
    
    gaps = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
    print(f"Gaps (near 0 -> near N): {gaps}")
    print(f"Max Sparsity (Gap near 0): {gaps[0]}")
    print(f"Min Sparsity (Gap near N): {gaps[-1]}")
    
    return gaps[0]

# Run for a few Ns
simulate_distribution(100)
simulate_distribution(1000)
simulate_distribution(10000)
