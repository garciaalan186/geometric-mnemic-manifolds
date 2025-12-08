import time
import sys
import os
import numpy as np

# Add parent directory to path to find gmm_native
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hnswlib
import gmm_native  # The compiled module

def benchmark_native(n=100000, dim=128, k=10):
    print(f"Benchmarking Native GMM vs NumPy vs HNSW (N={n}, D={dim})")
    
    # 1. Prepare Data
    data = np.random.rand(n, dim).astype(np.float32)
    indices = np.arange(n, dtype=np.int32)
    query = np.random.rand(dim).astype(np.float32)
    
    # 2. Benchmark HNSW (Approximate)
    p = hnswlib.Index(space='ip', dim=dim)
    p.init_index(max_elements=n, ef_construction=200, M=16)
    p.add_items(data, indices)
    p.set_ef(50)
    
    start = time.time()
    for _ in range(100):
        labels, distances = p.knn_query(query, k=k)
    hnsw_time = (time.time() - start) / 100 * 1000
    print(f"HNSW Latency: {hnsw_time:.3f} ms (Approx)")

    # 3. Benchmark NumPy (Exact)
    start = time.time()
    for _ in range(100):
        scores = np.dot(data, query)
        top_k_idx = np.argpartition(scores, -k)[-k:]
    numpy_time = (time.time() - start) / 100 * 1000
    print(f"NumPy Latency: {numpy_time:.3f} ms (Exact)")
    
    # 4. Benchmark Native C++ (Exact)
    # Ensure contiguous C-order arrays
    data_c = np.ascontiguousarray(data)
    indices_c = np.ascontiguousarray(indices)
    query_c = np.ascontiguousarray(query)
    
    start = time.time()
    for _ in range(100):
        results = gmm_native.fast_scan(data_c, indices_c, query_c, k)
    native_time = (time.time() - start) / 100 * 1000
    print(f"Native C++ Latency: {native_time:.3f} ms (Exact)")
    
    print("-" * 30)
    print(f"Speedup vs NumPy: {numpy_time / native_time:.2f}x")
    print(f"Gap vs HNSW: {native_time / hnsw_time:.2f}x Slowdown")

if __name__ == "__main__":
    benchmark_native(n=1000000) # 1 Million Vectors
