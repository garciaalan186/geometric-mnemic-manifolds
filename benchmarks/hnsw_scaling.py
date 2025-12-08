
import time
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from benchmarks.baselines.hnsw import HNSWIndex

def run_hnsw_benchmark(n_items, dim=128):
    print(f"Benchmarking Pure Python HNSW (N={n_items}, D={dim})...")
    
    # 1. Generate Data
    data = np.random.rand(n_items, dim).astype(np.float32)
    query = np.random.rand(dim).astype(np.float32)
    
    # 2. Build Index
    print("  Building Graph (this takes time in Python)...")
    start_build = time.time()
    index = HNSWIndex(embedding_dim=dim, m=16, ef_construction=100)
    for i in range(n_items):
        index.add_item(data[i], i)
        if i % 1000 == 0 and i > 0:
            sys.stdout.write(f"\r  Indexed {i}/{n_items}")
            sys.stdout.flush()
    print(f"\r  Indexed {n_items}/{n_items} in {time.time()-start_build:.2f}s")
    
    # 3. Query
    print("  Querying...")
    search_times = []
    for _ in range(10):
        start = time.time()
        index.search(query, k=10)
        lat = (time.time() - start) * 1000
        search_times.append(lat)
        
    avg_lat = sum(search_times) / len(search_times)
    print(f"  Avg Latency: {avg_lat:.2f} ms")
    return avg_lat

if __name__ == "__main__":
    # We test small scales because Pure Python HNSW construction is O(N log N) but slow constants.
    # N=10k should be feasible.
    l_1k = run_hnsw_benchmark(1000)
    l_10k = run_hnsw_benchmark(10000)
    
    print("-" * 40)
    print("SCALABILITY REPORT (Python HNSW):")
    print(f"N=1,000  : {l_1k:.2f} ms")
    print(f"N=10,000 : {l_10k:.2f} ms")
