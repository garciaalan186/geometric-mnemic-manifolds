
import time
import multiprocessing
import statistics
import numpy as np
import psutil
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Configuration
TOTAL_VECTORS = 1000000 # 1 Million Vectors
VECTOR_DIM = 128
NUM_SHARDS = 10
NUM_RUNS = 10

class BenchmarkWorker(multiprocessing.Process):
    """
    Persistent Worker simulating a shard server.
    Data is resident in memory (passed via constructor).
    """
    def __init__(self, shard_id, shard_data, req_queue, resp_queue):
        super().__init__()
        self.shard_id = shard_id
        # Data is copied to the process memory space upon start
        self.shard_data = shard_data 
        self.req_queue = req_queue
        self.resp_queue = resp_queue
        self.daemon = True # Auto-kill if main dies

    def run(self):
        # Worker Event Loop
        while True:
            query_vector = self.req_queue.get()
            if query_vector is None:
                break # Poison pill
            
            # REAL COMPUTE ON RESIDENT DATA
            # No IPC cost for 'self.shard_data' here. 
            # Only 'query_vector' (small) came over IPC.
            scores = np.dot(self.shard_data, query_vector)
            
            # Find Top 10 (simulated index work)
            # In GMM this would be finding max pattern matches
            top_k = np.argpartition(scores, -10)[-10:]
            
            self.resp_queue.put(top_k.tolist())

def run_mono_benchmark(all_vectors, query_vector):
    """Run workload on a single process (Linear Scan)."""
    start = time.time()
    scores = np.dot(all_vectors, query_vector)
    top_k = np.argpartition(scores, -10)[-10:]
    return (time.time() - start) * 1000

def run_robust_benchmark():
    print("================================================================================")
    print("ROBUST REAL-COMPUTE BENCHMARK (RESIDENT MEMORY)")
    print("================================================================================")
    print(f"Vectors: {TOTAL_VECTORS}")
    print(f"Dimensions: {VECTOR_DIM}")
    print(f"Shards (Processes): {NUM_SHARDS}")
    print(f"CPU Cores Available: {multiprocessing.cpu_count()}")
    print("-" * 80)
    
    # 1. Generate Data
    print("Generating Synthetic Data (1M Vectors)...")
    all_vectors = np.random.rand(TOTAL_VECTORS, VECTOR_DIM).astype(np.float32)
    query_vector = np.random.rand(VECTOR_DIM).astype(np.float32)
    
    # 2. Warmup Mono
    print("Warming up Mono...")
    run_mono_benchmark(all_vectors[:1000], query_vector)
    
    # 3. Run Mono Benchmark
    print(f"Running Monolithic Scan (N={TOTAL_VECTORS})...")
    mono_times = []
    for _ in range(NUM_RUNS):
        mono_times.append(run_mono_benchmark(all_vectors, query_vector))
    avg_mono = statistics.mean(mono_times)
    print(f"  Avg Latency: {avg_mono:.2f} ms")
    print(f"  Throughput: {1000/avg_mono:.1f} QPS")
    
    # 4. Initialize Parallel Workers
    print(f"\nBooting {NUM_SHARDS} Persistent Shard Workers...")
    shards_data = np.array_split(all_vectors, NUM_SHARDS)
    workers = []
    req_queues = []
    resp_queues = []
    
    for i in range(NUM_SHARDS):
        req_q = multiprocessing.Queue()
        resp_q = multiprocessing.Queue()
        # Pass shard data to constructor -> Copied to worker
        w = BenchmarkWorker(i, shards_data[i], req_q, resp_q)
        w.start()
        workers.append(w)
        req_queues.append(req_q)
        resp_queues.append(resp_q)
        
    time.sleep(1) # Let them settle
    
    # 5. Run Parallel Benchmark
    print(f"Running Parallel Broadcast (N={TOTAL_VECTORS})...")
    par_times = []
    
    try:
        for _ in range(NUM_RUNS):
            start = time.time()
            
            # Broadcast Query
            for q in req_queues:
                q.put(query_vector)
                
            # Gather Results
            for q in resp_queues:
                _ = q.get()
                
            lat = (time.time() - start) * 1000
            par_times.append(lat)
            
        avg_par = statistics.mean(par_times)
        print(f"  Avg Latency: {avg_par:.2f} ms")
        print(f"  Throughput: {1000/avg_par:.1f} QPS")
        
        # 6. Analysis
        print("-" * 80)
        print("ANALYSIS:")
        speedup = avg_mono / avg_par
        print(f"Speedup: {speedup:.1f}x")
        
        # Resources
        process = psutil.Process(os.getpid())
        print("-" * 80)
        print("RESOURCE USAGE (Coordinator):")
        print(f"CPU Usage: {psutil.cpu_percent(interval=1.0)}%")
        print(f"Memory Usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        print("-" * 80)
        
    finally:
        # Cleanup
        for q in req_queues:
            q.put(None)
        for w in workers:
            w.join()

if __name__ == "__main__":
    # Ensure safe cloning
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass
    run_robust_benchmark()
