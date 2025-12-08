
import time
import threading
import queue
import random
import statistics
import numpy as np

# Configuration
NUM_SHARDS = 10
NETWORK_DELAY_S = 0.005  # 5ms simulated network/RPC overhead per request
NUM_QUERIES = 50

class ShardThread(threading.Thread):
    """
    Simulates a database shard running on a separate thread/server.
    """
    def __init__(self, shard_id, request_queue, response_queue):
        super().__init__()
        self.shard_id = shard_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.running = True
        self.data_count = 0

    def run(self):
        while self.running:
            try:
                # Wait for request
                req = self.request_queue.get(timeout=0.1)
                
                if req == "STOP":
                    break
                    
                # Simulate Network + RPC processing time
                time.sleep(NETWORK_DELAY_S)
                
                # Mock Compute (very fast relative to network)
                # In real life, HNSW search is ~2ms.
                # We add a small jitter to simulate "Tail Latency"
                processing_time = 0.002 + random.uniform(0, 0.002) 
                time.sleep(processing_time)
                
                # Send response
                self.response_queue.put({
                    'shard_id': self.shard_id,
                    'result': 'top_k_results',
                    'latency': processing_time
                })
                
                self.request_queue.task_done()
                
            except queue.Empty:
                continue

class BenchmarkSystem:
    def __init__(self, num_shards=10):
        self.shards = []
        self.req_queues = []
        self.resp_queues = []
        
        for i in range(num_shards):
            req_q = queue.Queue()
            resp_q = queue.Queue()
            t = ShardThread(i, req_q, resp_q)
            self.shards.append(t)
            self.req_queues.append(req_q)
            self.resp_queues.append(resp_q)
            t.start()
            
    def shutdown(self):
        for q in self.req_queues:
            q.put("STOP")
        for t in self.shards:
            t.join()

    def query_gmm_unicast(self):
        """
        GMM Model: Router identifies ONE target shard and sends query there.
        """
        start = time.time()
        
        # Router logic (instant lookup)
        target_shard = random.randint(0, len(self.shards) - 1)
        
        # Send Request
        self.req_queues[target_shard].put("QUERY")
        
        # Wait for Response
        _ = self.resp_queues[target_shard].get()
        
        return (time.time() - start) * 1000 # ms

    def query_hnsw_broadcast(self):
        """
        Distributed HNSW Model: Scatter-Gather to ALL shards.
        Latency is determined by the SLOWEST shard (tail latency).
        """
        start = time.time()
        
        # Broadcast
        for q in self.req_queues:
            q.put("QUERY")
            
        # Gather (Wait for ALL)
        for q in self.resp_queues:
            _ = q.get()
            
        return (time.time() - start) * 1000 # ms

def run_concurrent_benchmark():
    print("================================================================================")
    print("CONCURRENT SCALABILITY BENCHMARK (REAL THREADS)")
    print("================================================================================")
    print(f"Shards: {NUM_SHARDS}")
    print(f"Network Cost: {NETWORK_DELAY_S*1000:.1f} ms")
    print(f"Compute Cost: ~3.0 ms (jittered)")
    print("-" * 80)
    
    # Initialize System with 10 Shards
    print("Initializing 10 Shard Threads...")
    system = BenchmarkSystem(NUM_SHARDS)
    
    # Initialize Mono System (1 Big Shard)
    print("Initializing Monolithic Baseline...")
    mono_system = BenchmarkSystem(1)
    
    time.sleep(1) # Warmup
    print("\nRunning Benchmarks...")
    
    try:
        # 1. GMM
        gmm_latencies = []
        for _ in range(NUM_QUERIES):
            lat = system.query_gmm_unicast()
            gmm_latencies.append(lat)
        print(f"GMM (Unicast) Avg Latency: {statistics.mean(gmm_latencies):.2f} ms")
        
        # 2. Distributed HNSW
        d_hnsw_latencies = []
        for _ in range(NUM_QUERIES):
            lat = system.query_hnsw_broadcast()
            d_hnsw_latencies.append(lat)
        print(f"Dist. HNSW (Broadcast) Avg Latency: {statistics.mean(d_hnsw_latencies):.2f} ms")
            
        # 3. Monolithic HNSW
        m_hnsw_latencies = []
        for _ in range(NUM_QUERIES):
            # Mono system has only 1 shard (index 0)
            target = 0
            start = time.time()
            mono_system.req_queues[target].put("QUERY")
            _ = mono_system.resp_queues[target].get()
            lat = (time.time() - start) * 1000
            m_hnsw_latencies.append(lat)
        print(f"Mono HNSW (Baseline) Avg Latency: {statistics.mean(m_hnsw_latencies):.2f} ms")

        # Analysis
        print("-" * 80)
        print("ANALYSIS:")
        print(f"1. GMM vs Dist. HNSW: GMM is {statistics.mean(d_hnsw_latencies)/statistics.mean(gmm_latencies):.1f}x Faster")
        print("   (Reason: Unicast avoids waiting for the slowest of 10 shards)")
        
        print(f"2. Mono vs Dist HNSW: Mono is {statistics.mean(d_hnsw_latencies)/statistics.mean(m_hnsw_latencies):.1f}x Faster")
        print("   (Reason: Scatter-Gather overhead dominates simple requests)")

    finally:
        system.shutdown()
        mono_system.shutdown()

if __name__ == "__main__":
    run_concurrent_benchmark()
