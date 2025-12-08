
import requests
import time
import numpy as np
import threading
import sys
import psutil
from concurrent.futures import ThreadPoolExecutor

# Config
N_SHARDS = 8
SHARD_DNS_TEMPLATE = "http://gmm-shards-{id}.gmm-service:5000"
TOTAL_ITEMS = 4000000 # 4M Total
DIM = 128
ITEMS_PER_SHARD = TOTAL_ITEMS // N_SHARDS

def get_shard_url(i):
    # In K8s, StatefulSet pods are reachable via DNS
    # If running locally (for testing), use localhost:5000+i ?
    # We assume this runs INSIDE K8s.
    return SHARD_DNS_TEMPLATE.format(id=i)

def load_shard(i, n_items):
    url = f"{get_shard_url(i)}/setup"
    try:
        resp = requests.post(url, json={"n": n_items, "dim": DIM}, timeout=600)
        return resp.status_code == 200
    except Exception as e:
        print(f"Shard {i} Load Error: {e}")
        return False

def query_shard(i, vector, algo="gmm"):
    url = f"{get_shard_url(i)}/query_{algo}"
    try:
        start = time.time()
        resp = requests.post(url, json={"vector": vector.tolist(), "k": 10}, timeout=5)
        lat_network = (time.time() - start) * 1000
        data = resp.json()
        return data.get('lat_ms', 0), lat_network
    except Exception as e:
        print(f"Shard {i} Query Error: {e}")
        return 0, 0

def wait_for_shards():
    print(f"Waiting for {N_SHARDS} shards to be ready...")
    ready_shards = set()
    start_wait = time.time()
    
    while len(ready_shards) < N_SHARDS:
        if time.time() - start_wait > 600: # 10 min timeout
            raise TimeoutError("Shards did not become ready in time.")
            
        for i in range(N_SHARDS):
            if i in ready_shards:
                continue
                
            try:
                url = f"{get_shard_url(i)}/health"
                resp = requests.get(url, timeout=2)
                if resp.status_code == 200:
                    print(f"Shard {i} is READY.")
                    ready_shards.add(i)
            except:
                pass
        
        if len(ready_shards) < N_SHARDS:
            time.sleep(5)
    print("All shards are READY.")

def run_benchmark():
    print(f"Starting V2 Benchmark (N={TOTAL_ITEMS}, Shards={N_SHARDS})")
    
    # Wait for infrastructure
    wait_for_shards()
    
    # 1. Setup Phase
    print("----------------------------------------------------------------")
    print("PHASE 1: Loading Data & Building Indices (HNSW takes time)...")
    start_load = time.time()
    with ThreadPoolExecutor(max_workers=N_SHARDS) as executor:
        futures = [executor.submit(load_shard, i, ITEMS_PER_SHARD) for i in range(N_SHARDS)]
        results = [f.result() for f in futures]
    
    if not all(results):
        print("CRITICAL: Failed to load some shards!")
        return
        
    print(f"Load Complete in {time.time() - start_load:.2f}s")
    
    # 2. Benchmark Phase
    query_vector = np.random.rand(DIM).astype(np.float32)
    
    print("----------------------------------------------------------------")
    print("PHASE 2: GMM Broadcast Latency (Linear Scan)")
    # GMM Architecture: Broadcast to all, merge top K.
    # Latency = Max(Shard Latencies) + Network RTT + Coordinator Merge (negligible)
    
    latencies_gmm = []
    gmm_internal = []
    for _ in range(20):
        start_req = time.time()
        with ThreadPoolExecutor(max_workers=N_SHARDS) as executor:
            futures = [executor.submit(query_shard, i, query_vector, "gmm") for i in range(N_SHARDS)]
            # We wait for ALL to complete (Broadcast & Gather)
            shard_results = [f.result() for f in futures]
        
        total_lat = (time.time() - start_req) * 1000
        latencies_gmm.append(total_lat)
        
        # Calculate max internal compute time
        max_internal = max([r[0] for r in shard_results])
        gmm_internal.append(max_internal)
        print(f"  GMM Query: {total_lat:.2f}ms (Max Internal: {max_internal:.2f}ms)")
        
    avg_gmm = sum(latencies_gmm) / len(latencies_gmm)
    avg_gmm_internal = sum(gmm_internal) / len(gmm_internal)
    
    print("----------------------------------------------------------------")
    print("PHASE 3: HNSW Scatter-Gather Latency")
    # HNSW Architecture: Scatter to all, merge top K (Standard ElasticSearch/Milvus approach)
    # Note: True distributed HNSW graph traversal is WORSE than this. 
    # This is the "Best Case" HNSW Distributed implementation (Partitioned Index).
    
    latencies_hnsw = []
    hnsw_internal = []
    for _ in range(20):
        start_req = time.time()
        with ThreadPoolExecutor(max_workers=N_SHARDS) as executor:
            futures = [executor.submit(query_shard, i, query_vector, "hnsw") for i in range(N_SHARDS)]
            shard_results = [f.result() for f in futures]
            
        total_lat = (time.time() - start_req) * 1000
        latencies_hnsw.append(total_lat)
        max_internal = max([r[0] for r in shard_results])
        hnsw_internal.append(max_internal)
        print(f"  HNSW Query: {total_lat:.2f}ms (Max Internal: {max_internal:.2f}ms)")

    avg_hnsw = sum(latencies_hnsw) / len(latencies_hnsw)
    avg_hnsw_internal = sum(hnsw_internal) / len(hnsw_internal)
    
    print("================================================================")
    print("V2 CLOUD BENCHMARK RESULTS")
    print(f"Scale: {TOTAL_ITEMS} Vectors (8 Shards)")
    print("----------------------------------------------------------------")
    print(f"GMM Latency  : {avg_gmm:.2f} ms (Compute: {avg_gmm_internal:.2f} ms | Network: {avg_gmm - avg_gmm_internal:.2f} ms)")
    print(f"HNSW Latency : {avg_hnsw:.2f} ms (Compute: {avg_hnsw_internal:.2f} ms | Network: {avg_hnsw - avg_hnsw_internal:.2f} ms)")
    print("----------------------------------------------------------------")
    
    # Save to file
    with open("/app/results.txt", "w") as f:
        f.write(f"GMM:{avg_gmm}\n")
        f.write(f"HNSW:{avg_hnsw}\n")

if __name__ == "__main__":
    # Give pods time to start
    print("Waiting for shards to warm up...")
    time.sleep(20) 
    run_benchmark()
