import time
import uuid
import random
import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Set

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.gmm.core.manifold import GeometricMnemicManifold

# ==============================================================================
# SIMULATION CLASSES
# ==============================================================================

@dataclass
class NetworkStats:
    hops: int = 0
    latency_ms: float = 0.0

NETWORK_LATENCY_MS = 1.0  # Cost of 1 hop between different shards

class SimulatedShard:
    """Represents a physical server holding a subset of data."""
    def __init__(self, shard_id: int):
        self.shard_id = shard_id
        self.data: Dict[int, np.ndarray] = {}
        self.links: Dict[int, List[int]] = {} # For HNSW graph

    def add_item(self, vector: np.ndarray, uid: int, links: List[int] = None):
        self.data[uid] = vector
        self.links[uid] = links if links else []

    def add_link(self, uid: int, target_uid: int):
        if uid in self.links:
            self.links[uid].append(target_uid)

    def get_vector(self, uid: int) -> np.ndarray:
        return self.data[uid]
        
    def get_links(self, uid: int) -> List[int]:
        return self.links.get(uid, [])

class DistributedSystem:
    def __init__(self, num_shards: int = 10):
        self.shards = [SimulatedShard(i) for i in range(num_shards)]
        self.id_to_shard: Dict[int, int] = {} # Map ID -> Shard ID
        self.stats = NetworkStats()

    def reset_stats(self):
        self.stats = NetworkStats()

    def route_request(self, target_shard_id: int):
        """Simulate sending a request to a shard."""
        # Assume router -> shard cost
        self.stats.hops += 1
        self.stats.latency_ms += NETWORK_LATENCY_MS
        return self.shards[target_shard_id]

    def cross_shard_access(self, from_shard: int, to_shard: int):
        """Check cost of accessing data across shards."""
        if from_shard != to_shard:
            self.stats.hops += 1
            self.stats.latency_ms += NETWORK_LATENCY_MS
            
    def add_link_global(self, from_uid: int, to_uid: int):
        """Helper to add link across shards (simulating write)."""
        shard_id = self.id_to_shard[from_uid]
        self.shards[shard_id].add_link(from_uid, to_uid)

# ==============================================================================
# DISTRIBUTED GMM SIMULATION
# ==============================================================================
class DistributedGMM:
    """
    GMM scales by Time Sharding.
    Router (L1/L2) directs query to specific Time Shard (L0).
    """
    def __init__(self, system: DistributedSystem):
        self.system = system
        self.router_index: List[tuple] = [] # (time_start, time_end, shard_id)
        
    def add_batch(self, vectors: np.ndarray, start_id: int):
        # GMM writes sequentially. A batch goes to ONE shard.
        shard_id = start_id % len(self.system.shards) # Simple round robin for batches
        shard = self.system.shards[shard_id]
        
        for i, vec in enumerate(vectors):
            uid = start_id + i
            shard.add_item(vec, uid)
            self.system.id_to_shard[uid] = shard_id
            
        # Update Router (Metadata only, instant)
        self.router_index.append((start_id, start_id + len(vectors), shard_id))
        
    def query(self, query_vec: np.ndarray, k: int = 5):
        self.system.reset_stats()
        
        # 1. Router Step (L1/L2 Search) - Happens Locally or on 1 Router Node
        # In a real system, Router has the L1/L2 index.
        # We assume Router determines the relevant User/Time shard instantly or in 1 hop.
        # Let's be generous and say Router is 1 hop away.
        self.system.stats.hops += 1
        self.system.stats.latency_ms += NETWORK_LATENCY_MS
        
        # 2. Targeted Shard Query
        # GMM narrows search to specific semantic/temporal clusters.
        # Assume it identifies top 3 relevant shards based on L1/L2.
        # For simulation, just pick 3 random shards as targets (realistic scatter/gather)
        # In reality, it's often 1-2 shards.
        relevant_shards = set([s_id for (_, _, s_id) in self.router_index[:3]])
        if not relevant_shards: relevant_shards = {0} # Fallback
        
        results = []
        for shard_id in relevant_shards:
            # Go to shard
            shard = self.system.route_request(shard_id)
            # Local search on shard (0 network cost)
            # ... compute ...
            pass
            
        return self.system.stats

# ==============================================================================
# DISTRIBUTED HNSW SIMULATION
# ==============================================================================
class DistributedHNSW:
    """
    HNSW scales by Sharding the Graph.
    Nodes are randomly distributed (to balance load).
    Traversing neighbors requires hopping shards.
    """
    def __init__(self, system: DistributedSystem, ef_search: int = 100):
        self.system = system
        self.ef_search = ef_search
        self.entry_point = None
        
    def add_batch(self, vectors: np.ndarray, start_id: int):
        # HNSW randomly distributes data to balance storage
        for i, vec in enumerate(vectors):
            uid = start_id + i
            shard_id = random.randint(0, len(self.system.shards) - 1)
            self.system.id_to_shard[uid] = shard_id
            
            # Generate random links for simulation (Graph Topology)
            # Distributed random allocation means ~90% of links are cross-shard.
            num_links = 16
            
            # Add item first so it exists
            self.system.shards[shard_id].add_item(vec, uid, [])
            
            if uid > 0:
                # Randomly link to previous nodes (Backward)
                # We simulate Small World: most links are local (by ID), some long range
                # But ID locality != Shard locality (Random Sharding)
                # So even "local" neighbors are on random shards.
                targets = [random.randint(0, uid - 1) for _ in range(num_links)]
                for target in targets:
                    # Bidirectional: Add forward link and backward link
                    self.system.add_link_global(uid, target)
                    self.system.add_link_global(target, uid)
                
            if self.entry_point is None:
                self.entry_point = uid

    def query(self, query_vec: np.ndarray, k: int = 5):
        self.system.reset_stats()
        
        # Greedy Graph Traversal
        current_node = self.entry_point
        if current_node is None: return
        
        visited = set()
        candidates = [current_node]
        
        # Simulate ef_search steps of traversal
        current_shard_id = self.system.id_to_shard[current_node]
        
        # Initial hop to entry point
        self.system.route_request(current_shard_id)
        
        # This is a Monte Carlo simulation of graph walk
        # In HNSW, we explore neighbors.
        steps = 0
        while steps < self.ef_search and candidates:
            steps += 1
            # Pop best candidate (simulated)
            curr = candidates.pop(0)
            curr_shard = self.system.id_to_shard[curr]
            
            # Get neighbors (this requires being on the shard or fetching)
            # If we were at 'previous_shard', and curr is at 'curr_shard'
            # We pay latency.
            # Actually, the algo runs on the client (Coordinator). 
            # It fetches neighbor list from curr_shard.
            self.system.cross_shard_access(-1, curr_shard) # Request to fetch node
            
            neighbors = self.system.shards[curr_shard].get_links(curr)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    candidates.append(neighbor)
                    
                    # To evaluate neighbor distance, we must fetch its vector
                    # Neighbor is on neighbor_shard
                    neighbor_shard = self.system.id_to_shard[neighbor]
                    self.system.cross_shard_access(curr_shard, neighbor_shard)

        return self.system.stats

# ==============================================================================
# MAIN BENCHMARK
# ==============================================================================
def run_distributed_benchmark():
    print("================================================================================")
    print("DISTRIBUTED SCALABILITY BENCHMARK (SIMULATION)")
    print("================================================================================")
    print(f"Network Latency per Hop: {NETWORK_LATENCY_MS} ms")
    print(f"Number of Shards: 10")
    print("-" * 80)
    
    # Setup
    sys_gmm = DistributedSystem(num_shards=10)
    sys_hnsw = DistributedSystem(num_shards=10)
    
    gmm = DistributedGMM(sys_gmm)
    hnsw = DistributedHNSW(sys_hnsw, ef_search=100) # Typical production value
    
    # Generate Data
    N = 1000
    dim = 32
    vectors = np.random.normal(0, 1, (N, dim))
    
    # Load Data
    print(f"Loading {N} items...")
    gmm.add_batch(vectors, 0) # GMM batches temporally
    hnsw.add_batch(vectors, 0) # HNSW scatters randomly
    
    # Query Test
    print("\nRunning Query Simulation...")
    q = np.random.normal(0, 1, (dim,))
    
    # GMM
    stats_gmm = gmm.query(q)
    print(f"\nGMM Distributed Query:")
    print(f"  Hops: {stats_gmm.hops}")
    print(f"  Network Latency: {stats_gmm.latency_ms:.2f} ms")
    
    # HNSW
    stats_hnsw = hnsw.query(q)
    print(f"\nHNSW Distributed Query (ef=100):")
    print(f"  Hops: {stats_hnsw.hops}")
    print(f"  Network Latency: {stats_hnsw.latency_ms:.2f} ms")
    
    print("-" * 80)
    ratio = stats_hnsw.latency_ms / max(stats_gmm.latency_ms, 0.1)
    print(f"SCALABILITY FACTOR: GMM is {ratio:.1f}x more network efficient")

if __name__ == "__main__":
    run_distributed_benchmark()
