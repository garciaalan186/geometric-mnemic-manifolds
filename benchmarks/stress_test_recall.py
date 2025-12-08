import time
import uuid
import random
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.gmm.core.manifold import GeometricMnemicManifold
from benchmarks.baselines.hnsw import HNSWIndex

def run_stress_test():
    """
    Incrementally add data to stress test recall.
    """
    print("================================================================================")
    print("HNSW RECALL STRESS TEST")
    print("================================================================================")
    
    embedding_dim = 384
    
    # Initialize systems
    # GMM optimized params
    gmm = GeometricMnemicManifold(
        embedding_dim=embedding_dim,
        lambda_decay=0.0014, # From optimization
        beta1=64,
        beta2=16,
        foveal_beta=1.1
    )
    
    # HNSW baseline params
    hnsw = HNSWIndex(embedding_dim=embedding_dim, m=16, ef_construction=200)
    
    current_size = 0
    step_size = 5000
    max_size = 50000 # Cap at 50k to be reasonable with time
    
    print(f"Starting stress test. Step size: {step_size}. Max size: {max_size}")
    print("Depth\t| GMM Recall\t| HNSW Recall\t| GMM Time\t| HNSW Time")
    print("-" * 80)
    
    rng = np.random.default_rng(42)
    
    while current_size < max_size:
        # 1. Generate noise batch
        # print(f"Adding {step_size} items...")
        batch_embeddings = rng.normal(0, 1, (step_size, embedding_dim))
        # Normalize
        norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
        batch_embeddings = batch_embeddings / norms
        
        # 2. Add to systems
        for i in range(step_size):
            emb = batch_embeddings[i]
            idx = current_size + i
            
            # GMM
            gmm.add_engram(f"Noise {idx}", emb)
            
            # HNSW
            hnsw.add_item(emb, idx)
            
        current_size += step_size
        
        # 3. Insert Needle
        passkey = str(uuid.uuid4())
        needle_embedding = rng.normal(0, 1, (embedding_dim,))
        needle_embedding = needle_embedding / np.linalg.norm(needle_embedding)
        
        # GMM add needle
        needle_engram = gmm.add_engram(f"Passkey: {passkey}", needle_embedding)
        # HNSW add needle (ID = current_size, effectively)
        hnsw_id = current_size # The needle is at index current_size (0-indexed + 1? No, just next ID)
        hnsw.add_item(needle_embedding, hnsw_id)
        
        # 4. Query
        # Pre-warm GMM matrices
        gmm._update_matrices()
        
        # GMM Query
        start = time.time()
        gmm_results = gmm.query(needle_embedding, k=1)
        gmm_time = (time.time() - start) * 1000
        
        gmm_recall = False
        if gmm_results and gmm_results[0][0].context_window == f"Passkey: {passkey}":
            gmm_recall = True
            
        # HNSW Query
        start = time.time()
        hnsw_results = hnsw.search(needle_embedding, k=1)
        hnsw_time = (time.time() - start) * 1000
        
        hnsw_recall = False
        if hnsw_results and hnsw_results[0][0] == hnsw_id:
            hnsw_recall = True
            
        print(f"{current_size}\t| {gmm_recall}\t\t| {hnsw_recall}\t\t| {gmm_time:.2f}ms\t| {hnsw_time:.2f}ms")
        
        if not hnsw_recall:
            print("!!! HNSW RECALL FAILURE DETECTED !!!")
            break
            
        # Clean up needle? No, just leave it as noise for next round.
        # But we need to increment current_size by 1 (the needle) so indices don't overlap?
        # Actually indices in GMM layer0 are list indices.
        # Indices in HNSW are whatever we pass.
        # The loop uses `current_size + i`.
        # If we added 5000 items, max index used was current_size + 4999.
        # Needle used current_size + 5000 (effectively).
        # Next batch starts at new current_size.
        # We need to account for needle in size.
        current_size += 1

if __name__ == "__main__":
    run_stress_test()
