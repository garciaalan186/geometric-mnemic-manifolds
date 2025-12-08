import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gmm.core.manifold import GeometricMnemicManifold

def verify():
    print("Initializing GMM with foveal_beta=1.2")
    gmm = GeometricMnemicManifold(foveal_beta=1.2, embedding_dim=16)
    dummy_embed = np.zeros(16)

    print("\n--- Adding 100 engrams ---")
    for i in range(100):
       gmm.add_engram(f"ctx {i}", dummy_embed)

    indices = gmm.fovea_indices
    print(f"Indices (N=100): {indices}")
    print(f"Count: {len(indices)}")

    if 0 not in indices:
        raise AssertionError("Oldest index 0 missing from fovea")
    if 99 not in indices:
        raise AssertionError("Newest index 99 missing from fovea")
    
    # Check gap calculation
    # Oldest gap (indices[1] - indices[0]) should be larger than newest gap (indices[-1] - indices[-2])
    gap_old = indices[1] - indices[0]
    gap_new = indices[-1] - indices[-2]
    print(f"Gap Oldest: {gap_old}")
    print(f"Gap Newest: {gap_new}")
    
    if gap_old <= gap_new:
         print("WARNING: Gap density check might be flaky for small N if distribution is flat")

    print("\n--- Adding 900 more (N=1000) ---")
    for i in range(900):
       gmm.add_engram(f"ctx {i+100}", dummy_embed)
       
    indices = gmm.fovea_indices
    print(f"Indices (N=1000): {indices}")
    print(f"Count: {len(indices)}")
    
    assert 0 in indices
    assert 999 in indices
    
    # Check strict density assumption
    gap_old = indices[1] - indices[0]
    gap_new = indices[-1] - indices[-2]
    
    if gap_old <= gap_new:
        raise AssertionError(f"Distribution failed to decay! Old gap {gap_old} <= New gap {gap_new}")
        
    print(f"Distribution Confirmed: Oldest Gap ({gap_old}) > Newest Gap ({gap_new})")
    print("Verification Passed!")

if __name__ == "__main__":
    verify()
