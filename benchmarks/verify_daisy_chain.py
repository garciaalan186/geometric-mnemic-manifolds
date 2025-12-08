import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gmm.core.manifold import GeometricMnemicManifold

def verify():
    print("Initializing GMM...")
    gmm = GeometricMnemicManifold(embedding_dim=16)
    dummy_embed = np.zeros(16)

    # 1. Add First Engram
    print("\nAdding Engram 0: 'The quick brown fox jumps over the lazy dog'")
    e0 = gmm.add_engram("The quick brown fox jumps over the lazy dog", dummy_embed)
    
    # Check it has no prefix
    print(f"Stored E0: '{e0.context_window}'")
    if "[PREV:" in e0.context_window:
        raise AssertionError("First engram should not have PREV prefix")
        
    # 2. Add Second Engram
    print("\nAdding Engram 1: 'It was a sunny day'")
    e1 = gmm.add_engram("It was a sunny day", dummy_embed)
    
    # Check it has prefix
    print(f"Stored E1: '{e1.context_window}'")
    if "[PREV:" not in e1.context_window:
        raise AssertionError("Second engram must have [PREV: ...] prefix")
        
    # Check seed content (telegraphic summary of fox string)
    # Expected: "quick brown fox jumps lazy dog" (Stop words removed)
    if "fox" not in e1.context_window or "dog" not in e1.context_window:
         raise AssertionError("Seed context missing key entities from E0")
         
    # Check metadata linkage
    if e1.metadata.get('previous_id') != e0.id:
        raise AssertionError(f"Metadata linkage failed. Expected {e0.id}, got {e1.metadata.get('previous_id')}")
        
    print("\nSuccess! Daisy chaining verified.")

if __name__ == "__main__":
    verify()
