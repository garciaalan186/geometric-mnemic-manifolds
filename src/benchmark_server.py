
import os
import numpy as np
import hnswlib
from flask import Flask, request, jsonify
import time

app = Flask(__name__)

# Global State
class ShardState:
    def __init__(self):
        self.data = None
        self.hnsw_index = None
        self.dim = 128
        self.n = 0
        self.gmm_matrix = None

state = ShardState()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "n": state.n})

@app.route('/setup', methods=['POST'])
def setup():
    """Generates random data and builds indices."""
    try:
        req = request.json
        n_items = req.get('n', 100000)
        dim = req.get('dim', 128)
        
        state.dim = dim
        state.n = n_items
        
        print(f"Generating {n_items} vectors (d={dim})...")
        # 1. Generate Data
        state.data = np.random.rand(n_items, dim).astype(np.float32)
        
        # 2. Build GMM (Matrix)
        # GMM 'index' is just the raw matrix for scan
        state.gmm_matrix = state.data
        
        # 3. Build HNSW
        print("Building HNSW Index...")
        p = hnswlib.Index(space='cosine', dim=dim)
        p.init_index(max_elements=n_items, ef_construction=200, M=16)
        p.add_items(state.data, np.arange(n_items))
        p.set_ef(50) # Search ef
        state.hnsw_index = p
        
        print("Setup Complete.")
        return jsonify({"status": "ready", "n": n_items})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/query_gmm', methods=['POST'])
def query_gmm():
    """Simulates GMM Shard Scan (Linear)."""
    if state.gmm_matrix is None:
        return jsonify({"error": "Not initialized"}), 400
        
    start = time.time()
    vector = np.array(request.json['vector'], dtype=np.float32)
    
    # Cosine Similarity: dot product (assume normalized or raw dot)
    # We use raw dot for speed benchmark as per main loop
    scores = np.dot(state.gmm_matrix, vector)
    
    # Top K
    k = request.json.get('k', 10)
    top_indices = np.argpartition(scores, -k)[-k:]
    # Sort
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    
    duration = (time.time() - start) * 1000
    return jsonify({
        "lat_ms": duration,
        "results": top_indices.tolist()
    })

@app.route('/query_hnsw', methods=['POST'])
def query_hnsw():
    """Simulates HNSW Shard Search."""
    if state.hnsw_index is None:
        return jsonify({"error": "Not initialized"}), 400
        
    start = time.time()
    vector = np.array(request.json['vector'], dtype=np.float32)
    k = request.json.get('k', 10)
    
    labels, distances = state.hnsw_index.knn_query(vector, k=k)
    
    duration = (time.time() - start) * 1000
    return jsonify({
        "lat_ms": duration,
        "results": labels[0].tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
