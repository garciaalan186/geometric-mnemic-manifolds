# Reproducibility Guide

This guide provides detailed instructions for reproducing all results presented in the Geometric Mnemic Manifolds research.

## 🎯 Reproducibility Goals

This research is designed to be fully reproducible:
- ✅ Deterministic algorithms with fixed random seeds
- ✅ Documented software versions
- ✅ Synthetic data generation (no data contamination)
- ✅ Open-source implementation
- ✅ Comprehensive parameter documentation

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum
- **Storage**: 500MB for repository and outputs

### Recommended for Full Benchmarks
- **RAM**: 8GB+
- **CPU**: Multi-core processor for faster benchmarks
- **Storage**: 2GB for large-scale experiments

### Tested Environments

The code has been verified on:

```
- Ubuntu 22.04 LTS, Python 3.10.12, NumPy 1.24.3, Matplotlib 3.7.1
- macOS Sonoma 14.1, Python 3.11.5, NumPy 1.26.0, Matplotlib 3.8.0
- Windows 11, Python 3.9.13, NumPy 1.23.5, Matplotlib 3.6.2
```

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/garciaalan186/geometric-mnemic-manifolds.git
cd geometric-mnemic-manifolds
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv gmm_env
source gmm_env/bin/activate  # On Windows: gmm_env\Scripts\activate

# Or using conda
conda create -n gmm python=3.10
conda activate gmm
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Exact versions used in paper:**
```
numpy==1.24.3
matplotlib==3.7.1
```

## 🔬 Reproducing Core Results

### Experiment 1: Prototype Demonstration

**Reproduces**: Figure 1, Table 1 (Manifold Statistics)

```bash
python gmm_prototype.py
```

**Expected Output:**
- 500 synthetic life events generated
- 3-layer hierarchical structure built
- Layer 0: 500 engrams
- Layer 1: 8 engrams (64:1 compression)
- Layer 2: 1 engram (512:1 compression)
- Visualization data saved to `gmm_visualization.json`

**Random Seed**: Fixed at 42 in `SyntheticBiographyGenerator`

**Runtime**: ~2-5 seconds on standard hardware

### Experiment 2: Needle in the Spiral Benchmark

**Reproduces**: Figure 2, Table 2 (Performance Comparison)

```bash
python benchmark_suite.py
```

**Parameters** (as specified in paper):
- Memory depths: [100, 500, 1000, 5000]
- Trials per depth: 5
- Embedding dimension: 128
- Passkey position: 50% depth

**Expected Results:**
```
Depth    100: GMM ~5.0ms, HNSW ~7.0ms, Speedup ~1.4x
Depth    500: GMM ~5.5ms, HNSW ~12.0ms, Speedup ~2.2x
Depth  1,000: GMM ~6.0ms, HNSW ~16.0ms, Speedup ~2.7x
Depth  5,000: GMM ~6.5ms, HNSW ~24.0ms, Speedup ~3.7x
```

**Output Files:**
- `benchmark_results/benchmark_results.png` - Performance plots
- `benchmark_results/benchmark_results.json` - Raw data

**Runtime**: 2-5 minutes depending on hardware

**Note**: HNSW times are simulated based on theoretical O(log N) complexity. For comparison with actual HNSW implementation, see Extended Benchmarks below.

### Experiment 3: Interactive Demonstrations

**Reproduces**: Conceptual demonstrations from Section 4

```bash
python demo.py
```

Select option 8 to run all demonstrations sequentially.

**Demonstrations:**
1. Spiral Geometry - Shows Kronecker positioning
2. Hierarchical Layers - Visualizes compression ratios
3. Memory Retrieval - Demonstrates query algorithm
4. Foveal Ring - Shows working memory concept
5. Temporal Foveation - Illustrates density decay
6. Cold Start - Compares initialization times
7. Interactive Queries - Shows retrieval in action

## 🔍 Parameter Sensitivity Analysis

### Key Parameters and Their Effects

**λ (Lambda Decay)**: Controls radial expansion rate
```python
# Default: 0.01
gmm = GeometricMnemicManifold(lambda_decay=0.01)

# Test range: [0.001, 0.05]
# Lower values: Slower expansion, denser spiral
# Higher values: Faster expansion, sparser spiral
```

**β₁ (Layer 1 Compression Ratio)**: Pattern layer granularity
```python
# Default: 64
gmm = GeometricMnemicManifold(beta1=64)

# Test range: [32, 128]
# Lower values: More pattern nodes, finer granularity
# Higher values: Fewer pattern nodes, coarser granularity
```

**β₂ (Layer 2 Compression Ratio)**: Abstract layer granularity
```python
# Default: 16
gmm = GeometricMnemicManifold(beta2=16)

# Test range: [8, 32]
# Similar trade-offs as β₁
```

### Running Parameter Sweeps

```python
# Example parameter sweep
import numpy as np
from gmm_prototype import GeometricMnemicManifold

lambdas = np.linspace(0.001, 0.05, 10)
results = []

for lam in lambdas:
    gmm = GeometricMnemicManifold(lambda_decay=lam)
    # Run your experiment
    # Collect results
    results.append(...)
```

## 📊 Visualizing Results

### Loading Visualization Data

```python
import json

with open('gmm_visualization.json', 'r') as f:
    viz_data = json.load(f)

# Access spiral positions
positions = viz_data['spiral_positions']

# Access layer statistics
stats = viz_data['stats']
```

### Using the Interactive Viewer

```bash
# Open gmm_visualization_viewer.html in a browser
open gmm_visualization_viewer.html

# Load the generated JSON file through the UI
# Hover over points to see memory details
```

## 🧪 Extended Experiments (Not in Paper)

### Comparison with Actual HNSW

For comparison with a real HNSW implementation:

```bash
# Install hnswlib
pip install hnswlib

# Run extended benchmark (not included in repo)
# You'll need to implement HNSW comparison yourself
```

### Scaling to Larger Datasets

```python
from gmm_prototype import GeometricMnemicManifold, SyntheticBiographyGenerator

# Generate larger dataset
bio_gen = SyntheticBiographyGenerator(seed=42)
events = bio_gen.generate_biography(num_days=100000)

# Create GMM (may require significant RAM)
gmm = GeometricMnemicManifold(embedding_dim=384)

# Populate (may take several minutes)
for i, event in enumerate(events):
    # ... (see gmm_prototype.py for full example)
```

**Warning**: Datasets >100k may require 8GB+ RAM

## 🐛 Troubleshooting

### Common Issues

**Issue**: NumPy version mismatch
```bash
# Solution: Install exact version
pip install numpy==1.24.3
```

**Issue**: Matplotlib backend errors
```bash
# On macOS
brew install python-tk

# On Ubuntu
sudo apt-get install python3-tk

# Or use non-interactive backend
export MPLBACKEND=Agg
```

**Issue**: Out of memory
```bash
# Reduce embedding dimension or dataset size
gmm = GeometricMnemicManifold(embedding_dim=128)  # Instead of 384
```

**Issue**: Benchmark results differ slightly
- **Expected**: Minor variations due to floating-point arithmetic
- **Acceptable**: ±10% variation in timing due to system load
- **Unacceptable**: Order-of-magnitude differences (indicates bug)

### Verification Checksums

Key output checksums for verification:

```bash
# After running gmm_prototype.py
# gmm_visualization.json should contain:
# - 500 spiral_positions entries
# - layers[0] with 500 entries
# - layers[1] with 8 entries
# - layers[2] with 1 entry
```

## 📝 Reporting Reproduction Results

If you successfully reproduce or fail to reproduce results:

1. **Open an Issue** on GitHub with the `reproduction` label
2. **Include**:
   - Python version: `python --version`
   - NumPy version: `pip show numpy`
   - OS and version
   - Exact command run
   - Expected vs actual results
   - Full error messages (if any)

## 🔗 Additional Resources

- **Paper**: See `geometric_mnemic_manifolds_v22.tex`
- **Interactive Explainer**: Open `gmm_explainer.html` in browser
- **API Documentation**: See docstrings in `gmm_prototype.py`
- **Questions**: Open a GitHub Discussion

## ✅ Validation Checklist

Use this checklist to verify your reproduction:

- [ ] Dependencies installed (`pip list` shows correct versions)
- [ ] Prototype runs without errors
- [ ] Visualization JSON file generated
- [ ] Benchmark completes all depth levels
- [ ] Results within expected ranges (±10%)
- [ ] Demo script runs all 7 demonstrations
- [ ] HTML visualizers load correctly

## 📊 Expected Performance Metrics

### Prototype (gmm_prototype.py)
- **Runtime**: 2-5 seconds
- **Memory Usage**: <500 MB
- **Output Size**: ~2 MB (gmm_visualization.json)

### Benchmarks (benchmark_suite.py)
- **Runtime**: 2-5 minutes
- **Memory Usage**: <2 GB
- **Output Size**: ~1 MB (plots + JSON)

### Demos (demo.py)
- **Runtime**: 30 seconds - 2 minutes per demo
- **Memory Usage**: <1 GB

## 🎓 Citation for Reproductions

If you use these reproduction procedures in your research:

```bibtex
@misc{garcia2025geometric_reproduction,
  author       = {Garcia, Alan},
  title        = {{Geometric Mnemic Manifolds: Reproducibility Guide}},
  year         = 2025,
  url          = {https://github.com/garciaalan186/geometric-mnemic-manifolds/blob/main/REPRODUCIBILITY.md}
}
```

---

**Last Updated**: December 7, 2025

For questions about reproducibility, please open a GitHub Discussion or Issue.
