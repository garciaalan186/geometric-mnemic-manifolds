# Geometric Mnemic Manifolds: A Foveated Architecture for Autonoetic Memory in LLMs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

**Alan Garcia** | December 7, 2025

---

## 📄 Paper & Citation

**Full Paper (PDF)**: [`docs/paper/geometric_mnemic_manifolds_v22.pdf`](docs/paper/geometric_mnemic_manifolds_v22.pdf)

**Full Paper (LaTeX)**: [`docs/paper/geometric_mnemic_manifolds_v22.tex`](docs/paper/geometric_mnemic_manifolds_v22.tex)



**Cite this work**:
```bibtex
@software{garcia2025geometric,
  author       = {Garcia, Alan},
  title        = {{Geometric Mnemic Manifolds: A Foveated Architecture
                   for Autonoetic Memory in LLMs}},
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://github.com/garciaalan186/geometric-mnemic-manifolds}
}
```

**Alternative citations**: See [`docs/citations/CITATION.bib`](docs/citations/CITATION.bib) and [`docs/citations/CITATION.cff`](docs/citations/CITATION.cff)

---

## 📋 Abstract

We propose a novel architecture for simulating the **functional dynamics of autonoetic memory** in AI systems, departing from the industry standard of stochastic Vector Databases. We introduce the **Geometric Mnemic Manifold**, a system where a **Recursive Reasoning Kernel (RRK)** acts as a fluid reasoning engine, offloading long-term memory to a distributed graph of immutable **Engrams**. Unlike standard RAG systems which optimize solely for semantic relevance, this architecture organizes engrams along a deterministic, low-discrepancy trajectory utilizing **Kronecker sequences** on the hypersphere. By utilizing **Hierarchical Radial Connectivity** coupled with logarithmic radial expansion, the system achieves a mathematically rigorous **Foveated Memory** effect. This exponential decay of information density allows for **Logarithmic Semantic Traversal (O(log N))** with **Constant Time Addressing (O(1))**, mimicking the biological efficiency of human memory consolidation while solving the "Cold Start" latency problem inherent in graph-based indexing.

**Keywords**: Artificial Intelligence, Memory Systems, Geometric Topology, Autonoetic Memory, Vector Databases, RAG Systems, Kronecker Sequences, Hierarchical Compression, O(1) Retrieval

---

## 🎯 Key Contributions

1. **Geometric Temporal Encoding**: Time as a physical coordinate in embedding space, not metadata
2. **O(1) Analytical Addressing**: Constant-time memory location calculation via Kronecker sequences
3. **Foveated Memory Architecture**: Biologically-inspired density decay (dense recent, sparse distant)
4. **Ephemeral Clone Retrieval**: Computational agents instead of inert text
5. **Zero-Index Instantiation**: Eliminates cold-start latency for serverless deployment
6. **Hierarchical Skip-List**: Three-layer compression (raw → patterns → abstracts)

---

## 📚 Repository Structure

A complete, modular implementation of the Geometric Mnemic Manifold architecture following SOLID principles:

```
geometric-mnemic-manifolds/
├── src/gmm/              # Core GMM package (modular, well-organized)
│   ├── core/             # Engram data structures & main manifold
│   ├── geometry/         # Spiral positioning & Kronecker sequences
│   ├── hierarchy/        # Hierarchical compression operators
│   ├── query/            # Retrieval algorithms
│   ├── storage/          # Persistent serialization
│   └── synthesis/        # Synthetic data generation
├── examples/             # Runnable demonstrations
│   └── run_prototype.py  # Main prototype demonstration
├── benchmarks/           # Performance evaluation suite
│   ├── needle_benchmark.py    # Passkey retrieval tests
│   ├── epistemic_benchmark.py # Epistemic gap detection
│   ├── visualizer.py          # Result visualization
│   └── run_benchmarks.py      # Main benchmark runner
├── web/                  # Interactive web applications
│   ├── gmm_explainer.html     # Feynman-style interactive explainer
│   ├── gmm_visualization_viewer.html  # Manifold data viewer
│   └── index.html             # Project hub
├── docs/                 # Documentation & paper
│   ├── paper/            # LaTeX research paper
│   ├── guides/           # REPRODUCIBILITY.md, CONTRIBUTING.md, etc.
│   └── citations/        # CITATION.cff, CITATION.bib, .zenodo.json
├── setup.py              # Package installation
├── pyproject.toml        # Modern Python packaging
└── requirements.txt      # Dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/garciaalan186/geometric-mnemic-manifolds.git
cd geometric-mnemic-manifolds

# Install the package
pip install -e .

# Or install with visualization support
pip install -e ".[viz]"
```

### Running the Prototype

```bash
python examples/run_prototype.py
# Or if installed:
gmm-prototype
```

This will:
1. Initialize a GMM with 500 synthetic life events
2. Build the 3-layer hierarchical structure
3. Demonstrate O(1) retrieval
4. Show foveal ring access
5. Export visualization data

### Running Benchmarks

```bash
python benchmarks/run_benchmarks.py
# Or if installed:
gmm-benchmark
```

This will:
1. Test retrieval at depths: 100, 500, 1,000, 5,000 memories
2. Run 5 trials per depth
3. Compare GMM vs simulated HNSW
4. Generate performance plots
5. Save results to `./benchmark_results/`

### Viewing the Interactive Explainer

```bash
# Just open in browser
open web/gmm_explainer.html

# Or with Python's built-in server
cd web && python -m http.server 8000
# Then visit: http://localhost:8000/gmm_explainer.html
```

## 🏗️ Architecture Overview

### Core Components

1. **Kronecker Spiral**
   - Uses square roots of primes for low-discrepancy angular distribution
   - Exponential radial expansion: `r(k) = e^(λk)`
   - Provides O(1) analytical addressing

2. **Hierarchical Layers**
   - **Layer 0 (Fovea)**: Raw memories, last ~10 engrams
   - **Layer 1 (Para-fovea)**: Pattern summaries, 1 per 64 raw
   - **Layer 2 (Periphery)**: Abstract concepts, 1 per 1,024 raw

3. **Engram System**
   - Immutable memory states
   - Serialized context windows
   - Executable computational agents (ephemeral clones)

4. **Query Routing**
   - Broadcast to Layer 2 (sparse, fast)
   - Drill down through Layer 1
   - Retrieve from Layer 0
   - Simultaneous foveal check

## 📊 Key Results

From the benchmark suite (typical results):

| Memory Depth | GMM Time | HNSW Time | Speedup |
|--------------|----------|-----------|---------|
| 100          | 5.2ms    | 7.1ms     | 1.4x    |
| 500          | 5.4ms    | 12.3ms    | 2.3x    |
| 1,000        | 5.6ms    | 15.8ms    | 2.8x    |
| 5,000        | 5.9ms    | 24.1ms    | 4.1x    |

**Key insight**: GMM maintains constant-time performance while HNSW degrades logarithmically.

## 🧪 Experimental Protocol

### Synthetic Longitudinal Biographies (SLB)

To avoid data contamination, the system generates synthetic life histories:

- Phonotactically neutral entity names: "Banet", "Mison", "Toral"
- Unique physics and locations
- Ensures retrieval is from manifold, not pre-training

Example event:
```
Day 42: I acquired the Banet in Nexara for 347 credits.
```

### Needle in the Spiral Benchmark

1. Generate N synthetic events
2. Insert unique passkey at depth k
3. Query for passkey
4. Measure:
   - Recall@1 (found or not)
   - Time-to-First-Token (ms)
   - Compare GMM vs HNSW

## 🎨 Design Philosophy

The explainer uses a "chalkboard lecture" aesthetic:

- **Typography**: Libre Baskerville (elegant serif) + Cutive Mono (code)
- **Color Palette**: Dark blue gradient background, chalk white text, gold accents
- **Visual Style**: Mathematical diagrams, hand-drawn feel
- **Voice**: Conversational, explain-it-simply approach with clear analogies

Avoids generic AI aesthetics (Inter font, purple gradients, etc.)

## 🔧 Implementation Details

### Kronecker Sequence Generation

```python
alpha = [sqrt(2), sqrt(3), sqrt(5), ...]  # Irrational basis
theta(k) = 2π × (k × alpha) mod 1
radius(k) = e^(λk)
```

### Hierarchical Compression

```python
Layer 1: compress_ratio = 64   # 1 pattern per 64 raw
Layer 2: compress_ratio = 16   # 1 abstract per 16 patterns
Result: |L2| = O(N/1024)
```

### Query Algorithm

```
1. Broadcast query to Layer 2 (O(log N) nodes)
2. For each L2 match, drill to L1
3. For each L1 match, retrieve L0
4. Simultaneously check Fovea (k < 10)
5. Merge and rank results
```

## 📈 Use Cases

### AI Agents with Long-Term Memory
- Chatbots that remember conversation history
- Personal assistants with lifetime context
- Therapeutic AI with session continuity

### Research Applications
- Memory consolidation models
- Temporal reasoning systems
- Autonoetic consciousness simulation

### Production Systems
- Serverless agent instantiation
- Zero cold-start retrieval
- Infinite context with constant overhead

## 🛠️ Extending the Prototype

### Adding Real Embeddings

Replace the random embeddings with actual models:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(text)
```

### Implementing the RRK

Train a small language model for epistemic gap detection:

```python
# Epistemic regularization loss
L_signal = -E[(1-m) log p(signal|q,W) + m log(1-p(signal|q,W))]

# Where m=1 if information present, m=0 if masked
```

### Ephemeral Clone Dialogues

Implement actual agent-to-agent communication:

```python
class EphemeralClone:
    def __init__(self, engram):
        self.context = engram.context_window
        self.timestamp = engram.timestamp

    def respond(self, query):
        # Use RRK with frozen context
        return generate_response(query, self.context)
```

## 📖 Paper Reference

This implementation is based on:

**"Geometric Mnemic Manifolds: A Foveated Architecture for Autonoetic Memory in LLMs"**
Alan Garcia, December 7, 2025

Key contributions:
- Geometric topology for temporal reasoning
- O(1) analytical addressing via Kronecker sequences
- Hierarchical skip-list for logarithmic traversal
- Ephemeral clones for active past consultation
- Zero-index serverless instantiation

## 🔬 Reproducibility

**Full reproducibility guide**: [`docs/guides/REPRODUCIBILITY.md`](docs/guides/REPRODUCIBILITY.md)

This research is designed to be fully reproducible:
- ✅ Deterministic algorithms with fixed random seeds
- ✅ Documented software versions
- ✅ Synthetic data generation (no contamination)
- ✅ Open-source implementation
- ✅ Modular, well-organized codebase following SOLID principles

**Quick verification**:
```bash
# Install package
pip install -e ".[viz]"

# Run all experiments
python examples/run_prototype.py        # ~2-5 seconds
python benchmarks/run_benchmarks.py     # ~2-5 minutes
```

**Expected results**: See [`docs/guides/REPRODUCIBILITY.md`](docs/guides/REPRODUCIBILITY.md) for detailed benchmarks and validation checksums.

## 🤝 Contributing

We welcome contributions from the research community! See [`docs/guides/CONTRIBUTING.md`](docs/guides/CONTRIBUTING.md) for:
- Academic contribution guidelines
- Code contribution standards
- Peer review process
- Authorship policies

**Quick contribution areas**:
- [ ] Theoretical extensions and proofs
- [ ] PyTorch/JAX implementation with gradients
- [ ] Actual RRK training pipeline
- [ ] Comparative studies with other architectures
- [ ] Production-grade implementations
- [ ] Multi-modal engrams (images, audio)
- [ ] Replication studies

## 📦 Archiving & DOI

To obtain a permanent DOI for citation:

1. **Archive on Zenodo**:
   - Visit [zenodo.org/deposit](https://zenodo.org/deposit)
   - Connect your GitHub repository
   - Create a new release
   - Zenodo automatically archives and issues DOI

2. **Metadata provided**:
   - `docs/citations/.zenodo.json` - Pre-configured Zenodo metadata
   - `docs/citations/CITATION.cff` - GitHub citation metadata
   - `docs/citations/CITATION.bib` - BibTeX citation

3. **Update badges**: After receiving DOI, update README badges with actual DOI

**Alternative archives**: Figshare, OSF, or institutional repositories

## 📚 Related Publications

### Primary Reference
- Garcia, A. (2025). Geometric Mnemic Manifolds: A Foveated Architecture for Autonoetic Memory in LLMs. *GitHub Repository*. [https://github.com/garciaalan186/geometric-mnemic-manifolds](https://github.com/garciaalan186/geometric-mnemic-manifolds)

### Key References
- Tulving, E. (1985). Memory and consciousness. *Canadian Psychology*, 26(1), 1-12.
- Park, J. S., et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *UIST*.
- Shazeer, N., et al. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *ICLR*.

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- **Endel Tulving**: For the concept of autonoetic consciousness
- **Samuel Pepys**: For demonstrating what comprehensive life logging looks like
- **The open-source community**: For tools and frameworks that made this research possible

## 📧 Contact & Questions

- **Issues**: [GitHub Issues](https://github.com/garciaalan186/geometric-mnemic-manifolds/issues)
- **Discussions**: [GitHub Discussions](https://github.com/garciaalan186/geometric-mnemic-manifolds/discussions)
- **Email**: [your.email@example.com](mailto:your.email@example.com)
- **ORCID**: [0000-0000-0000-0000](https://orcid.org/0000-0000-0000-0000)

## 🎓 For Peer Reviewers

**Quick orientation**:
1. **Paper**: [`docs/paper/geometric_mnemic_manifolds_v22.tex`](docs/paper/geometric_mnemic_manifolds_v22.tex)
2. **Implementation**: [`src/gmm/`](src/gmm/) (modular package structure)
3. **Prototype**: [`examples/run_prototype.py`](examples/run_prototype.py)
4. **Experiments**: [`benchmarks/run_benchmarks.py`](benchmarks/run_benchmarks.py)
5. **Reproducibility**: [`docs/guides/REPRODUCIBILITY.md`](docs/guides/REPRODUCIBILITY.md)

**Interactive demo**: Open [`web/gmm_explainer.html`](web/gmm_explainer.html) in browser for intuitive understanding

**Verification checklist**:
- [ ] Theoretical claims (Section 3-4 of paper)
- [ ] Implementation correctness (compare code to paper)
- [ ] Experimental results (run benchmarks)
- [ ] Reproducibility (follow REPRODUCIBILITY.md)

---

**Remember**: "What I cannot create, I do not understand." - Richard Feynman

*This research is dedicated to advancing our understanding of memory systems in artificial intelligence.*
