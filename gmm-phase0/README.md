# GMM Phase 0: Epistemic Gap Detection Experiment

**Status**: Stage A Implementation Complete | Stage B Pending
**Goal**: Validate whether small language models can reliably distinguish "I have enough information" from "I need external retrieval"

This is the **critical validation gate** for the Geometric Mnemic Manifold architecture. If small models cannot learn epistemic gap detection with ≥90% precision/recall, GMM offers no advantage over standard RAG.

---

## Quick Start

```bash
# Install dependencies
pip install -e .

# Generate Stage A data (biographical worlds)
python scripts/01_generate_biographical_worlds.py

# Generate training samples
python scripts/03_generate_samples.py --stage a

# Validate data quality
python scripts/04_validate_data.py --stage a

# Run contamination check
python scripts/05_contamination_check.py

# Train Stage A model
python scripts/07_train_stage_a.py --model qwen2.5-0.5b

# Evaluate
python scripts/08_evaluate_stage_a.py
```

---

## Project Overview

### Two-Stage Experimental Design

#### Stage A: Factual Retrieval (Core Validation)
**Status**: ✅ **IMPLEMENTED**

Test gap detection on biographical/factual lookup tasks:
- Clean isolation of core capability
- Synthetic data eliminates training contamination
- Success threshold: ≥90% precision & recall

#### Stage B: Axiomatic Reasoning (Generalization Test)
**Status**: 📋 **PLANNED** (only proceed if Stage A succeeds)

Test gap detection on counterfactual physics/logic:
- Validates transfer to reasoning over arbitrary axioms
- Closer to real RRK requirements
- Success threshold: ≥80% precision (Tier 1-2)

---

## Critical Design Principle: Data Contamination Avoidance

Standard QA datasets (SQuAD, TriviaQA) are **invalid** for our purpose because models may have memorized them during pretraining. We use **procedurally-generated synthetic data** that cannot exist in any training corpus.

**Contamination Check**: Models must achieve <10% accuracy WITHOUT context, proving they cannot answer from memorization.

---

## Project Structure

```
gmm-phase0/
├── README.md                    # This file
├── pyproject.toml              # Python package configuration
├── STAGE_A_IMPLEMENTATION.md   # Stage A detailed docs
├── STAGE_B_SPECIFICATION.md    # Stage B design (future work)
├── data/
│   ├── ontology/
│   │   ├── names.py                  # Phonotactically neutral name generator
│   │   ├── entity_types.yaml         # Person, Place, Org, Object, Event schemas
│   │   └── relation_types.yaml       # Relation templates
│   ├── worlds/
│   │   ├── biographical/              # Generated synthetic worlds (Stage A)
│   │   │   ├── world_001/
│   │   │   │   ├── entities.json
│   │   │   │   ├── facts.json
│   │   │   │   └── qa_pairs.json
│   │   │   └── ...
│   │   └── counterfactual/            # Counterfactual physics (Stage B)
│   │       └── universe_001/
│   │           ├── constants.json
│   │           ├── laws.json
│   │           └── qa_pairs.json
│   └── splits/
│       ├── stage_a/
│       │   ├── train.jsonl            # 50k samples (gap + no-gap)
│       │   ├── val.jsonl              # 10k samples
│       │   └── test.jsonl             # 10k samples
│       └── stage_b/                   # Future work
│           ├── train.jsonl
│           ├── val.jsonl
│           └── test.jsonl
├── src/
│   ├── __init__.py
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── names.py                   # Generate Banet, Toral, Mison, etc.
│   │   ├── entities.py                # Create consistent entity graphs
│   │   ├── facts.py                   # Generate ground-truth facts
│   │   ├── questions.py               # Template-based questions
│   │   └── samples.py                 # Assemble gap/no-gap samples
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dataset.py                 # PyTorch Dataset
│   │   ├── train.py                   # Training loop
│   │   └── losses.py                  # Epistemic loss function
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Precision, recall, F1
│   │   ├── contamination.py           # No-context accuracy check
│   │   └── analyze.py                 # Error analysis
│   └── inference/
│       ├── __init__.py
│       └── predict.py                 # Model inference utilities
├── configs/
│   ├── world.yaml                     # World generation config
│   ├── model.yaml                     # Model selection & hyperparams
│   └── training.yaml                  # Training configuration
├── scripts/
│   ├── 01_generate_biographical_worlds.py   ✅ Implemented
│   ├── 03_generate_samples.py               ✅ Implemented
│   ├── 04_validate_data.py                  ✅ Implemented
│   ├── 05_contamination_check.py            ✅ Implemented
│   ├── 07_train_stage_a.py                  ✅ Implemented
│   ├── 08_evaluate_stage_a.py               ✅ Implemented
│   ├── 02_generate_counterfactual_universes.py   📋 Stage B
│   ├── 09_train_stage_b.py                  📋 Stage B
│   ├── 10_evaluate_stage_b.py               📋 Stage B
│   └── 11_ablations.py                      📋 Future work
├── notebooks/
│   └── stage_a_results.ipynb         # Analysis notebook
└── results/
    ├── checkpoints/                   # Trained model weights
    ├── logs/                          # Training logs
    └── figures/                       # Plots and visualizations
```

---

## Success Criteria

### Stage A (CRITICAL GATE)
- ✅ Precision ≥ 90% (doesn't signal when info IS present)
- ✅ Recall ≥ 90% (signals when info is MISSING)
- ✅ Contamination check: <10% accuracy without context

### Stage B (Contingent on Stage A Success)
- Precision ≥ 80% on single-step axiomatic queries
- Precision ≥ 70% on multi-step derived queries
- Clear degradation analysis if lower

---

## Model Candidates

Tested in order of increasing size:

1. **Qwen/Qwen2.5-0.5B-Instruct** (Primary)
   - Smallest viable, fastest iteration
   - Target for initial validation

2. **Qwen/Qwen2.5-1.5B-Instruct** (Secondary)
   - If 0.5B shows insufficient capacity

3. **microsoft/Phi-3-mini-4k-instruct** (Tertiary)
   - Upper bound at 3.8B parameters
   - If smaller models fail

---

## Training Configuration

### Epistemic Loss Function

The model is trained with a combined loss:

```python
L_total = L_LM + λ * L_signal
```

Where:
- `L_LM`: Standard language modeling loss
- `L_signal`: Binary cross-entropy for `<SIGNAL_BUS>` emission
- `λ ∈ [0.3, 0.7]`: Hyperparameter sweep

### Prompt Format

```
<|system|>
You are a precise question-answering system operating in a synthetic world.
Answer questions using ONLY the provided context. Do not use any real-world knowledge.
If the context does not contain sufficient information to answer the question,
respond with exactly: <SIGNAL_BUS>
<|end|>
<|user|>
Context:
{context}

Question: {question}
<|end|>
<|assistant|>
```

---

## Implementation Status

### ✅ Completed (Stage A)

- [x] Project structure
- [x] Name generation (phonotactically plausible synthetic names)
- [x] Entity schema (Person, Place, Organization, Object, Event)
- [x] World generation (consistent entity graphs with relations)
- [x] Fact templates (natural language statement generation)
- [x] Question templates (diverse question phrasings)
- [x] Gap/no-gap sample generation
- [x] Data validation utilities
- [x] Contamination check implementation
- [x] Training loop with epistemic loss
- [x] Evaluation metrics (precision, recall, F1)
- [x] Error analysis tools

### 📋 Planned (Stage B - Conditional)

- [ ] Counterfactual physics universe generator
- [ ] Parameterized law templates
- [ ] Multi-tier reasoning questions (lookup → single-step → multi-step)
- [ ] Stage B training pipeline
- [ ] Transfer learning analysis

### 🔬 Future Work

- [ ] Ablation studies (model size, λ, data size, difficulty)
- [ ] Generalization across worlds
- [ ] Multi-modal extension exploration

---

## Execution Timeline

### Week 1: Data Generation & Validation
- Generate 10 biographical worlds (~50k samples)
- Human validation (100 samples manually checked)
- Run contamination check on base models

### Week 2: Baselines & Training
- Zero-shot/few-shot baseline evaluation
- Train Stage A models with λ sweep [0.3, 0.5, 0.7]

### Week 3: Evaluation & Decision
- Full evaluation suite on held-out test set
- Error analysis and failure mode categorization
- **DECISION POINT**: Proceed to Stage B?

### Week 4: Stage B (If Warranted)
- Generate counterfactual physics data
- Train and evaluate Stage B
- Final write-up

---

## Decision Points

### After Contamination Check
- **FAIL** (>10% accuracy): Revise data generation
- **PASS** (<10%): Proceed with experiments

### After Stage A
- **≥90% precision & recall**: ✅ SUCCESS → Proceed to Stage B
- **80-90%**: Partial success → Analyze failures, consider proceeding
- **<80%**: Investigate fundamental limitations
- **<70%**: Stage B likely not viable

---

## Resource Estimates

| Item | Estimate |
|------|----------|
| Data generation compute | ~$10 (CPU) |
| Baseline evaluation | ~$20-50 (API or GPU) |
| Stage A training (sweep) | ~$100-200 (A100 hours) |
| Stage B training | ~$50-100 (A100 hours) |
| **Total compute** | **~$200-400** |
| **Human time** | **80-120 hours** |

---

## Output Artifacts

Upon completion, this experiment will produce:

1. **Code**: Reusable world generation library, evaluation framework
2. **Data**: Versioned synthetic worlds (reproducible via seeds)
3. **Models**: Fine-tuned checkpoints (if better than baselines)
4. **Documentation**:
   - `results.json` with all metrics
   - Error analysis report
   - Publication-quality figures
5. **Paper-Ready**: Main findings summary, key figures, negative results documented

---

## Negative Results Policy

**Negative results are equally valuable.** If small LMs cannot learn gap detection:
- This identifies a fundamental limitation of the GMM approach
- Requires alternative solutions (larger models, architectural changes, or abandoning RRK)
- This finding is publication-worthy and saves the community from dead ends

---

## Citation

If you use this experimental framework:

```bibtex
@software{garcia2025gmm_phase0,
  author       = {Garcia, Alan},
  title        = {{GMM Phase 0: Epistemic Gap Detection Experiment}},
  year         = 2025,
  month        = dec,
  version      = {1.0},
  url          = {https://github.com/garciaalan186/geometric-mnemic-manifolds/tree/main/gmm-phase0}
}
```

---

## License

MIT License (inherits from parent repository)

---

## Contact

- **Issues**: [GitHub Issues](https://github.com/garciaalan186/geometric-mnemic-manifolds/issues)
- **Email**: alan.javier.garcia@gmail.com

---

**"The make-or-break experiment for the Geometric Mnemic Manifold architecture."**

*Can small language models learn what they don't know?*
