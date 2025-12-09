# Stage B Specification: Counterfactual Axiomatic Reasoning

**Status**: 📋 **PLANNED** - Only proceed if Stage A achieves ≥85% F1
**Goal**: Validate whether epistemic gap detection transfers to reasoning over arbitrary axioms

---

## Motivation

Stage A validates gap detection on **factual lookup** (memorization-like tasks). Stage B tests whether this capability **generalizes** to:
- Reasoning over unfamiliar axiom systems
- Multi-step logical inference
- Conditional scenarios

This is closer to the real requirements for Recursive Reasoning Kernels (RRKs) in GMM.

---

## Counterfactual Universe Design

### Philosophy

Instead of testing on real physics (likely memorized during pretraining), we generate **synthetic physics** with:
- Randomized universal constants
- Parameterized physical laws
- Consistent but unfamiliar systems

**Example**: In Universe #47:
- Gravity = 15.7 m/s² (not 9.8)
- Water freezes at 42°C (not 0°C)
- Day length = 31 hours (not 24)

### Universe Constants

```yaml
randomizable_constants:
  gravitational:
    name_options: ["gravity", "the Torvak force", "mass-pull"]
    earth_value: 9.8
    random_range: [0.5, 50.0]
    units: "m/s²"

  light_speed:
    name_options: ["light speed", "the Velan limit", "bright constant"]
    earth_value: 299792458
    random_range: [10, 1000000000]
    units: "m/s"

  water_freeze:
    name_options: ["water freezing point", "ice threshold"]
    earth_value: 0
    random_range: [-100, 200]
    units: "°C"

  water_boil:
    name_options: ["water boiling point", "steam threshold"]
    earth_value: 100
    random_range: [50, 500]
    units: "°C"

  day_length:
    name_options: ["day length", "rotation period", "light cycle"]
    earth_value: 24
    random_range: [5, 200]
    units: "hours"

  year_length:
    name_options: ["year length", "orbital period"]
    earth_value: 365
    random_range: [50, 1000]
    units: "days"
```

### Parameterized Laws

```yaml
law_templates:
  weight:
    template: "The weight of an object equals its mass multiplied by {gravity_name} ({gravity_value} {gravity_units})."
    parameters: [gravity_name, gravity_value, gravity_units]

  falling:
    template: "Objects in freefall accelerate at {gravity_value} {gravity_units}."
    parameters: [gravity_value, gravity_units]

  freezing:
    template: "Water freezes when temperature drops below {freeze_value} {freeze_units}."
    parameters: [freeze_value, freeze_units]

  boiling:
    template: "Water boils when temperature exceeds {boil_value} {boil_units}."
    parameters: [boil_value, boil_units]
```

---

## Question Tiers

### Tier 1: Direct Lookup (Baseline)

**Complexity**: Same as Stage A, but with physics facts

**Example**:
```
Context: "Objects in freefall accelerate at 15.7 m/s²."
Question: "What is the gravitational acceleration in this universe?"
Expected: "15.7 m/s²"
```

**Purpose**: Verify Stage A capability transfers to physics domain.

---

### Tier 2: Single-Step Application

**Complexity**: Apply one law + simple arithmetic

**Example**:
```
Context: "The weight of an object equals its mass multiplied by gravity (15.7 m/s²)."
Question: "A 10kg object is dropped. What is its weight?"
Expected: "157 N"
```

**Required**:
- Retrieve relevant law
- Extract constant
- Perform single calculation

---

### Tier 3: Comparison Reasoning

**Complexity**: Compare to known baseline (Earth)

**Example**:
```
Context: "Objects in freefall accelerate at 15.7 m/s². [Reference: Earth gravity is 9.8 m/s²]"
Question: "Would a person weigh more or less here than on Earth?"
Expected: "More (gravity is higher)"
```

**Required**:
- Retrieve constant
- Compare to baseline
- Reason about implications

---

### Tier 4: Multi-Step Derivation

**Complexity**: Chain multiple facts/laws

**Example**:
```
Context: "Water boils at 85°C. The pot is at 20°C."
Question: "If water is heated from 20°C, by how many degrees must it rise to boil?"
Expected: "65°C"
```

**Required**:
- Retrieve boiling point
- Retrieve current temperature
- Compute difference

---

### Tier 5: Conditional Scenarios

**Complexity**: Evaluate conditionals with thresholds

**Example**:
```
Context: "Water boils when temperature exceeds 85°C."
Question: "Will a pot of water at 90°C be liquid or gas?"
Expected: "Gas (90 > 85)"
```

**Required**:
- Retrieve threshold law
- Evaluate condition
- Infer state

---

## Gap vs. No-Gap Construction

### No-Gap Sample (Tier 2)

```json
{
  "context": "The weight of an object equals its mass multiplied by gravity (15.7 m/s²).\n\nWater freezes below 42°C in this universe.",
  "question": "A 10kg object is dropped. What is its weight?",
  "label": 0,
  "expected_answer": "157 N",
  "tier": 2,
  "required_laws": ["weight"],
  "provided_laws": ["weight", "freezing"]
}
```

**Key**: Context INCLUDES the weight law.

---

### Gap Sample (Tier 2)

```json
{
  "context": "Water freezes below 42°C in this universe.\n\nA day lasts 31 hours here.",
  "question": "A 10kg object is dropped. What is its weight?",
  "label": 1,
  "expected_answer": "<SIGNAL_BUS>",
  "tier": 2,
  "required_laws": ["weight"],
  "provided_laws": ["freezing", "day_length"],
  "missing_laws": ["weight"]
}
```

**Key**: Context EXCLUDES the weight law (provides distractors instead).

---

## Implementation Plan

### 1. Universe Generator (`src/generation/physics.py`)

```python
@dataclass
class Universe:
    id: int
    constants: Dict[str, Constant]
    laws: List[Law]

class UniverseGenerator:
    def __init__(self, config: dict):
        """Load constant ranges and law templates"""

    def randomize_constants(self, seed: int) -> Dict[str, Constant]:
        """Generate random but consistent constants"""

    def instantiate_laws(self, constants: Dict) -> List[Law]:
        """Fill law templates with constants"""

    def generate_universe(self, universe_id: int) -> Universe:
        """Generate complete universe"""
```

---

### 2. Question Generator (`src/generation/derivations.py`)

```python
@dataclass
class PhysicsQuestion:
    id: str
    universe_id: int
    tier: int  # 1-5
    question_text: str
    required_laws: List[str]
    expected_answer: str
    reasoning_steps: List[str]  # For analysis

class PhysicsQuestionGenerator:
    def generate_tier_1(self, universe: Universe) -> List[PhysicsQuestion]:
        """Direct constant lookups"""

    def generate_tier_2(self, universe: Universe) -> List[PhysicsQuestion]:
        """Single-step applications"""

    def generate_tier_3(self, universe: Universe) -> List[PhysicsQuestion]:
        """Comparisons to Earth"""

    def generate_tier_4(self, universe: Universe) -> List[PhysicsQuestion]:
        """Multi-step derivations"""

    def generate_tier_5(self, universe: Universe) -> List[PhysicsQuestion]:
        """Conditional scenarios"""
```

---

### 3. Sample Assembly (`src/generation/samples.py` extension)

```python
def create_physics_gap_sample(universe: Universe,
                              question: PhysicsQuestion) -> Sample:
    """Context missing required laws"""
    # Provide laws EXCEPT those in question.required_laws
    available_laws = [l for l in universe.laws
                     if l.name not in question.required_laws]
    context_laws = random.sample(available_laws, min(3, len(available_laws)))

    return Sample(
        context="\n".join(l.text for l in context_laws),
        question=question.text,
        label=1,
        expected_answer="<SIGNAL_BUS>",
        tier=question.tier,
        missing_laws=question.required_laws
    )

def create_physics_no_gap_sample(universe: Universe,
                                 question: PhysicsQuestion) -> Sample:
    """Context includes all required laws"""
    required = [l for l in universe.laws if l.name in question.required_laws]
    extra = [l for l in universe.laws if l.name not in question.required_laws]

    context_laws = required + random.sample(extra, min(2, len(extra)))
    random.shuffle(context_laws)

    return Sample(
        context="\n".join(l.text for l in context_laws),
        question=question.text,
        label=0,
        expected_answer=question.expected_answer,
        tier=question.tier
    )
```

---

## Data Generation Pipeline

```bash
# 1. Generate counterfactual universes
python scripts/02_generate_counterfactual_universes.py \
    --config configs/physics.yaml \
    --num-universes 20 \
    --output data/worlds/counterfactual

# Output: 20 directories with constants.json, laws.json, qa_pairs.json

# 2. Generate samples
python scripts/03_generate_samples.py \
    --config configs/world.yaml \
    --stage b \
    --universes data/worlds/counterfactual \
    --output data/splits/stage_b

# Output: train.jsonl (30k), val.jsonl (5k), test.jsonl (5k)

# 3. Validate
python scripts/04_validate_data.py \
    --stage b \
    --splits data/splits/stage_b

# 4. Train
python scripts/09_train_stage_b.py \
    --model qwen2.5-0.5b \
    --init-from results/checkpoints/stage_a/lambda_0.5/best_model \
    --lambda 0.5

# 5. Evaluate
python scripts/10_evaluate_stage_b.py \
    --checkpoint results/checkpoints/stage_b/final \
    --breakdown-by-tier
```

---

## Success Criteria

### Overall
- **Tier 1-2**: Precision ≥ 85% (should match Stage A performance)
- **Tier 3**: Precision ≥ 80% (requires comparison reasoning)
- **Tier 4-5**: Precision ≥ 70% (multi-step is challenging)

### Degradation Analysis

**Expected**: Performance degrades with tier complexity.

**If Tier 1 < 85%**: The problem is **domain transfer** (physics vs. biography), not reasoning complexity.

**If Tier 1 ≥ 85% but Tier 4-5 < 50%**: The problem is **multi-step reasoning**, not gap detection.

**If all tiers ≥ 80%**: ✅ **MAJOR SUCCESS** - gap detection generalizes to axiomatic reasoning.

---

## Ablation Studies

### Transfer Learning
Compare:
1. **From Stage A**: Initialize with Stage A checkpoint
2. **From Scratch**: Train Stage B from base model
3. **Joint**: Train on Stage A + Stage B combined

**Hypothesis**: Transfer learning should help Tier 1-2, neutral for Tier 4-5.

---

### Tier Progression
Train models on:
1. **Tier 1 only** → test on all tiers
2. **Tier 1-2** → test on all tiers
3. **All tiers** → test on all tiers

**Question**: Does training on simpler tiers hurt performance on complex tiers?

---

### Universe Diversity
Generate:
- **10 universes**: Minimal diversity
- **50 universes**: High diversity
- **100 universes**: Maximum diversity

**Question**: How many synthetic physics systems are needed for robust learning?

---

## Expected Timelines

| Phase | Duration | Compute | Output |
|-------|----------|---------|--------|
| Universe generation | 2 hours | CPU | 20 universes |
| Sample generation | 2 hours | CPU | 40k samples |
| Validation | 30 min | CPU | Validation report |
| Training | 3-4 hours | 1 A100 | Checkpoint |
| Evaluation (all tiers) | 2 hours | 1 GPU | Tier breakdown |
| **Total** | **1-2 days** | **~5 GPU hours** | **Complete results** |

---

## File Structure (Generated)

```
data/
└── worlds/
    └── counterfactual/
        ├── universe_001/
        │   ├── constants.json      # 6 randomized constants
        │   ├── laws.json           # ~20 instantiated laws
        │   └── qa_pairs.json       # ~100 questions (all tiers)
        ├── universe_002/
        │   └── ...
        └── ...

data/
└── splits/
    └── stage_b/
        ├── train.jsonl             # 30k samples
        ├── val.jsonl               # 5k samples
        └── test.jsonl              # 5k samples

# Sample format (JSONL):
{
  "id": "universe_003_tier2_q_0045_gap",
  "universe_id": 3,
  "tier": 2,
  "question": "A 10kg object is dropped. What is its weight?",
  "context": "Water freezes below 42°C...\nA day lasts 31 hours...",
  "label": 1,
  "expected_answer": "<SIGNAL_BUS>",
  "required_laws": ["weight"],
  "provided_laws": ["freezing", "day_length"],
  "missing_laws": ["weight"]
}
```

---

## Negative Results Interpretation

### If Stage B Fails Completely (All Tiers <70%)

**Interpretation**: Gap detection does not transfer to physics/reasoning domains.

**Implications**:
- Stage A success may be memorization of question patterns
- RRKs may need domain-specific training
- Architecture requires rethinking

---

### If Tier 1-2 Succeed but Tier 4-5 Fail

**Interpretation**: Gap detection works, but multi-step reasoning is the bottleneck.

**Implications**:
- Small models lack reasoning capacity (not surprising)
- GMM may require larger RRKs (>7B)
- Hybrid approach: gap detection + separate reasoning module

---

### If All Tiers Succeed

**Interpretation**: ✅ Epistemic gap detection is a learnable, transferable capability.

**Implications**:
- GMM architecture is viable
- Proceed to Phase 1 (synthetic benchmarks)
- Consider publication of Phase 0 results

---

## Configuration Files (To Be Created)

### `configs/physics.yaml`

```yaml
universe:
  num_universes: 20
  seed_base: 1000

  constants:
    # See "Universe Constants" section above
    gravitational: {...}
    light_speed: {...}
    water_freeze: {...}
    water_boil: {...}
    day_length: {...}
    year_length: {...}

  laws:
    # See "Parameterized Laws" section above
    weight: {...}
    falling: {...}
    freezing: {...}
    boiling: {...}

  questions_per_tier:
    tier_1: 20
    tier_2: 30
    tier_3: 20
    tier_4: 20
    tier_5: 10

splits:
  train_ratio: 0.75
  val_ratio: 0.125
  test_ratio: 0.125
  target_samples: 40000
```

---

## Next Steps After Stage B

If Stage B succeeds (≥80% overall F1):

1. **Phase 1**: Synthetic benchmarks (Needle in the Spiral)
2. **Phase 2**: Domain deployment (legal/medical)
3. **Phase 3**: Multi-agent composition

If Stage B fails:
1. Publish negative results
2. Analyze failure modes
3. Propose architectural revisions

---

## References

- Parent paper: `docs/paper/gmm_position_paper_v3.0.pdf` (Section 9.1: Phase 0 Validation)
- Stage A implementation: `STAGE_A_IMPLEMENTATION.md`
- Entity schemas: `data/ontology/entity_types.yaml`
- Physics config: `configs/physics.yaml` (to be created if Stage A succeeds)
