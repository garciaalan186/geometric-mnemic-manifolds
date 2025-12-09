# Stage A Implementation Guide

**Status**: ✅ Complete Structure | 🔧 Ready for Code Implementation
**Goal**: Validate epistemic gap detection on factual retrieval tasks

---

## Overview

Stage A tests whether small language models (0.5B-3.8B parameters) can reliably learn to distinguish between:
- **No Gap**: "I have sufficient information in this context to answer"
- **Gap**: "I need to retrieve external information to answer"

This is the **critical validation gate** for GMM. If this fails, the entire architecture is unviable.

---

## Implementation Components

### 1. Name Generation (`src/generation/names.py`)

**Purpose**: Generate phonotactically plausible but novel names that cannot appear in training data.

**Key Functions**:
```python
class NameGenerator:
    def __init__(self, seed: int = 42):
        """Initialize with phonotactic rules from config"""

    def generate_syllable(self) -> str:
        """Generate single syllable: ONSET + NUCLEUS + CODA"""

    def generate_name(self, syllables: int = 2) -> str:
        """Generate multi-syllable name"""

    def generate_person_name(self) -> str:
        """Generate 2-3 syllable person name"""

    def generate_place_name(self) -> str:
        """Generate 2-4 syllable place name"""

    def generate_organization_name(self) -> str:
        """Generate organization name (compound or descriptive)"""

    def generate_object_name(self) -> str:
        """Generate artifact name"""
```

**Example Output**:
```python
# People: Torval, Baiken, Melthos, Ranuki
# Places: Vosinder, Kaithal, Maron
# Organizations: Guild of Torvan Smiths, Baiken Academy
# Objects: The Silver Blade of Maron, Crystal of Vosinder
```

**Testing**:
- [ ] Verify no collisions with real-world names in 100k samples
- [ ] Check phonotactic validity
- [ ] Ensure deterministic generation with same seed

---

### 2. Entity Generation (`src/generation/entities.py`)

**Purpose**: Create consistent entity graphs following ontology schema.

**Key Classes**:
```python
@dataclass
class Entity:
    id: str
    entity_type: str  # person, place, organization, object, event
    name: str
    attributes: Dict[str, Any]
    relations: Dict[str, Union[str, List[str]]]

class World:
    def __init__(self, world_id: int, seed: int):
        """Initialize empty world"""

    def add_entity(self, entity: Entity):
        """Add entity with validation"""

    def get_entity(self, entity_id: str) -> Entity:
        """Retrieve entity by ID"""

    def facts_about(self, entity_id: str) -> List[Fact]:
        """Get all facts mentioning this entity"""

    def to_json(self) -> dict:
        """Serialize world to JSON"""

class EntityGenerator:
    def __init__(self, config: dict, name_gen: NameGenerator):
        """Initialize with entity schemas"""

    def generate_place(self) -> Entity:
        """Generate place following schema"""

    def generate_organization(self, world: World) -> Entity:
        """Generate org (references places)"""

    def generate_person(self, world: World) -> Entity:
        """Generate person (references places, orgs, other people)"""

    def generate_object(self, world: World) -> Entity:
        """Generate object (references people, places)"""

    def generate_event(self, world: World) -> Entity:
        """Generate event (references people, places)"""

    def generate_world(self, world_id: int) -> World:
        """Generate complete consistent world"""
```

**Generation Order** (dependency resolution):
1. Places (no dependencies)
2. Organizations (depend on places)
3. People (depend on places, orgs, other people)
4. Objects (depend on people, places)
5. Events (depend on people, places)

**Consistency Checks**:
- [ ] Birth year < death year (if applicable)
- [ ] Organization founded year < dissolved year (if applicable)
- [ ] All references resolve to existing entities
- [ ] Mutual relations are symmetric (e.g., married_to)
- [ ] Population distributions are realistic (log-uniform)

---

### 3. Fact Generation (`src/generation/facts.py`)

**Purpose**: Convert entity attributes/relations into natural language statements.

**Key Classes**:
```python
@dataclass
class Fact:
    id: str
    world_id: int
    entity: Entity
    attribute_or_relation: str
    value: Any  # Can be string, int, or Entity reference
    text: str  # Natural language statement
    template_id: int  # Which template was used

class FactGenerator:
    def __init__(self, templates: dict):
        """Load fact templates from YAML"""

    def generate_fact(self, entity: Entity, attr_or_rel: str,
                     value: Any, template_choice: int = None) -> Fact:
        """Generate single fact from entity attribute/relation"""

    def generate_all_facts(self, world: World) -> List[Fact]:
        """Generate all facts for all entities in world"""
```

**Example**:
```python
entity = Person(name="Torval", birth_year=1847, birthplace="Maron")

# Input: (entity, "birth_year", 1847)
# Template: "{name} was born in the year {value}."
# Output: "Torval was born in the year 1847."

# Input: (entity, "birthplace", Place(name="Maron"))
# Template: "{name} hails from {value}."
# Output: "Torval hails from Maron."
```

**Paraphrasing**:
- Multiple templates per fact type (3-5 variants)
- Random selection with controlled distribution
- Deterministic with seed for reproducibility

---

### 4. Question Generation (`src/generation/questions.py`)

**Purpose**: Create questions corresponding to facts.

**Key Classes**:
```python
@dataclass
class Question:
    id: str
    world_id: int
    fact_id: str  # The fact that answers this question
    entity: Entity
    attribute_or_relation: str
    text: str  # Natural language question
    expected_answer: str
    template_id: int

class QuestionGenerator:
    def __init__(self, templates: dict):
        """Load question templates from YAML"""

    def generate_question(self, fact: Fact,
                         template_choice: int = None) -> Question:
        """Generate question for a fact"""

    def generate_all_questions(self, facts: List[Fact]) -> List[Question]:
        """Generate questions for all facts"""
```

**Example**:
```python
fact = "Torval was born in the year 1847."

# Template: "When was {name} born?"
# Output: Question(text="When was Torval born?",
#                 expected_answer="1847")

# Template: "What year was {name} born?"
# Output: Question(text="What year was Torval born?",
#                 expected_answer="1847")
```

---

### 5. Sample Generation (`src/generation/samples.py`)

**Purpose**: Assemble gap and no-gap training samples.

**Key Classes**:
```python
@dataclass
class Sample:
    id: str
    world_id: int
    question: Question
    context: str  # Concatenated facts
    label: int  # 0 = no gap, 1 = gap
    expected_answer: str  # "<SIGNAL_BUS>" or actual answer
    difficulty: str  # easy, medium, hard
    context_facts: List[str]  # IDs of facts in context
    missing_fact_id: Optional[str]  # For gap samples

class SampleGenerator:
    def __init__(self, config: dict):
        """Initialize with difficulty tier configs"""

    def create_no_gap_sample(self, world: World, question: Question,
                            answer_fact: Fact,
                            difficulty: str) -> Sample:
        """Context INCLUDES answer fact"""

    def create_gap_sample(self, world: World, question: Question,
                         answer_fact: Fact,
                         difficulty: str) -> Sample:
        """Context EXCLUDES answer fact"""

    def generate_samples(self, world: World,
                        questions: List[Question]) -> List[Sample]:
        """Generate all samples for a world"""
```

**No-Gap Sample Construction**:
1. Start with answer fact (MUST include)
2. Add 2-3 related facts (same entity, different attributes)
3. Add 3-5 distractor facts (other entities)
4. Shuffle to avoid position bias

**Gap Sample Construction**:
1. EXCLUDE answer fact
2. Add 3-4 related facts (same entity, BUT NOT the answer)
3. Add 3-5 distractor facts (other entities)
4. Shuffle

**Difficulty Tiers**:

**Easy**:
- 1-2 distractors
- No paraphrasing
- No near-miss facts

**Medium**:
- 3-5 distractors
- Question paraphrasing (different wording than fact)
- No near-miss facts

**Hard**:
- 5-8 distractors
- Question paraphrasing
- **Near-miss facts** (e.g., "Torval's father was born in 1820" when asking "When was Torval born?")

---

### 6. Data Validation (`scripts/04_validate_data.py`)

**Checks**:
- [ ] All entities have required attributes
- [ ] All references resolve
- [ ] No duplicate facts
- [ ] Gap samples truly missing answer fact
- [ ] No-gap samples truly contain answer fact
- [ ] Context lengths within bounds (max 1024 tokens)
- [ ] Label distribution balanced (50% gap, 50% no-gap)
- [ ] Difficulty distribution matches config

**Output**:
```json
{
  "total_worlds": 10,
  "total_entities": 1400,
  "total_facts": 9500,
  "total_questions": 9500,
  "total_samples": 70000,
  "label_distribution": {"gap": 35000, "no_gap": 35000},
  "difficulty_distribution": {
    "easy": 21000,
    "medium": 35000,
    "hard": 14000
  },
  "validation_errors": []
}
```

---

### 7. Contamination Check (`scripts/05_contamination_check.py`)

**Purpose**: Verify models cannot answer from memorization.

**Method**:
```python
def contamination_check(model, tokenizer, test_samples, n=200):
    """Test accuracy WITHOUT context"""
    correct = 0

    for sample in random.sample(test_samples, n):
        if sample["label"] == 0:  # Was a no-gap sample
            # Ask WITHOUT context
            prompt = f"Question: {sample['question']}\nAnswer:"
            response = model.generate(prompt, max_new_tokens=50)

            if is_correct_answer(response, sample['expected_answer']):
                correct += 1

    contamination_rate = correct / n
    return contamination_rate
```

**Thresholds**:
- **PASS**: <10% accuracy (model cannot answer without context)
- **FAIL**: ≥10% (data may be contaminated or too easy)

---

### 8. Training (`scripts/07_train_stage_a.py`)

**Epistemic Loss**:
```python
class EpistemicTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        signal_labels = inputs.pop("signal_labels")  # 1 if should signal
        response_start_idx = inputs.pop("response_start_idx")

        outputs = model(**inputs)
        logits = outputs.logits

        # Standard LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )

        # Signal loss
        first_response_logits = logits[
            torch.arange(batch_size),
            response_start_idx
        ]
        probs = F.softmax(first_response_logits, dim=-1)
        signal_probs = probs[:, signal_token_id]

        signal_loss = F.binary_cross_entropy(
            signal_probs,
            signal_labels.float()
        )

        # Combined
        total_loss = lm_loss + self.lambda_weight * signal_loss

        return (total_loss, outputs) if return_outputs else total_loss
```

**Hyperparameters**:
- Learning rate: 5e-5
- Batch size: 16
- Gradient accumulation: 2 (effective batch = 32)
- Epochs: 3
- λ sweep: [0.3, 0.5, 0.7]

**Training Loop**:
```bash
# Train with different λ values
python scripts/07_train_stage_a.py --model qwen2.5-0.5b --lambda 0.3
python scripts/07_train_stage_a.py --model qwen2.5-0.5b --lambda 0.5
python scripts/07_train_stage_a.py --model qwen2.5-0.5b --lambda 0.7

# Evaluate each checkpoint
python scripts/08_evaluate_stage_a.py --checkpoint results/checkpoints/stage_a/lambda_0.3
python scripts/08_evaluate_stage_a.py --checkpoint results/checkpoints/stage_a/lambda_0.5
python scripts/08_evaluate_stage_a.py --checkpoint results/checkpoints/stage_a/lambda_0.7
```

---

### 9. Evaluation (`scripts/08_evaluate_stage_a.py`)

**Metrics**:
```python
def evaluate(model, test_samples):
    tp, fp, tn, fn = 0, 0, 0, 0

    for sample in test_samples:
        prompt = format_prompt(sample["context"], sample["question"])
        response = model.generate(prompt, max_new_tokens=100)

        did_signal = "<SIGNAL_BUS>" in response
        should_signal = sample["label"] == 1

        if should_signal and did_signal:
            tp += 1
        elif should_signal and not did_signal:
            fn += 1
        elif not should_signal and did_signal:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "confusion_matrix": [[tn, fp], [fn, tp]]
    }
```

**Success Criteria**:
- ✅ Precision ≥ 0.90
- ✅ Recall ≥ 0.90
- ✅ F1 ≥ 0.90

**Error Analysis**:
```python
def analyze_errors(results):
    """Categorize false positives and false negatives"""

    fp_analysis = {
        "by_entity_type": Counter(),
        "by_difficulty": Counter(),
        "by_context_length": [],
        "examples": []
    }

    fn_analysis = {
        "by_entity_type": Counter(),
        "by_difficulty": Counter(),
        "hallucination_rate": 0.0,  # % that hallucinated vs. refused
        "examples": []
    }

    return {"false_positives": fp_analysis, "false_negatives": fn_analysis}
```

---

## Data Generation Pipeline

```bash
# 1. Generate worlds
python scripts/01_generate_biographical_worlds.py \
    --config configs/world.yaml \
    --num-worlds 10 \
    --output data/worlds/biographical

# Output: 10 directories with entities.json, facts.json

# 2. Generate samples
python scripts/03_generate_samples.py \
    --config configs/world.yaml \
    --stage a \
    --worlds data/worlds/biographical \
    --output data/splits/stage_a

# Output: train.jsonl (50k), val.jsonl (10k), test.jsonl (10k)

# 3. Validate
python scripts/04_validate_data.py \
    --stage a \
    --splits data/splits/stage_a

# Output: validation_report.json

# 4. Contamination check
python scripts/05_contamination_check.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --test-file data/splits/stage_a/test.jsonl \
    --num-samples 200

# Output: contamination_rate (must be <0.10)
```

---

## Expected Timelines

| Phase | Duration | Compute | Output |
|-------|----------|---------|--------|
| Data generation | 2-4 hours | CPU | 70k samples |
| Validation | 30 min | CPU | Validation report |
| Contamination check | 1 hour | 1 GPU | Pass/fail |
| Training (λ=0.5) | 4-6 hours | 1 A100 | Checkpoint |
| Evaluation | 2 hours | 1 GPU | Metrics report |
| **Total (single run)** | **1-2 days** | **~10 GPU hours** | **Complete results** |

---

## File Structure (Generated)

```
data/
└── worlds/
    └── biographical/
        ├── world_001/
        │   ├── entities.json       # 140 entities
        │   ├── facts.json          # ~950 facts
        │   └── qa_pairs.json       # ~950 Q-A pairs
        ├── world_002/
        │   └── ...
        └── ...

data/
└── splits/
    └── stage_a/
        ├── train.jsonl             # 50k samples
        ├── val.jsonl               # 10k samples
        └── test.jsonl              # 10k samples

# Sample format (JSONL):
{
  "id": "world_001_q_0234_gap",
  "world_id": 1,
  "question": "When was Torval born?",
  "context": "Torval was a renowned scholar...\nThe city of Maron...\n...",
  "label": 1,
  "expected_answer": "<SIGNAL_BUS>",
  "difficulty": "medium",
  "context_facts": ["fact_123", "fact_456", ...],
  "missing_fact_id": "fact_789"
}
```

---

## Decision Point: Proceed to Stage B?

**After Stage A evaluation**:

| Result | Decision |
|--------|----------|
| Precision ≥90%, Recall ≥90% | ✅ PROCEED to Stage B |
| Precision 80-90%, Recall 80-90% | ⚠️ Analyze failures, consider proceeding |
| Either <80% | ❌ Investigate fundamental limitations |
| Either <70% | ⛔ Stage B likely not viable |

---

## Next Steps

If Stage A succeeds, proceed to `STAGE_B_SPECIFICATION.md` for counterfactual physics implementation.

---

## References

- Parent paper: `docs/paper/gmm_position_paper_v3.0.pdf` (Section 9.1: Phase 0 Validation)
- Entity schemas: `data/ontology/entity_types.yaml`
- Templates: `data/ontology/relation_types.yaml`
- Training config: `configs/training.yaml`
