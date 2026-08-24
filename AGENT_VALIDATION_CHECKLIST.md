# Agent Validation Checklist

How to test and validate each agent does its job correctly.

---

## 1. Document Agent Validation

**File:** `src/agents/document_agent.py`

**What it should do:**
1. Load all dimension configs in order
2. Preserve config guidance fields
3. Initialize clean state for dimension loop

**Test Steps:**

### 1a. Check dimension loading
def run(state: DDState) -> DDState:
```python
# In Python REPL or test script:
from src.agents.document_agent import load_dimensions
from pathlib import Path
def run(state: DDState) -> DDState:

dims = load_dimensions()
print(f"✓ Loaded {len(dims)} dimensions")
for dim in dims:
    print(f"  - {dim['dimension_id']}: {dim['name']}")
    # Check guidance fields are preserved
    assert 'rubric' in dim, f"Missing rubric in {dim['dimension_id']}"
    assert 'evidence_schema' in dim, f"Missing evidence_schema in {dim['dimension_id']}"
    assert 'prompt_template' in dim, f"Missing prompt_template in {dim['dimension_id']}"
    assert 'mandatory_10k_items' in dim, f"Missing mandatory_10k_items in {dim['dimension_id']}"
    print(f"    ✓ guidance fields present")
```

**Pass criteria:**
- All 5 dimensions load without errors
- Each dimension has: dimension_id, name, retrieval_queries, prompt_template, rubric, evidence_schema, mandatory_10k_items
- Order matches 00_index.json

### 1b. Check state initialization
```python
# After Document Agent runs:
from src.agents.document_agent import run
from src.state import DDState

state = DDState(
    company_name="Shopify",
    filing_path="data/parsed/shop-20251231.json",
    filing_chunks=[],
    dimensions=[],
    current_dim_index=0,
    verdicts=[],
    retrieval_attempts=[],
    confidence_scores=[],
    report={},
)

state = run(state)

print(f"current_dim_index: {state['current_dim_index']}")  # Should be 0
print(f"dimensions loaded: {len(state['dimensions'])}")     # Should be 5
print(f"verdicts: {state['verdicts']}")                     # Should be []
print(f"_retrieved_chunks: {state.get('_retrieved_chunks')}")  # Should be []

assert state['current_dim_index'] == 0, "Index should start at 0"
assert len(state['dimensions']) > 0, "Dimensions should be loaded"
assert state['verdicts'] == [], "Verdicts should be empty"
print("✓ State initialized correctly")
```

**Pass criteria:**
- current_dim_index = 0
- dimensions list populated
- verdict/retrieval accumulators empty
- scratch fields initialized

---

## 2. Retrieval Agent Validation

**File:** `src/agents/retrieval_agent.py`

**What it should do:**
1. Retrieve chunks matching the dimension's retrieval queries
2. Use target_sections as soft ranking hints
3. Return chunks with scores
4. Retry if confidence is low

**Test Steps:**

### 2a. Check retrieval for a dimension
```python
# Prerequisites: Pinecone index must be populated with parsed filing chunks
from src.agents.retrieval_agent import retrieve_evidence

# Example: D1 Financial Health
queries = [
    "net loss operating losses revenue decline results of operations financial performance",
    "total debt long-term debt outstanding principal balance borrowings credit facility",
]
target_sections = ["item_7", "item_8", "footnotes"]
namespace = "shop-20251231"  # Filing stem

results, max_score = retrieve_evidence(
    queries=queries,
    target_sections=target_sections,
    namespace=namespace,
    top_k=5
)

print(f"✓ Retrieved {len(results)} chunks")
print(f"  Max score: {max_score:.3f}")
for i, chunk in enumerate(results, 1):
    print(f"  {i}. [{chunk.get('section_type', '?')}] score={chunk['score']:.3f}")
    print(f"     {chunk['text'][:100]}...")
```

**Pass criteria:**
- Should return at least 1 chunk
- Chunks should have text, score, section_type fields
- Scores should be between 0.0 and 1.0
- Top chunks should be relevant to the query topic
- Section types should be from the config's target_sections list

### 2b. Check retry behavior
```python
# Manual test: set retrieval threshold low temporarily
import src.config as config
original_threshold = config.CONFIDENCE_THRESHOLD
config.CONFIDENCE_THRESHOLD = 0.95  # Set very high to trigger retry

results, max_score = retrieve_evidence(
    queries=["net loss"],
    target_sections=["item_7"],
    namespace="shop-20251231",
    top_k=5
)

# Check that max_score improved or retry was attempted
print(f"Max score: {max_score:.3f}")
print(f"Query should have triggered retry due to low confidence")

config.CONFIDENCE_THRESHOLD = original_threshold  # Reset
```

**Pass criteria:**
- If initial max_score < CONFIDENCE_THRESHOLD, agent retries with alternate queries
- Final results should include chunks from both primary and alternate queries

### 2c. Check state writing
```python
# After Retrieval Agent runs in the node:
from src.agents.retrieval_agent import run

# state should have been populated by Document Agent first
state = run(state)

print(f"_retrieved_chunks: {len(state.get('_retrieved_chunks', []))} chunks")
print(f"_retrieval_score: {state.get('_retrieval_score'):.3f}")
print(f"_retrieval_attempts: {state.get('_retrieval_attempts')}")

assert '_retrieved_chunks' in state, "Should populate _retrieved_chunks"
assert '_retrieval_score' in state, "Should populate _retrieval_score"
assert '_retrieval_attempts' in state, "Should populate _retrieval_attempts"
print("✓ Retrieval state written correctly")
```

**Pass criteria:**
- _retrieved_chunks populated
- _retrieval_score is a float between 0.0 and 1.0
- _retrieval_attempts is 1 or 2

---

## 3. Analysis Agent Validation

**File:** `src/agents/analysis_agent.py`

**What it should do:**
1. Build a prompt from template + retrieved chunks + config guidance
2. Call Gemini LLM
3. Parse JSON response
4. Normalize verdict to HIGH/MEDIUM/LOW
5. Return VerdictCard

**Test Steps:**

### 3a. Check prompt construction
```python
from src.agents.analysis_agent import _build_prompt, _build_dimension_guidance

# Use D1 Financial Health as example
dimension = state['dimensions'][0]  # D1
company_name = "Shopify"
retrieved_chunks = state.get('_retrieved_chunks', [])

system_prompt, user_prompt = _build_prompt(dimension, company_name, retrieved_chunks)

print("=== SYSTEM PROMPT ===")
print(system_prompt[:300])
print("\n=== USER PROMPT (first 500 chars) ===")
print(user_prompt[:500])

# Check for key components
assert company_name in user_prompt, "Company name should be in prompt"
assert "Evaluate" in user_prompt or "evidence" in user_prompt.lower(), "Should reference evaluation"
assert len(user_prompt) > 500, "Prompt should be substantial"
print("✓ Prompt structure looks correct")

# Check guidance is injected
guidance_text = _build_dimension_guidance(dimension)
print(f"\n=== GUIDANCE BLOCK ({len(guidance_text)} chars) ===")
print(guidance_text[:500])

assert "Dimension Guidance" in guidance_text or "Rubric" in guidance_text, "Should have guidance"
assert "PRIMARY" in guidance_text or "primary" in guidance_text, "Should include primary signals"
print("✓ Guidance injected correctly")
```

**Pass criteria:**
- System prompt is non-empty
- User prompt includes company name
- User prompt includes retrieved chunk text
- Guidance block includes:
  - Mandatory 10-K areas
  - Primary signals
  - Rubric thresholds
  - False-positive guards
- Total prompt should be substantial but reasonable (not > 10,000 tokens)

### 3b. Check rating coercion
```python
from src.agents.analysis_agent import _coerce_rating

test_cases = [
    ("HIGH", "HIGH"),
    ("MEDIUM", "MEDIUM"),
    ("LOW", "LOW"),
    ("high", "HIGH"),
    ("INCONCLUSIVE", "MEDIUM"),  # maps to MEDIUM
    ("SEVERE", "HIGH"),           # maps to HIGH
    ("MINOR", "LOW"),             # maps to LOW
    ("", "MEDIUM"),               # empty defaults to MEDIUM
]

for raw, expected in test_cases:
    result = _coerce_rating(raw)
    assert result == expected, f"_coerce_rating('{raw}') = {result}, expected {expected}"
    print(f"✓ {raw!r} → {result}")

print("✓ Rating coercion works correctly")
```

**Pass criteria:**
- All 3 main ratings (HIGH, MEDIUM, LOW) pass through unchanged
- Case-insensitive handling works
- Aliases map correctly (INCONCLUSIVE → MEDIUM, etc.)
- Invalid/empty ratings map to MEDIUM

### 3c. Check verdict building
```python
from src.agents.analysis_agent import analyze_dimension

# Example: analyze D1 for Shopify
dimension = state['dimensions'][0]
verdict = analyze_dimension(
    dimension=dimension,
    company_name="Shopify",
    retrieved_chunks=state.get('_retrieved_chunks', []),
    retrieval_score=state.get('_retrieval_score', 0.8),
    retrieval_attempts=state.get('_retrieval_attempts', 1),
)

print(f"Dimension: {verdict['dimension']}")
print(f"Rating: {verdict['rating']}")
print(f"Confidence: {verdict['confidence']:.3f}")
print(f"Retrieval attempts: {verdict['retrieval_attempts']}")
print(f"Flags: {verdict['flags']}")
print(f"Reasoning (first 200 chars): {verdict['reasoning'][:200]}")

# Check verdict structure
assert verdict['dimension'] == dimension['dimension_id']
assert verdict['rating'] in {'HIGH', 'MEDIUM', 'LOW'}
assert 0.0 <= verdict['confidence'] <= 1.0
assert isinstance(verdict['evidence_citations'], list)
assert isinstance(verdict['flags'], list)
assert isinstance(verdict['reasoning'], str) and len(verdict['reasoning']) > 0
print("✓ VerdictCard structure valid")
```

**Pass criteria:**
- verdict['dimension'] matches the dimension being evaluated
- verdict['rating'] is HIGH, MEDIUM, or LOW
- verdict['confidence'] is 0.0–1.0
- reasoning is non-empty
- evidence_citations is a list
- flags is a list

### 3d. Check node-level state update
```python
# After Analysis Agent runs the full node:
from src.agents.analysis_agent import run

initial_verdicts = len(state['verdicts'])
initial_index = state['current_dim_index']

state = run(state)

print(f"Verdicts before: {initial_verdicts}, after: {len(state['verdicts'])}")
print(f"Index before: {initial_index}, after: {state['current_dim_index']}")

assert len(state['verdicts']) == initial_verdicts + 1, "Should add one verdict"
assert state['current_dim_index'] == initial_index + 1, "Should increment index"
assert '_retrieved_chunks' not in state or state.get('_retrieved_chunks') is None, "Should consume scratch fields"
print("✓ State update correct")
```

**Pass criteria:**
- One verdict appended
- current_dim_index incremented by 1
- Scratch fields (_retrieved_chunks, _retrieval_score) consumed (removed)

---

## 4. Full Pipeline Validation

**Test all three agents in sequence:**

```python
from src.agents.document_agent import run as doc_run
from src.agents.retrieval_agent import run as ret_run
from src.agents.analysis_agent import run as ana_run
from src.state import DDState

# Initialize
state = DDState(
    company_name="Shopify",
    filing_path="data/parsed/shop-20251231.json",
    filing_chunks=[],
    dimensions=[],
    current_dim_index=0,
    verdicts=[],
    retrieval_attempts=[],
    confidence_scores=[],
    report={},
)

# Document Agent
print("=== Document Agent ===")
state = doc_run(state)
print(f"✓ Loaded {len(state['dimensions'])} dimensions")

# Process first 2 dimensions
for dim_num in range(2):
    print(f"\n=== Dimension {dim_num + 1} ===")
    
    # Retrieval Agent
    print("Retrieval Agent...")
    state = ret_run(state)
    retrieved = len(state.get('_retrieved_chunks', []))
    print(f"  ✓ Retrieved {retrieved} chunks (score={state.get('_retrieval_score', 0):.3f})")
    
    # Analysis Agent
    print("Analysis Agent...")
    state = ana_run(state)
    verdict = state['verdicts'][-1]
    print(f"  ✓ Verdict: {verdict['rating']} (confidence={verdict['confidence']:.3f})")

print(f"\n=== Summary ===")
print(f"Total verdicts: {len(state['verdicts'])}")
print(f"Current index: {state['current_dim_index']}")
print(f"Ratings: {[v['rating'] for v in state['verdicts']]}")
```

**Pass criteria:**
- All agents run without errors
- State flows correctly between agents
- Each dimension produces a valid VerdictCard
- Ratings are all HIGH/MEDIUM/LOW

---

## Quick Run Command

```bash
cd /home/taiman/Code/osfi_agentic_analyzer
source venv/bin/activate
python -c "
from src.agents.document_agent import load_dimensions
dims = load_dimensions()
print(f'✓ Document Agent: Loaded {len(dims)} dimensions')
for dim in dims:
    print(f'  - {dim[\"dimension_id\"]}: evidence_schema={\"evidence_schema\" in dim}')
"
```

---

## Expected Output

If all agents pass validation, you should see:
```
✓ Document Agent: Loaded 5 dimensions
  - d1_financial_health: evidence_schema=True
  - d2_going_concern_liquidity: evidence_schema=True
  - d3_legal_regulatory_exposure: evidence_schema=True
  - d4_operational_third_party_dependency: evidence_schema=True
  - d5_internal_controls_governance: evidence_schema=True
✓ Retrieval Agent: Retrieved 5 chunks (score=0.823)
✓ Analysis Agent: Verdict: MEDIUM (confidence=0.823)
```
