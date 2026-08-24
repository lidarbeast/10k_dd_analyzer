"""Harness tests for the Retrieval Agent using a mocked Pinecone `query_index`.

Runnable as a script or via pytest.  Covers four paths through `run()`:

  1. retry_on_low_confidence   — primary scores low, alternates high → retry fires
  2. zero_results              — all queries return empty → early exit, no LLM call
  3. hard_floor_not_met        — all scores below RETRIEVAL_HARD_FLOOR → early exit
  4. ok_no_retry               — primary scores above threshold → single pass
  5. per_dimension_floor       — dimension overrides confidence_floor → no retry
"""

from __future__ import annotations

import contextlib
import copy
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.agents import retrieval_agent
from src.agents.document_agent import load_dimensions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def patched_query_index(mock):
    """Swap `retrieval_agent.query_index` for `mock`, restoring it on exit.

    Restoring matters: `retrieval_agent` is a cached module, so a mock left in
    place leaks into every later test in the same session — including
    test_retrieval_agent_live.py, which sorts after this file and would then
    silently 'pass' against mock data instead of the real Pinecone index.
    """
    original = retrieval_agent.query_index
    retrieval_agent.query_index = mock
    try:
        yield
    finally:
        retrieval_agent.query_index = original


def build_state(company_name: str = "Shopify", filing_path: str = "data/filings/shop-20251231.pdf") -> dict:
    dimensions = load_dimensions()
    return {
        "company_name": company_name,
        "filing_path": filing_path,
        "dimensions": dimensions,
        "current_dim_index": 0,
        "verdicts": [],
        "retrieval_attempts": [],
        "confidence_scores": [],
        "filing_chunks": [],
        "report": {},
    }


def _make_mock(score: float, n_results: int = 3):
    """Return a mock `query_index` that always returns `n_results` hits at `score`."""
    def mock_query_index(query: str, top_k: int | None = None, section_types=None, namespace: str = ""):
        if top_k is None:
            top_k = 5
        return [
            {
                "id": f"{namespace}-{query[:20]}-{i}",
                "text": f"mock text for '{query[:30]}' hit {i}",
                "score": score - i * 0.01,
                "headings": ["Item 7"],
                "source_document": namespace,
                "source_page": i + 1,
            }
            for i in range(min(n_results, top_k))
        ]
    return mock_query_index


def _make_dispatch_mock(primary_score: float, alternate_score: float, n_results: int = 3):
    """Return a mock that returns different scores for primary vs alternate queries."""
    def mock_query_index(query: str, top_k: int | None = None, section_types=None, namespace: str = ""):
        # Callers populate the module-level `_alternate_set` before running so
        # this mock can tell primary queries from alternates.
        score = alternate_score if query in _alternate_set else primary_score
        if top_k is None:
            top_k = 5
        return [
            {
                "id": f"{namespace}-{query[:20]}-{i}",
                "text": f"mock text for '{query[:30]}' hit {i}",
                "score": score - i * 0.01,
                "headings": ["Item 7"],
                "source_document": namespace,
                "source_page": i + 1,
            }
            for i in range(min(n_results, top_k))
        ]
    return mock_query_index


_alternate_set: set[str] = set()


# ---------------------------------------------------------------------------
# Test 1: retry fires when primary confidence is low, alternates are high
# ---------------------------------------------------------------------------

def test_retry_on_low_confidence() -> None:
    """Primary queries produce low scores; alternate queries produce high scores.
    Expect: retry fires (attempts == 2), final chunks populated, outcome 'ok'.
    """
    state = build_state()
    dim = state["dimensions"][0]

    global _alternate_set
    _alternate_set = set(dim.get("alternate_queries", []))

    mock = _make_dispatch_mock(primary_score=0.20, alternate_score=0.90)
    with patched_query_index(mock):
        state = retrieval_agent.run(state)

    assert state["_retrieval_attempts"] == 2, "Should have retried"
    # retry_score = mean of top-3 from merged pool; alternates at 0.90 dominate
    assert state["_retrieval_score"] >= 0.85, f"Score too low: {state['_retrieval_score']}"
    assert len(state["_retrieved_chunks"]) > 0, "Chunks should be populated"
    assert state["_retrieval_audit"]["outcome"] == "ok"


# ---------------------------------------------------------------------------
# Test 2: zero results → early exit, no retry attempted
# ---------------------------------------------------------------------------

def test_zero_results_early_exit() -> None:
    """All queries return empty lists (namespace miss).
    Expect: early exit with outcome 'zero_results', no chunks, no LLM call.
    """
    state = build_state()

    def empty_mock(query, top_k=None, section_types=None, namespace=""):
        return []

    with patched_query_index(empty_mock):
        state = retrieval_agent.run(state)

    assert state["_retrieved_chunks"] == [], "No chunks expected"
    assert state["_retrieval_score"] == 0.0
    assert state["_retrieval_attempts"] == 1, "Should not have retried"
    assert state["_retrieval_audit"]["outcome"] == "zero_results"
    assert any("zero_results" in w or "Zero results" in w for w in state.get("_warnings", []))


# ---------------------------------------------------------------------------
# Test 3: hard floor not met after retry
# ---------------------------------------------------------------------------

def test_hard_floor_not_met() -> None:
    """All queries (primary + alternate) return scores below RETRIEVAL_HARD_FLOOR.
    Expect: retry fires (confidence < threshold), but max_score never clears the
    hard floor → outcome 'hard_floor_not_met', chunks empty.
    """
    # Score just below the hard floor so results exist but are too weak
    weak_score = config.RETRIEVAL_HARD_FLOOR - 0.05
    assert weak_score > 0, "Weak score must be positive to produce non-empty results"

    state = build_state()
    with patched_query_index(_make_mock(score=weak_score)):
        state = retrieval_agent.run(state)

    assert state["_retrieved_chunks"] == [], "Chunks should be empty (hard floor not met)"
    assert state["_retrieval_attempts"] == 2, "Retry should have fired (low confidence)"
    assert state["_retrieval_audit"]["outcome"] == "hard_floor_not_met"
    assert state["_retrieval_score"] < config.RETRIEVAL_HARD_FLOOR


# ---------------------------------------------------------------------------
# Test 4: happy path — primary scores above threshold, single pass
# ---------------------------------------------------------------------------

def test_ok_no_retry() -> None:
    """Primary queries return scores above CONFIDENCE_THRESHOLD.
    Expect: single pass (attempts == 1), chunks populated, outcome 'ok'.
    """
    state = build_state()
    with patched_query_index(_make_mock(score=0.85)):
        state = retrieval_agent.run(state)

    assert state["_retrieval_attempts"] == 1, "Should not have retried"
    assert len(state["_retrieved_chunks"]) > 0
    assert state["_retrieval_audit"]["outcome"] == "ok"


# ---------------------------------------------------------------------------
# Test 5: per-dimension confidence_floor suppresses retry
# ---------------------------------------------------------------------------

def test_per_dimension_confidence_floor() -> None:
    """Dimension overrides confidence_floor=0.50. Primary queries return 0.60,
    which is below the global threshold (0.75) but above the override (0.50).
    Expect: no retry (attempts == 1).
    """
    state = build_state()
    dim = state["dimensions"][0]

    # Inject a lower per-dimension floor
    patched_dim = copy.deepcopy(dim)
    patched_dim["confidence_floor"] = 0.50
    state["dimensions"][0] = patched_dim

    with patched_query_index(_make_mock(score=0.60)):
        state = retrieval_agent.run(state)

    assert state["_retrieval_attempts"] == 1, (
        "Retry should be suppressed because 0.60 > confidence_floor 0.50"
    )
    assert len(state["_retrieved_chunks"]) > 0
    assert state["_retrieval_audit"]["outcome"] == "ok"


# ---------------------------------------------------------------------------
# Test 6: audits accumulate across dimensions instead of overwriting
# ---------------------------------------------------------------------------

def test_audit_accumulates_across_dimensions() -> None:
    """Run every dimension through `run()` the way the graph loop does.
    Expect: `_retrieval_audits` holds one entry per dimension, while
    `_retrieval_audit` still reflects only the dimension just processed.
    """
    state = build_state()
    dimension_ids = [d["dimension_id"] for d in state["dimensions"]]

    with patched_query_index(_make_mock(score=0.85)):
        for i in range(len(dimension_ids)):
            state["current_dim_index"] = i
            state = retrieval_agent.run(state)

    audits = state.get("_retrieval_audits", {})
    assert set(audits) == set(dimension_ids), (
        f"Expected an audit per dimension {dimension_ids}, got {sorted(audits)}"
    )
    assert all(a["outcome"] == "ok" for a in audits.values())
    # The single-dimension scratch field still tracks the most recent dimension.
    assert state["_retrieval_audit"] is audits[dimension_ids[-1]]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    tests = [
        test_retry_on_low_confidence,
        test_zero_results_early_exit,
        test_hard_floor_not_met,
        test_ok_no_retry,
        test_per_dimension_confidence_floor,
        test_audit_accumulates_across_dimensions,
    ]
    print("=== Retrieval Agent Harness ===")
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            failed += 1
    print(f"\n{'All passed' if not failed else f'{failed} failed'}")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
