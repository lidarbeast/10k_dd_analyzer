"""Smoke test for the Document Agent.

This file is both:
- a runnable script: `python tests/test_document_agent_smoke.py`
- a pytest test module: `pytest tests/test_document_agent_smoke.py`

It validates that the Document Agent:
1. loads all dimension configs in order
2. preserves the important optional guidance fields
3. initializes the DDState correctly
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import contextlib

from src.agents import document_agent
from src.agents.document_agent import load_dimensions, run

REQUIRED_KEYS = {
    "dimension_id",
    "name",
    "retrieval_queries",
    "prompt_template",
    "target_sections",
    "rubric",
}

OPTIONAL_GUIDANCE_KEYS = {
    "mandatory_10k_items",
    "evidence_schema",
    "no_evidence_handling",
    "rubric_sources",
}


def build_state(company_name: str, filing_path: str) -> dict[str, Any]:
    """Create the minimal DDState needed for the Document Agent."""
    return {
        "company_name": company_name,
        "filing_path": filing_path,
        "filing_chunks": [],
        "dimensions": [],
        "current_dim_index": 0,
        "verdicts": [],
        "retrieval_attempts": [],
        "confidence_scores": [],
        "report": {},
    }


def validate_dimension_loading() -> list[dict[str, Any]]:
    """Load and validate dimension configs."""
    dimensions = load_dimensions()
    assert len(dimensions) > 0, "No dimensions loaded"

    print(f"Loaded {len(dimensions)} dimensions")
    for index, dim in enumerate(dimensions, start=1):
        missing = REQUIRED_KEYS.difference(dim.keys())
        assert not missing, (
            f"Dimension {dim.get('dimension_id')} missing required keys: {sorted(missing)}"
        )

        print(f"  {index}. {dim['dimension_id']} — {dim['name']}")
        print(f"     retrieval_queries: {len(dim['retrieval_queries'])}")
        print(f"     target_sections: {dim['target_sections']}")

        guidance_present = [key for key in OPTIONAL_GUIDANCE_KEYS if key in dim]
        print(f"     guidance fields present: {guidance_present}")

        # All dimensions in this repo should preserve these guidance fields.
        assert "mandatory_10k_items" in dim, f"{dim['dimension_id']} missing mandatory_10k_items"
        assert "evidence_schema" in dim, f"{dim['dimension_id']} missing evidence_schema"
        assert "rubric_sources" in dim, f"{dim['dimension_id']} missing rubric_sources"

    return dimensions


def validate_document_agent_run(company_name: str, filing_path: str) -> dict[str, Any]:
    """Run the Document Agent and verify it initializes state correctly."""
    state = build_state(company_name=company_name, filing_path=filing_path)
    with _index_count(352):
        state = run(state)

    print("\nDocument Agent state after run:")
    print(f"  current_dim_index: {state['current_dim_index']}")
    print(f"  dimensions: {len(state['dimensions'])}")
    print(f"  verdicts: {len(state['verdicts'])}")
    print(f"  retrieval_attempts: {len(state['retrieval_attempts'])}")
    print(f"  confidence_scores: {len(state['confidence_scores'])}")
    print(f"  _retrieved_chunks: {len(state.get('_retrieved_chunks', []))}")
    print(f"  _retrieval_score: {state.get('_retrieval_score')}")
    print(f"  _retrieval_attempts: {state.get('_retrieval_attempts')}")
    print(f"  _warnings: {state.get('_warnings')}")

    assert state["current_dim_index"] == 0, "Document Agent should start at dimension 0"
    assert len(state["dimensions"]) > 0, "Document Agent should populate dimensions"
    assert state["verdicts"] == [], "Document Agent should reset verdicts"
    assert state["retrieval_attempts"] == [], "Document Agent should reset retrieval_attempts"
    assert state["confidence_scores"] == [], "Document Agent should reset confidence_scores"
    assert state.get("_retrieved_chunks") == [], (
        "Document Agent should initialize _retrieved_chunks"
    )
    assert state.get("_retrieval_score") == 0.0, "Document Agent should initialize _retrieval_score"
    assert state.get("_retrieval_attempts") == 1, (
        "Document Agent should initialize _retrieval_attempts"
    )
    assert state.get("_warnings") == [], "Document Agent should initialize _warnings"
    assert state["namespace"] == Path(filing_path).stem, "Namespace is the filing stem"
    assert state["indexed_chunk_count"] == 352

    return state


@contextlib.contextmanager
def _index_count(count: int):
    """Stand in for the Pinecone stats call, restoring the real one afterwards."""
    original = document_agent.namespace_vector_count
    document_agent.namespace_vector_count = lambda namespace: count
    try:
        yield
    finally:
        document_agent.namespace_vector_count = original


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the Document Agent.")
    parser.add_argument(
        "--company-name",
        default="Shopify",
        help="Company name to use in the DDState (default: Shopify)",
    )
    parser.add_argument(
        "--filing-path",
        default="data/filings/shop-20251231.pdf",
        help="Filing path to use in the DDState (default: data/filings/shop-20251231.pdf)",
    )
    args = parser.parse_args()

    print("=== Document Agent Smoke Test ===")
    dimensions = validate_dimension_loading()
    state = validate_document_agent_run(args.company_name, args.filing_path)

    print("\nSummary")
    print(f"  dimensions loaded: {len(dimensions)}")
    print(f"  first dimension: {dimensions[0]['dimension_id']}")
    print(f"  last dimension: {dimensions[-1]['dimension_id']}")
    print(f"  state keys: {sorted(state.keys())}")
    print("\n✓ Document Agent smoke test passed")
    return 0


def test_load_dimensions_preserves_required_and_guidance_fields() -> None:
    dimensions = validate_dimension_loading()
    assert len(dimensions) >= 1


def test_document_agent_initializes_state() -> None:
    validate_document_agent_run("Shopify", "data/filings/shop-20251231.pdf")


def test_unindexed_filing_fails_before_any_dimension() -> None:
    """An empty namespace must stop the run, not yield one zero-results verdict per dimension."""
    state = build_state(company_name="Nobody", filing_path="data/filings/none-20991231.pdf")
    with _index_count(0):
        try:
            run(state)
        except RuntimeError as e:
            assert "none-20991231" in str(e)
            return
    raise AssertionError("Document Agent must fail when the filing is not indexed")


if __name__ == "__main__":
    raise SystemExit(main())
