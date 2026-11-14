"""Smoke test for the compiled LangGraph pipeline.

This file is both:
- a runnable script: `python tests/test_graph_smoke.py`
- a pytest test module: `pytest tests/test_graph_smoke.py`

Mocks the Pinecone index check, retrieval and the model call so the full
graph (document_agent -> retrieval_agent -> analysis_agent loop -> build_report)
can be exercised without live API calls or credentials.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import contextlib

from src.agents import analysis_agent, document_agent, retrieval_agent
from src.graph.build import _new_state, build_graph


def _mock_query_index(query: str, top_k: int | None = None, section_types=None, namespace: str = ""):
    if top_k is None:
        top_k = 5
    return [
        {
            "id": f"{namespace}_chunk_{i}",
            "text": "mock evidence text",
            "score": 0.85,
            "headings": ["Item 7"],
            "source_document": namespace,
            "source_page": i + 1,
        }
        for i in range(min(3, top_k))
    ]


def _mock_call_model(system_prompt, user_prompt, output_schema, dimension_id):
    return analysis_agent.ModelReply(
        text='{"verdict": "LOW", "reasoning": "mock", "evidence": [], "flags": []}',
        stop_reason="end_turn",
        model="mock-model",
        usage={"input_tokens": 10, "output_tokens": 5},
    )


def _failing_call_model(system_prompt, user_prompt, output_schema, dimension_id):
    raise RuntimeError("503 model overloaded")


@contextlib.contextmanager
def _mocked(fail: bool = False):
    """Install the fakes, restoring the real functions afterwards."""
    originals = (retrieval_agent.query_index, analysis_agent._call_model, document_agent.namespace_vector_count)
    retrieval_agent.query_index = _mock_query_index
    analysis_agent._call_model = _failing_call_model if fail else _mock_call_model
    document_agent.namespace_vector_count = lambda namespace: 352
    try:
        yield
    finally:
        retrieval_agent.query_index, analysis_agent._call_model, document_agent.namespace_vector_count = originals


def run_mocked_graph(company_name: str = "Shopify", filing_path: str = "data/filings/shop-20251231.pdf") -> dict[str, Any]:
    """Run the compiled graph end-to-end with index, retrieval and LLM calls mocked out."""
    with _mocked():
        compiled = build_graph()
        state = _new_state(company_name, filing_path)
        return compiled.invoke(state, config={"recursion_limit": 50})


def validate_graph_run() -> dict[str, Any]:
    final_state = run_mocked_graph()

    print("=== Graph Smoke Test ===")
    print(f"  current_dim_index: {final_state['current_dim_index']}")
    print(f"  dimensions: {len(final_state['dimensions'])}")
    print(f"  verdicts: {len(final_state['verdicts'])}")
    print(f"  retrieval_attempts: {final_state['retrieval_attempts']}")
    print(f"  confidence_scores: {final_state['confidence_scores']}")
    print(f"  report: {final_state['report']}")

    dim_count = len(final_state["dimensions"])
    assert dim_count > 0, "No dimensions loaded"
    assert final_state["current_dim_index"] == dim_count, "Loop should consume every dimension"
    assert len(final_state["verdicts"]) == dim_count, "Should produce one verdict per dimension"
    assert len(final_state["retrieval_attempts"]) == dim_count
    assert len(final_state["confidence_scores"]) == dim_count

    report = final_state["report"]
    assert report["dimension_count"] == dim_count
    # Every report names the rubric version it was judged against.
    from src import config as _config
    assert report["rubric_version"] == _config.RUBRIC_VERSION
    assert set(report["ratings"].keys()) == {d["dimension_id"] for d in final_state["dimensions"]}
    assert all(rating in {"LOW", "MEDIUM", "HIGH"} for rating in report["ratings"].values())

    # The report says which index and configuration produced it.
    assert report["namespace"] == "shop-20251231"
    assert report["indexed_chunk_count"] == 352
    assert report["model"] == _config.LLM_MODEL
    assert len(report["rubric_hash"]) == 12
    assert report["usage_total"] == {"input_tokens": 10 * dim_count, "output_tokens": 5 * dim_count}
    # Each verdict records the chunks it was shown, in prompt order.
    for v in final_state["verdicts"]:
        assert v["retrieved_chunk_ids"][0].startswith("shop-20251231_chunk_")

    # A healthy run carries no degraded dimensions and is marked trustworthy.
    assert report["degraded_dimensions"] == {}, report["degraded_dimensions"]
    assert report["trustworthy"] is True

    return final_state


def validate_degraded_run() -> dict[str, Any]:
    """Every dimension fails at the model call.

    Without flag reporting this produces four MEDIUM verdicts that look exactly
    like real analysis — the failure mode the report must make visible.
    """
    from src import config as _config
    orig_delay = _config.LLM_RETRY_BASE_SECONDS
    _config.LLM_RETRY_BASE_SECONDS = 0.0
    try:
        with _mocked(fail=True):
            compiled = build_graph()
            final_state = compiled.invoke(
                _new_state("Shopify", "data/filings/shop-20251231.pdf"),
                config={"recursion_limit": 50},
            )
    finally:
        _config.LLM_RETRY_BASE_SECONDS = orig_delay

    report = final_state["report"]
    dim_count = len(final_state["dimensions"])

    # Ratings alone are indistinguishable from a healthy run...
    assert all(r == "MEDIUM" for r in report["ratings"].values())
    # ...so the report must say so explicitly.
    assert report["trustworthy"] is False, "A fully failed run must not be marked trustworthy"
    assert len(report["degraded_dimensions"]) == dim_count, report["degraded_dimensions"]
    assert all("analysis_error" in f for f in report["degraded_dimensions"].values())

    return final_state


def main() -> int:
    validate_graph_run()
    validate_degraded_run()
    print("\n✓ Graph smoke test passed")
    return 0


def test_graph_runs_all_dimensions_and_builds_report() -> None:
    validate_graph_run()


def test_degraded_run_is_reported_as_untrustworthy() -> None:
    validate_degraded_run()


if __name__ == "__main__":
    raise SystemExit(main())
