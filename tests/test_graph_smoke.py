"""Smoke test for the compiled LangGraph pipeline.

This file is both:
- a runnable script: `python tests/test_graph_smoke.py`
- a pytest test module: `pytest tests/test_graph_smoke.py`

Mocks Pinecone retrieval and the Gemini LLM call so the full graph
(document_agent -> retrieval_agent -> analysis_agent loop -> build_report)
can be exercised without live API calls or credentials.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents import analysis_agent, retrieval_agent
from src.graph.build import build_graph, _new_state


def _mock_query_index(query: str, top_k: int | None = None, section_types=None, namespace: str = ""):
    if top_k is None:
        top_k = 5
    return [
        {
            "id": f"{namespace}-{i}",
            "text": "mock evidence text",
            "score": 0.85,
            "headings": ["Item 7"],
            "source_document": namespace,
            "source_page": i + 1,
        }
        for i in range(min(3, top_k))
    ]


class _FakeResponse:
    content = (
        '{"dimension": "d", "verdict": "LOW", "reasoning": "mock", '
        '"evidence": [], "flags": []}'
    )


class _FakeLLM:
    def invoke(self, messages: list[dict[str, str]]) -> _FakeResponse:
        return _FakeResponse()


def run_mocked_graph(company_name: str = "Shopify", filing_path: str = "data/parsed/shop-20251231.json") -> dict[str, Any]:
    """Run the compiled graph end-to-end with retrieval/LLM calls mocked out."""
    retrieval_agent.query_index = _mock_query_index
    analysis_agent._llm = _FakeLLM()

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

    # A healthy run carries no degraded dimensions and is marked trustworthy.
    assert report["degraded_dimensions"] == {}, report["degraded_dimensions"]
    assert report["trustworthy"] is True

    return final_state


def validate_degraded_run() -> dict[str, Any]:
    """Every dimension fails at the model call.

    Without flag reporting this produces four MEDIUM verdicts that look exactly
    like real analysis — the failure mode the report must make visible.
    """
    class _FailingLLM:
        def invoke(self, messages):
            raise RuntimeError("503 model overloaded")

    from src import config as _config
    orig_delay = _config.LLM_RETRY_BASE_SECONDS
    _config.LLM_RETRY_BASE_SECONDS = 0.0
    try:
        retrieval_agent.query_index = _mock_query_index
        analysis_agent._llm = _FailingLLM()
        compiled = build_graph()
        final_state = compiled.invoke(
            _new_state("Shopify", "data/parsed/shop-20251231.json"),
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
