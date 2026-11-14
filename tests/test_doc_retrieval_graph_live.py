"""Live test for the document_agent -> retrieval_agent connection via LangGraph.

Exercises ONLY the document->retrieval handoff (no analysis_agent, no LLM call)
through the compiled partial graph, against the real Pinecone index. Verifies:

  - document_agent initialises state (dimensions loaded, indices reset)
  - the graph edge hands that state to retrieval_agent
  - retrieval_agent surfaces real chunks for dimension 0 from the live index

Requirements:
  - PINECONE_API_KEY and GOOGLE_API_KEY set in .env
  - shop-20251231 must be indexed in Pinecone

Run:
  python tests/test_doc_retrieval_graph_live.py
  pytest tests/test_doc_retrieval_graph_live.py -v -s   # skips if no keys
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src import config
from src.agents import document_agent, retrieval_agent
from src.graph.build import _new_state
from src.state import DDState

_COMPANY = "Shopify"
_FILING_PATH = "data/filings/shop-20251231.pdf"


def _missing_keys() -> list[str]:
    missing = []
    if not config.PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if config.EMBEDDING_PROVIDER == "google" and not config.GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    return missing


def _build_document_retrieval_graph() -> CompiledStateGraph:
    """Partial graph: document_agent -> retrieval_agent -> END.

    Test-only scaffolding. Stops after retrieval (no analysis_agent / no LLM
    call) so the document->retrieval handoff can be verified through LangGraph
    in isolation. Retrieval runs for the dimension at current_dim_index
    (0 after document_agent initialises state).
    """
    graph = StateGraph(DDState)

    graph.add_node("document_agent", document_agent.run)
    graph.add_node("retrieval_agent", retrieval_agent.run)

    graph.add_edge(START, "document_agent")
    graph.add_edge("document_agent", "retrieval_agent")
    graph.add_edge("retrieval_agent", END)

    return graph.compile()


def run_doc_retrieval_graph() -> dict[str, Any]:
    """Compile and invoke the document->retrieval partial graph live."""
    compiled = _build_document_retrieval_graph()
    state = _new_state(_COMPANY, _FILING_PATH)
    return compiled.invoke(state)


def validate_doc_retrieval_connection() -> dict[str, Any]:
    final_state = run_doc_retrieval_graph()

    dimensions = final_state.get("dimensions", [])
    chunks = final_state.get("_retrieved_chunks", [])
    score = final_state.get("_retrieval_score", 0.0)
    outcome = final_state.get("_retrieval_audit", {}).get("outcome", "unknown")
    attempts = final_state.get("_retrieval_attempts", 0)

    print("=== Document -> Retrieval Graph Connection (live) ===")
    print(f"  dimensions loaded   : {len(dimensions)}")
    print(f"  current_dim_index   : {final_state.get('current_dim_index')}")
    print(f"  retrieval outcome   : {outcome}")
    print(f"  retrieval score     : {score:.4f}  (hard_floor={config.RETRIEVAL_HARD_FLOOR})")
    print(f"  retrieval attempts  : {attempts}")
    print(f"  chunks retrieved    : {len(chunks)}")
    if chunks:
        top = chunks[0]
        snippet = top.get("text", "")[:160].replace("\n", " ")
        print(f"  top chunk           : [{top.get('source_page')}] {snippet}...")

    # document_agent ran: dimensions populated, loop index initialised at 0
    assert len(dimensions) > 0, "document_agent did not populate dimensions"
    assert final_state.get("current_dim_index") == 0, "current_dim_index should be 0 after document_agent"

    # retrieval_agent ran on the handed-off state and produced real evidence
    assert outcome != "zero_results", (
        "Retrieval returned zero_results — namespace mismatch or filing not indexed"
    )
    assert outcome == "ok", f"Unexpected retrieval outcome '{outcome}'"
    assert len(chunks) > 0, "Retrieval produced no chunks for dimension 0"
    assert score >= config.RETRIEVAL_HARD_FLOOR, (
        f"Score {score:.4f} below hard floor {config.RETRIEVAL_HARD_FLOOR}"
    )

    return final_state


def main() -> int:
    missing = _missing_keys()
    if missing:
        print(f"Cannot run live test: missing env vars: {missing}")
        return 1
    validate_doc_retrieval_connection()
    print("\n✓ Document -> Retrieval graph connection verified live")
    return 0


try:
    import pytest

    @pytest.mark.live
    def test_doc_retrieval_graph_live():
        if _missing_keys():
            pytest.skip(f"missing: {_missing_keys()}")
        validate_doc_retrieval_connection()

except ImportError:
    pass


if __name__ == "__main__":
    raise SystemExit(main())
