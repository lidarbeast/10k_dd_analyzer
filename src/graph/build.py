"""
LangGraph wiring for the M&A Due Diligence Analyzer.

Graph shape:
  document_agent -> (retrieval_agent -> analysis_agent) * N dimensions -> build_report

This mirrors the sequential loop already verified in src/agents/local_runner.py,
replacing the manual while-loop with a LangGraph conditional edge.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src import config
from src.agents import analysis_agent, document_agent, retrieval_agent
from src.state import DDState

_log = logging.getLogger(__name__)


# Flags that mean the verdict is not a finding: the model was never consulted,
# its answer could not be used, or it reported that the evidence the rubric
# requires was not retrieved.  A rating carrying one of these must never be
# read as analysis.
DEGRADED_FLAGS = frozenset({
    "no_evidence",
    "parse_error",
    "analysis_error",
    "inconclusive",
    "invalid_rating",
})


def summarize_flags(verdicts: list) -> tuple[dict, dict]:
    """Split verdict flags into all-flags and degraded-only views.

    Returns (flags, degraded) keyed by dimension_id.  `degraded` is what makes a
    silently failing run visible: without it, four analysis_error verdicts and
    four genuine MEDIUM verdicts produce an identical report.
    """
    flags = {v["dimension"]: list(v.get("flags") or []) for v in verdicts if v.get("flags")}
    degraded = {
        dim: sorted(set(f) & DEGRADED_FLAGS)
        for dim, f in flags.items()
        if set(f) & DEGRADED_FLAGS
    }
    return flags, degraded


def _build_report(state: DDState) -> DDState:
    verdicts = state["verdicts"]
    flags, degraded = summarize_flags(verdicts)

    if degraded:
        _log.warning(
            "%d of %d dimensions produced a degraded verdict (no model finding): %s",
            len(degraded), len(verdicts), degraded,
        )

    state["report"] = {
        "company_name": state["company_name"],
        "filing_path": state["filing_path"],
        "rubric_version": config.RUBRIC_VERSION,
        "dimension_count": len(verdicts),
        "ratings": {v["dimension"]: v["rating"] for v in verdicts},
        "flags": flags,
        "degraded_dimensions": degraded,
        "trustworthy": not degraded,
        "retrieval_audits": state.get("_retrieval_audits", {}),
        "warnings": state.get("_warnings", []),
    }
    return state


def _route_after_analysis(state: DDState) -> str:
    """Loop back to retrieval_agent while dimensions remain, else build the report."""
    if state["current_dim_index"] < len(state["dimensions"]):
        return "retrieval_agent"
    return "build_report"


def build_graph() -> CompiledStateGraph:
    """Compile the due diligence LangGraph pipeline."""
    graph = StateGraph(DDState)

    graph.add_node("document_agent", document_agent.run)
    graph.add_node("retrieval_agent", retrieval_agent.run)
    graph.add_node("analysis_agent", analysis_agent.run)
    graph.add_node("build_report", _build_report)

    graph.add_edge(START, "document_agent")
    graph.add_edge("document_agent", "retrieval_agent")
    graph.add_edge("retrieval_agent", "analysis_agent")
    graph.add_conditional_edges(
        "analysis_agent",
        _route_after_analysis,
        {"retrieval_agent": "retrieval_agent", "build_report": "build_report"},
    )
    graph.add_edge("build_report", END)

    return graph.compile()


def _new_state(company_name: str, filing_path: str) -> DDState:
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


def run_graph(company_name: str, filing_path: str) -> DDState:
    """Compile and invoke the graph end-to-end for a single filing."""
    compiled = build_graph()
    state = _new_state(company_name, filing_path)
    return compiled.invoke(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the due diligence LangGraph pipeline.")
    parser.add_argument("--company", required=True, help="Company name")
    parser.add_argument("--filing", required=True, help="Path to filing PDF (or logical filing path)")
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output JSON path for full final state",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    final_state = run_graph(company_name=args.company, filing_path=args.filing)

    print("=== Graph Run Summary ===")
    print(json.dumps(final_state["report"], indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(final_state, indent=2), encoding="utf-8")
        _log.info("Wrote full state to %s", out_path)


if __name__ == "__main__":
    main()
