"""
Local runner for agent-first verification.

Runs:
  Document Agent -> (Retrieval Agent -> Analysis Agent) * N dimensions

This allows verifying agent behavior before LangGraph wiring.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.agents import analysis_agent, document_agent, retrieval_agent
from src.state import DDState

_log = logging.getLogger(__name__)


def _new_state(company_name: str, filing_path: str) -> DDState:
    # Minimal input state; Document Agent fills the rest.
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


def run_agent_chain(
    company_name: str,
    filing_path: str,
    max_dimensions: int | None = None,
) -> DDState:
    """Run all agents sequentially for verification."""
    state = _new_state(company_name, filing_path)
    state = document_agent.run(state)

    total_dims = len(state["dimensions"])
    if max_dimensions is not None:
        total_dims = min(total_dims, max_dimensions)

    while state["current_dim_index"] < total_dims:
        state = retrieval_agent.run(state)
        state = analysis_agent.run(state)

    state["report"] = {
        "company_name": state["company_name"],
        "filing_path": state["filing_path"],
        "dimension_count": len(state["verdicts"]),
        "ratings": {v["dimension"]: v["rating"] for v in state["verdicts"]},
        "warnings": state.get("_warnings", []),
    }
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent chain locally without LangGraph.")
    parser.add_argument("--company", required=True, help="Company name")
    parser.add_argument("--filing", required=True, help="Path to filing PDF (or logical filing path)")
    parser.add_argument(
        "--max-dimensions",
        type=int,
        default=None,
        help="Optional cap for dimensions to run (for quick checks)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output JSON path for full final state",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    final_state = run_agent_chain(
        company_name=args.company,
        filing_path=args.filing,
        max_dimensions=args.max_dimensions,
    )

    print("=== Agent Verification Summary ===")
    print(json.dumps(final_state["report"], indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(final_state, indent=2), encoding="utf-8")
        _log.info("Wrote full state to %s", out_path)


if __name__ == "__main__":
    main()
