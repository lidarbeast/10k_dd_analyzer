"""Live integration test for the Retrieval Agent against the real Pinecone index.

Unlike the mocked harness (test_retrieval_agent_harness.py), this test makes
real embedding and Pinecone calls to verify that:

  - The indexed filing is actually queryable
  - Dimension queries surface relevant chunks (not noise)
  - Cosine scores are meaningful (above hard floor)
  - Namespaces match between indexing and retrieval

Requirements:
  - PINECONE_API_KEY and GOOGLE_API_KEY must be set in .env
  - shop-20251231.pdf must have been indexed (run pinecone_ops.py or run_pipeline.py)

Run:
  python tests/test_retrieval_agent_live.py            # all configured dimensions
  python tests/test_retrieval_agent_live.py --dim 0    # single dimension (fast)
  pytest tests/test_retrieval_agent_live.py -v -s      # via pytest (skips if no keys)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config

# ---------------------------------------------------------------------------
# Skip guard — checked before any Pinecone/Google calls
# ---------------------------------------------------------------------------

def _missing_keys() -> list[str]:
    missing = []
    if not config.PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if config.EMBEDDING_PROVIDER == "google" and not config.GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    return missing


_COMPANY = "Shopify"
_FILING_PATH = "data/filings/shop-20251231.pdf"


# ---------------------------------------------------------------------------
# Core assertion helper
# ---------------------------------------------------------------------------

def _assert_dimension(dim_index: int, *, verbose: bool = True) -> dict:
    """
    Run the Retrieval Agent for one dimension against the live index and
    return the outcome dict.  Raises AssertionError on failure.
    """
    from src.agents import retrieval_agent
    from src.agents.document_agent import load_dimensions

    dimensions = load_dimensions()
    assert dim_index < len(dimensions), f"dim_index {dim_index} out of range"
    dim = dimensions[dim_index]

    state: dict = {
        "company_name": _COMPANY,
        "filing_path": _FILING_PATH,
        "dimensions": dimensions,
        "current_dim_index": dim_index,
        "verdicts": [],
        "retrieval_attempts": [],
        "confidence_scores": [],
        "filing_chunks": [],
        "report": {},
    }

    state = retrieval_agent.run(state)

    outcome = state.get("_retrieval_audit", {}).get("outcome", "unknown")
    score = state.get("_retrieval_score", 0.0)
    chunks = state.get("_retrieved_chunks", [])
    attempts = state.get("_retrieval_attempts", 0)

    if verbose:
        print(f"\n  Dimension [{dim_index}] {dim['name']}")
        print(f"    outcome   : {outcome}")
        print(f"    score     : {score:.4f}  (hard_floor={config.RETRIEVAL_HARD_FLOOR})")
        print(f"    attempts  : {attempts}")
        print(f"    chunks    : {len(chunks)}")
        # Per-query max/mean are what you need to set a per-dimension
        # `confidence_floor`: a dimension whose queries all top out well below
        # CONFIDENCE_THRESHOLD is retrying on every run for no benefit.
        per_query = state.get("_retrieval_audit", {}).get("per_query", {})
        if per_query:
            print(f"    per-query : (threshold={config.CONFIDENCE_THRESHOLD})")
            for q, stats in per_query.items():
                print(
                    f"      max={stats['max']:.4f} mean={stats['mean']:.4f} "
                    f"n={stats['count']:<3} {q[:60]}"
                )
        if chunks:
            top = chunks[0]
            snippet = top.get("text", "")[:200].replace("\n", " ")
            print(f"    top chunk : [{top.get('source_page')}] {snippet}...")

    # Assertions
    assert outcome != "zero_results", (
        f"Zero results for {dim['name']}. "
        "Check that shop-20251231 was indexed and the namespace matches."
    )
    assert outcome in ("ok", "hard_floor_not_met"), (
        f"Unexpected outcome '{outcome}' for {dim['name']}"
    )

    if outcome == "ok":
        assert len(chunks) > 0, f"outcome='ok' but no chunks returned for {dim['name']}"
        assert score >= config.RETRIEVAL_HARD_FLOOR, (
            f"Score {score:.4f} is below hard floor {config.RETRIEVAL_HARD_FLOOR} "
            f"but outcome was 'ok' — logic inconsistency"
        )
    elif outcome == "hard_floor_not_met":
        # Not a test failure — it means evidence is genuinely sparse for this
        # dimension in this filing, which is valid.  Just surface it clearly.
        print("    NOTE: hard floor not met — Analysis Agent will use no_evidence_handling")

    return {
        "dim_index": dim_index,
        "name": dim["name"],
        "outcome": outcome,
        "score": score,
        "chunks": len(chunks),
        "attempts": attempts,
    }


# ---------------------------------------------------------------------------
# Runners (script-mode helpers; not pytest-collected — see entry-points below)
# ---------------------------------------------------------------------------

def run_all_dimensions_live(verbose: bool = True) -> list[dict]:
    """Run every configured dimension and assert none returns zero_results."""
    missing = _missing_keys()
    if missing:
        print(f"SKIP: missing env vars: {missing}")
        return []

    from src.agents.document_agent import load_dimensions

    print(f"\n=== Live Retrieval Test: {_COMPANY} ({_FILING_PATH}) ===")

    results = []
    for i in range(len(load_dimensions())):
        result = _assert_dimension(i, verbose=verbose)
        results.append(result)

    # Summary table
    print("\n  Summary")
    print(f"  {'Dim':<4} {'Name':<40} {'Outcome':<20} {'Score':<8} {'Chunks':<7} {'Attempts'}")
    print("  " + "-" * 90)
    for r in results:
        print(
            f"  {r['dim_index']:<4} {r['name']:<40} {r['outcome']:<20} "
            f"{r['score']:<8.4f} {r['chunks']:<7} {r['attempts']}"
        )

    ok_count = sum(1 for r in results if r["outcome"] == "ok")
    floor_count = sum(1 for r in results if r["outcome"] == "hard_floor_not_met")
    print(f"\n  ok={ok_count}  hard_floor_not_met={floor_count}  zero_results=0")
    return results


def run_single_dimension_live(dim_index: int = 0) -> None:
    """Quick single-dimension smoke test — useful for fast iteration."""
    missing = _missing_keys()
    if missing:
        print(f"SKIP: missing env vars: {missing}")
        return

    print(f"\n=== Live Retrieval Smoke Test: dim={dim_index} ===")
    _assert_dimension(dim_index, verbose=True)


# ---------------------------------------------------------------------------
# pytest entry-points (skip if no keys)
# ---------------------------------------------------------------------------

try:
    import pytest

    @pytest.mark.live
    def test_live_retrieval_d0():
        if _missing_keys():
            pytest.skip(f"missing: {_missing_keys()}")
        _assert_dimension(0, verbose=True)

    @pytest.mark.live
    def test_live_retrieval_all():
        if _missing_keys():
            pytest.skip(f"missing: {_missing_keys()}")
        run_all_dimensions_live(verbose=True)

except ImportError:
    pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Live integration test for the Retrieval Agent")
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Run a single dimension by index. Omit to run all configured dimensions.",
    )
    args = parser.parse_args()

    missing = _missing_keys()
    if missing:
        print(f"Cannot run live test: missing env vars: {missing}")
        print("Set them in .env and retry.")
        return 1

    if args.dim is not None:
        run_single_dimension_live(args.dim)
    else:
        run_all_dimensions_live(verbose=True)

    print("\nLive test complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
