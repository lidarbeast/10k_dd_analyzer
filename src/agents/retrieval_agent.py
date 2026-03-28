"""
Retrieval Agent for the M&A Due Diligence Analyzer.

Responsibilities:
  1. Take the current dimension's retrieval queries + target sections
  2. Run Pinecone hybrid search via pinecone_ops.query_index()
  3. Return top-k chunks with similarity scores
  4. Implement confidence retry: if max score < threshold, reformulate
     with alternate queries and retry once
"""

import logging

from src import config
from src.state import DDState
from src.indexing.pinecone_ops import query_index

_log = logging.getLogger(__name__)


def retrieve_evidence(
    queries: list[str],
    target_sections: list[str],
    namespace: str = "",
    top_k: int | None = None,
) -> tuple[list[dict], float]:
    """
    Runs multiple retrieval queries and merges results by highest score.

    Args:
        queries: List of retrieval query strings.
        target_sections: Preferred 10-K section types for score boosting.
        namespace: Pinecone namespace (filing stem).
        top_k: Number of final results to return.

    Returns:
        Tuple of (merged_results, max_score).
    """
    if top_k is None:
        top_k = config.RETRIEVAL_TOP_K

    # Collect results from all queries, de-duplicate by text
    seen_texts: set[str] = set()
    all_results: list[dict] = []

    for query in queries:
        results = query_index(
            query=query,
            top_k=top_k,
            section_types=target_sections,
            namespace=namespace,
        )
        for r in results:
            # De-duplicate based on text content (first 200 chars)
            text_key = r["text"][:200]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                all_results.append(r)

    # Sort by score descending and take top_k
    all_results.sort(key=lambda x: x["score"], reverse=True)
    top_results = all_results[:top_k]

    max_score = top_results[0]["score"] if top_results else 0.0
    return top_results, max_score


def run(state: DDState) -> DDState:
    """
    Retrieval Agent node function for LangGraph.

    Retrieves evidence for the current dimension. If confidence is below
    threshold, retries with alternate queries.
    """
    dim_index = state["current_dim_index"]
    dimension = state["dimensions"][dim_index]
    company = state["company_name"]
    filing_stem = state["filing_path"].rsplit("/", 1)[-1].rsplit(".", 1)[0]

    _log.info(f"Retrieval Agent: {dimension['name']} for {company}")

    # First retrieval pass with primary queries
    results, max_score = retrieve_evidence(
        queries=dimension["retrieval_queries"],
        target_sections=dimension["target_sections"],
        namespace=filing_stem,
    )
    attempts = 1

    # Confidence-based retry with alternate queries
    if max_score < config.CONFIDENCE_THRESHOLD and dimension.get("alternate_queries"):
        _log.info(
            f"  Low confidence ({max_score:.3f} < {config.CONFIDENCE_THRESHOLD}). "
            f"Retrying with alternate queries..."
        )
        alt_results, alt_max = retrieve_evidence(
            queries=dimension["alternate_queries"],
            target_sections=dimension["target_sections"],
            namespace=filing_stem,
        )
        # Merge: keep best results from both passes
        seen = {r["text"][:200] for r in results}
        for r in alt_results:
            if r["text"][:200] not in seen:
                results.append(r)
                seen.add(r["text"][:200])
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[: config.RETRIEVAL_TOP_K]
        max_score = max(max_score, alt_max)
        attempts = 2

    _log.info(
        f"  Retrieved {len(results)} chunks, "
        f"max score={max_score:.3f}, attempts={attempts}"
    )

    # Store retrieved chunks in state for the Analysis Agent
    # We attach them to a temporary key that the Analysis Agent consumes
    state["_retrieved_chunks"] = results
    state["_retrieval_score"] = max_score
    state["_retrieval_attempts"] = attempts

    return state
