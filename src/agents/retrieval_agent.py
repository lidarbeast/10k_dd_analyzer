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
import math
from pathlib import Path
import statistics

from src import config
from src.state import DDState
from src.indexing.pinecone_ops import query_index

_log = logging.getLogger(__name__)

# Fetch more candidates per query before the merge step so that relevant chunks
# ranking just outside top_k for a single query are not silently dropped.
_PER_QUERY_FETCH_MULTIPLIER = 2


def _warn(state: DDState, message: str) -> None:
    warnings = state.get("_warnings")
    if warnings is None:
        warnings = []
        state["_warnings"] = warnings
    warnings.append(message)
    _log.warning(message)


def _set_audit(state: DDState, dimension_id: str, audit: dict) -> None:
    """Record a dimension's retrieval audit.

    Written to two places: `_retrieval_audit` holds the current dimension only
    (the scratch field single-dimension callers and tests read), while
    `_retrieval_audits` accumulates every dimension keyed by dimension_id so the
    trail survives the graph's dimension loop instead of being overwritten.
    """
    state["_retrieval_audit"] = audit
    audits = state.get("_retrieval_audits")
    if audits is None:
        audits = {}
        state["_retrieval_audits"] = audits
    audits[dimension_id] = audit


def _retrieved_summary(chunks: list[dict]) -> list[dict]:
    """The final evidence list as the audit records it, in prompt order."""
    return [
        {
            "rank": rank,
            "id": c.get("id", ""),
            "source_page": c.get("source_page"),
            "score": round(float(c.get("score", 0.0)), 4),
            "query": c.get("query"),
        }
        for rank, c in enumerate(chunks, 1)
    ]


def _chunk_key(chunk: dict) -> str:
    vid = chunk.get("id")
    if vid:
        return str(vid)
    return "{}|{}|{}".format(
        chunk.get("source_document", ""),
        chunk.get("source_page", ""),
        chunk.get("text", "")[:200],
    )


def _merge_results_keep_best(results: list[dict], top_k: int) -> list[dict]:
    """De-duplicate a flat list of chunks, keeping the highest-scoring copy of
    each chunk and accumulating provenance across duplicates.  Used for the
    alternate-query retry merge where global score ranking is appropriate.
    """
    by_key: dict[str, dict] = {}
    for item in results:
        key = _chunk_key(item)
        if not key:
            continue
        prov_entry = {
            "query": item.get("query"),
            "score": item.get("score", 0.0),
            "source_document": item.get("source_document"),
            "source_page": item.get("source_page"),
        }
        existing = by_key.get(key)
        if existing is None:
            record = item.copy()
            record["provenance"] = [prov_entry]
            by_key[key] = record
        else:
            existing.setdefault("provenance", []).append(prov_entry)
            if item.get("score", 0.0) > existing.get("score", 0.0):
                for fld in ("text", "score", "headings", "source_document", "source_page", "id"):
                    if fld in item:
                        existing[fld] = item[fld]

    merged = list(by_key.values())
    merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return merged[:top_k]


def _merge_with_quota(
    query_results: list[tuple[str, list[dict]]],
    top_k: int,
    per_query_k: int,
) -> list[dict]:
    """Per-query quota allocation followed by score-based fill.

    Pass 1: take up to `per_query_k` unique chunks from each query in order.
            This guarantees every rubric facet is represented in the final pool.
    Pass 2: fill any remaining slots from the full candidate pool by score.

    The final list is sorted by score descending.
    """
    seen: set[str] = set()
    guaranteed: list[dict] = []

    for _, results in query_results:
        taken = 0
        for chunk in results:
            if taken >= per_query_k:
                break
            key = _chunk_key(chunk)
            if key and key not in seen:
                seen.add(key)
                guaranteed.append(chunk)
                taken += 1

    if len(guaranteed) < top_k:
        all_chunks = sorted(
            (c for _, r in query_results for c in r),
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        )
        for chunk in all_chunks:
            if len(guaranteed) >= top_k:
                break
            key = _chunk_key(chunk)
            if key and key not in seen:
                seen.add(key)
                guaranteed.append(chunk)

    guaranteed.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return guaranteed[:top_k]


def retrieve_evidence(
    queries: list[str],
    target_sections: list[str],
    namespace: str = "",
    top_k: int | None = None,
) -> tuple[list[dict], float, dict]:
    """Run each query independently and merge results using per-query quota
    allocation so every rubric facet is represented in the final pool.

    Args:
        queries: List of retrieval query strings (one per rubric facet).
        target_sections: Preferred 10-K section types for score boosting.
        namespace: Pinecone namespace (filing stem).
        top_k: Total number of chunks to return.

    Returns:
        Tuple of (final_chunks, confidence_score, per_query_stats).
    """
    if top_k is None:
        top_k = config.RETRIEVAL_TOP_K

    # Each query gets an equal quota of slots; leftover filled by score.
    per_query_k = math.ceil(top_k / len(queries)) if queries else top_k

    query_results: list[tuple[str, list[dict]]] = []
    per_query_stats: dict[str, dict] = {}

    for query in queries:
        try:
            results = query_index(
                query=query,
                top_k=per_query_k * _PER_QUERY_FETCH_MULTIPLIER,
                section_types=target_sections,
                namespace=namespace,
            )
            for r in results:
                r["query"] = query
            scores = [r.get("score", 0.0) for r in results]
            per_query_stats[query] = {
                "count": len(results),
                "max": max(scores) if scores else 0.0,
                "mean": statistics.mean(scores) if scores else 0.0,
            }
            query_results.append((query, results))
        except Exception as e:
            _log.warning("Retrieval query failed for '%s': %s", query, e)
            per_query_stats[query] = {"count": 0, "max": 0.0, "mean": 0.0}
            query_results.append((query, []))

    final = _merge_with_quota(query_results, top_k=top_k, per_query_k=per_query_k)
    top_scores = [r.get("score", 0.0) for r in final[:3]]
    confidence_score = statistics.mean(top_scores) if top_scores else 0.0
    return final, confidence_score, per_query_stats


def run(state: DDState) -> DDState:
    """
    Retrieval Agent node function for LangGraph.

    Retrieves evidence for the current dimension. If confidence is below
    threshold, retries with alternate queries.
    """
    dim_index = state["current_dim_index"]
    dimensions = state.get("dimensions", [])
    if dim_index >= len(dimensions):
        _warn(state, "Retrieval Agent called with current_dim_index out of range; skipping.")
        state["_retrieved_chunks"] = []
        state["_retrieval_score"] = 0.0
        state["_retrieval_attempts"] = 1
        return state

    dimension = dimensions[dim_index]
    company = state["company_name"]

    # The Document Agent confirmed this namespace holds the filing's vectors.
    filing_stem = state.get("namespace") or Path(state["filing_path"]).stem
    _log.info("Retrieval Agent: namespace='%s' for %s / %s", filing_stem, company, dimension["name"])

    # First retrieval pass with primary queries
    primary_queries = dimension.get("retrieval_queries", [])
    if not primary_queries:
        _warn(state, f"No retrieval_queries configured for {dimension['dimension_id']}")
        state["_retrieved_chunks"] = []
        state["_retrieval_score"] = 0.0
        state["_retrieval_attempts"] = 1
        return state

    results, confidence_score, primary_stats = retrieve_evidence(
        queries=primary_queries,
        target_sections=dimension["target_sections"],
        namespace=filing_stem,
    )
    attempts = 1
    merged_query_stats = dict(primary_stats)
    retry_score = confidence_score

    # Zero results is a fundamentally different failure from low scores:
    # it means a namespace mismatch or a total vocabulary miss, and alternate
    # queries will not fix it.  Warn loudly and skip the LLM call.
    if not results:
        _warn(
            state,
            f"Zero results for {dimension['dimension_id']} in namespace '{filing_stem}'. "
            "Verify the filing was indexed and filing_path stem matches the Pinecone namespace.",
        )
        state["_retrieved_chunks"] = []
        state["_retrieval_score"] = 0.0
        state["_retrieval_attempts"] = attempts
        _set_audit(state, dimension["dimension_id"], {
            "per_query": merged_query_stats,
            "confidence": {"score": 0.0, "max": 0.0},
            "outcome": "zero_results",
        })
        return state

    # Confidence-based retry with alternate queries.
    # confidence_floor can be overridden per-dimension in the config JSON to
    # account for language sparsity (e.g. going concern language is rare by design).
    retry_enabled = config.MAX_RETRIEVAL_RETRIES > 0
    alternate_queries = dimension.get("alternate_queries", [])
    confidence_threshold = dimension.get("confidence_floor", config.CONFIDENCE_THRESHOLD)

    if retry_enabled and confidence_score < confidence_threshold and alternate_queries:
        _log.info(
            "  Low confidence (%.3f < %.3f). Retrying with %d alternate queries...",
            confidence_score, confidence_threshold, len(alternate_queries),
        )
        alt_results, alt_confidence, alt_stats = retrieve_evidence(
            queries=alternate_queries,
            target_sections=dimension["target_sections"],
            namespace=filing_stem,
        )
        results = _merge_results_keep_best(results + alt_results, config.RETRIEVAL_TOP_K)
        attempts = 2
        merged_query_stats.update(alt_stats)
        top_scores = [r.get("score", 0.0) for r in results[:3]]
        retry_score = statistics.mean(top_scores) if top_scores else alt_confidence

    # Hard floor quality gate: if the best chunk is still below the noise floor
    # after retry, sending it to the LLM risks a hallucinated verdict.  Signal
    # no-evidence so the Analysis Agent applies no_evidence_handling instead.
    max_score = max((r.get("score", 0.0) for r in results), default=0.0)
    if max_score < config.RETRIEVAL_HARD_FLOOR:
        _log.info(
            "  Hard floor not met (max_score=%.3f < %.3f) after %d attempt(s). "
            "Signalling no-evidence to Analysis Agent.",
            max_score, config.RETRIEVAL_HARD_FLOOR, attempts,
        )
        state["_retrieved_chunks"] = []
        state["_retrieval_score"] = max_score
        state["_retrieval_attempts"] = attempts
        _set_audit(state, dimension["dimension_id"], {
            "per_query": merged_query_stats,
            "confidence": {"score": retry_score, "max": max_score},
            "outcome": "hard_floor_not_met",
        })
        return state

    _log.info(
        "  Retrieved %d chunks, confidence=%.3f, attempts=%d",
        len(results), retry_score, attempts,
    )

    _set_audit(state, dimension["dimension_id"], {
        "per_query": merged_query_stats,
        "confidence": {"score": retry_score, "max": max_score},
        "outcome": "ok",
        "retrieved": _retrieved_summary(results),
    })
    state["_retrieved_chunks"] = results
    state["_retrieval_score"] = retry_score
    state["_retrieval_attempts"] = attempts

    return state
