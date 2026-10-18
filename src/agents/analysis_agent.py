"""
Analysis Agent for the M&A Due Diligence Analyzer.

Responsibilities:
  1. Receive retrieved chunks from the Retrieval Agent
  2. Apply the dimension-specific rubric and prompt template
  3. Call Gemini LLM to evaluate evidence against S&P thresholds
  4. Output a structured JSON VerdictCard
  5. Accumulate verdicts in DDState
"""

import json
import logging
import time
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from src import config
from src.state import DDState, VerdictCard

_log = logging.getLogger(__name__)

# Generous cap on guidance items per category: high enough that no curated
# signal in a well-formed config is silently dropped, bounded enough to catch a
# runaway/malformed field.  Token budget is not a constraint here (Gemini's
# context window dwarfs the full guidance block), so we do not truncate the
# per-item text — only bound the item count.
_MAX_GUIDANCE_ITEMS = 12

# Module-level singleton — avoids re-initialising the client on every dimension.
_llm: ChatGoogleGenerativeAI | None = None


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            google_api_key=config.GOOGLE_API_KEY,
            timeout=config.LLM_TIMEOUT_SECONDS,
            # Retry is handled explicitly in _invoke_with_retry so that each
            # attempt is logged; the client's own retry would be silent.
            max_retries=0,
        )
    return _llm


def _invoke_with_retry(messages: list[dict], dimension_id: str):
    """Invoke the analysis model with bounded retries and exponential backoff.

    Mirrors `_get_embeddings_with_retry` in src/indexing/pinecone_ops.py.  Without
    this a single transient failure — a rate limit on the free tier, most likely —
    becomes a MEDIUM verdict that reads exactly like a real one.
    """
    max_retries = max(1, config.LLM_MAX_RETRIES)
    base_delay = max(0.0, config.LLM_RETRY_BASE_SECONDS)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return _get_llm().invoke(messages)
        except Exception as e:
            last_error = e
            if attempt == max_retries:
                break

            sleep_seconds = base_delay * (2 ** (attempt - 1))
            _log.warning(
                "Analysis call failed for %s (attempt %s/%s): %s. Retrying in %.1fs",
                dimension_id,
                attempt,
                max_retries,
                e,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(
        f"Analysis model call failed for {dimension_id} after {max_retries} attempt(s)."
    ) from last_error


def _collapse_ws(value: Any) -> str:
    """Collapse runs of whitespace/newlines to single spaces, WITHOUT truncating.

    Used for guidance text that must reach the model in full — the notes carry
    operational thresholds (e.g. "above ~10%", "two or more consecutive years")
    that truncation would cut, degrading classification.
    """
    return " ".join(str(value).split())


def _chunk_section_label(chunk: dict) -> str:
    """Best-effort 10-K section label for an evidence block.

    Retrieval chunks carry section identity in `headings` (e.g.
    "Item 7: Management's Discussion..."). `label` is a chunk-type tag
    ("text"/"document_index"), NOT the section, so it must not be used here.
    Falls back to "unknown" when a chunk has no headings.
    """
    headings = chunk.get("headings") or []
    if headings:
        return str(headings[0])
    return "unknown"


def _coerce_to_text_list(value: Any, *, max_items: int = _MAX_GUIDANCE_ITEMS) -> list[str]:
    if not isinstance(value, list):
        return []

    lines: list[str] = []
    for item in value[:max_items]:
        if isinstance(item, dict):
            signal = item.get("signal") or item.get("item") or item.get("guard")
            note = item.get("what_to_look_for") or item.get("rationale") or item.get("explanation")
            if signal and note:
                lines.append(f"- {_collapse_ws(signal)}: {_collapse_ws(note)}")
            elif signal:
                lines.append(f"- {_collapse_ws(signal)}")
            elif note:
                lines.append(f"- {_collapse_ws(note)}")
        elif isinstance(item, str):
            lines.append(f"- {_collapse_ws(item)}")
    return lines


def _build_dimension_guidance(dimension: dict) -> str:
    lines: list[str] = []

    mandatory_items = dimension.get("mandatory_10k_items", [])
    mandatory_lines = _coerce_to_text_list(mandatory_items)
    if mandatory_lines:
        lines.append("## Dimension Guidance")
        lines.append("### Mandatory 10-K Areas")
        lines.extend(mandatory_lines)

    evidence_schema = dimension.get("evidence_schema", {})
    if isinstance(evidence_schema, dict):
        primary = _coerce_to_text_list(evidence_schema.get("primary_signals"))
        corroborating = _coerce_to_text_list(evidence_schema.get("corroborating_signals"))
        absence = _coerce_to_text_list(evidence_schema.get("absence_signals"))
        # A guard renders as "guard: explanation"; its "pattern" field is
        # documentation only, so an explanation must name its own examples.
        guards = _coerce_to_text_list(evidence_schema.get("false_positive_guards"))
        cross_section = _coerce_to_text_list(evidence_schema.get("cross_section_evidence"))

        if primary:
            if not lines:
                lines.append("## Dimension Guidance")
            lines.append("### Primary Signals")
            lines.extend(primary)
        if corroborating:
            if not lines:
                lines.append("## Dimension Guidance")
            lines.append("### Corroborating Signals")
            lines.extend(corroborating)
        if absence:
            if not lines:
                lines.append("## Dimension Guidance")
            lines.append("### Absence / Low-Risk Signals")
            lines.extend(absence)
        if guards:
            if not lines:
                lines.append("## Dimension Guidance")
            lines.append("### False Positive Guards")
            lines.extend(guards)
        if cross_section:
            if not lines:
                lines.append("## Dimension Guidance")
            lines.append("### Cross-Section Evidence Checks")
            lines.extend(cross_section)

        # String-valued note (not a list) — helps the model recognise how the
        # evidence actually names things (e.g. "Senior Notes"/"Term Loan" rather
        # than "long-term debt"), so it must reach analysis, not just retrieval.
        terminology_note = (
            evidence_schema.get("terminology_note")
            or evidence_schema.get("debt_terminology_note")
        )
        if isinstance(terminology_note, str) and terminology_note.strip():
            if not lines:
                lines.append("## Dimension Guidance")
            lines.append("### Terminology Note")
            lines.append(f"- {_collapse_ws(terminology_note)}")

    rubric = dimension.get("rubric", {})
    if isinstance(rubric, dict):
        rubric_lines: list[str] = []
        for level in ("HIGH", "MEDIUM", "LOW"):
            level_data = rubric.get(level)
            if isinstance(level_data, dict):
                threshold = level_data.get("threshold")
                if threshold:
                    # Full threshold text (whitespace-collapsed, not truncated):
                    # this guidance block is the single source of the rubric in
                    # the prompt, so it must carry the complete criteria.
                    line = f"- {level}: {_collapse_ws(threshold)}"
                    frp = level_data.get("sp_frp_equivalent")
                    if frp:
                        line += f"  [S&P FRP: {_collapse_ws(frp)}]"
                    rubric_lines.append(line)
        if rubric_lines:
            if not lines:
                lines.append("## Dimension Guidance")
            lines.append("### Rubric Thresholds")
            lines.extend(rubric_lines)

    no_evidence_handling = dimension.get("no_evidence_handling")
    if isinstance(no_evidence_handling, str) and no_evidence_handling.strip():
        if not lines:
            lines.append("## Dimension Guidance")
        lines.append("### No Evidence Handling")
        lines.append(f"- {_collapse_ws(no_evidence_handling)}")

    return "\n".join(lines).strip()


# Model answers meaning "the rubric's required evidence was not in the
# retrieved passages".  The rubrics ask for INCONCLUSIVE in exactly that case,
# so it must surface as a degraded verdict, not as the MEDIUM it coerces to.
_INCONCLUSIVE_ANSWERS = frozenset({"INCONCLUSIVE", "UNKNOWN"})

# Synonyms that still name a definite rating.
_RATING_ALIASES = {
    "MODERATE": config.RATING_MEDIUM,
    "SEVERE": config.RATING_HIGH,
    "MINOR": config.RATING_LOW,
}


def _coerce_rating(raw_rating: str) -> str:
    normalized = (raw_rating or "").strip().upper()
    if normalized in config.VALID_RATINGS:
        return normalized
    return _RATING_ALIASES.get(normalized, config.RATING_MEDIUM)


def _rating_flag(raw_rating: str) -> str | None:
    """Flag a model answer that is not a definite rating.

    Returns "inconclusive" for an explicit can't-tell answer, "invalid_rating"
    for anything else `_coerce_rating` had to replace with its MEDIUM default,
    and None when the answer named a rating.
    """
    normalized = (raw_rating or "").strip().upper()
    if normalized in config.VALID_RATINGS or normalized in _RATING_ALIASES:
        return None
    if normalized in _INCONCLUSIVE_ANSWERS:
        return "inconclusive"
    return "invalid_rating"


def _build_prompt(dimension: dict, company_name: str, retrieved_chunks: list[dict]) -> tuple[str, str]:
    """
    Builds the system and user prompts from the dimension config template.

    Args:
        dimension: DimensionSpec dict with prompt_template.
        company_name: Company name for template substitution.
        retrieved_chunks: Retrieved evidence chunks.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    template = dimension.get("prompt_template", {})
    system_prompt = template.get("system", "You are a financial due diligence analyst.")

    # Format retrieved chunks as numbered evidence blocks
    chunks_text = ""
    for i, chunk in enumerate(retrieved_chunks, 1):
        section = _chunk_section_label(chunk)
        score = chunk.get("score", 0.0)
        text = chunk.get("text", "")
        chunks_text += (
            f"### Evidence {i} [{section}] (score: {score:.3f})\n"
            f"{text}\n\n"
        )

    user_prompt = template.get("user_template", "Evaluate {company_name}.\n\n{retrieved_chunks}")
    user_prompt = user_prompt.replace("{company_name}", company_name)
    user_prompt = user_prompt.replace("{retrieved_chunks}", chunks_text)

    guidance_text = _build_dimension_guidance(dimension)
    if guidance_text:
        # Support explicit placeholders in template, with backwards-compatible append.
        if "{dimension_guidance}" in user_prompt:
            user_prompt = user_prompt.replace("{dimension_guidance}", guidance_text)
        else:
            user_prompt = f"{user_prompt}\n\n{guidance_text}"

        if "{evidence_schema_summary}" in user_prompt:
            user_prompt = user_prompt.replace("{evidence_schema_summary}", guidance_text)
        if "{rubric_summary}" in user_prompt:
            user_prompt = user_prompt.replace("{rubric_summary}", guidance_text)

    return system_prompt, user_prompt


def _parse_verdict(response_text: str, dimension_id: str) -> dict[str, Any]:
    """
    Parses the LLM response into a structured verdict dict.
    Handles JSON extraction from markdown code blocks.
    """
    # Try to extract JSON from code block
    text = response_text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1]
        text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        text = text.split("```", 1)[0]

    try:
        parsed = json.loads(text.strip())
        return parsed
    except json.JSONDecodeError: 
        _log.warning(f"Failed to parse LLM response as JSON for {dimension_id}")
        # Return a minimal structure from the raw text
        return {
            "dimension": dimension_id,
            "verdict": "MEDIUM",
            "confidence": 0.0,
            "reasoning": response_text[:500],
            "evidence": [],
            "flags": ["parse_error"],
        }


def analyze_dimension(
    dimension: dict,
    company_name: str,
    retrieved_chunks: list[dict],
    retrieval_score: float,
    retrieval_attempts: int,
) -> VerdictCard:
    """
    Runs the Analysis Agent for a single dimension.

    Args:
        dimension: DimensionSpec dict.
        company_name: Company being analyzed.
        retrieved_chunks: Evidence chunks from Retrieval Agent.
        retrieval_score: Mean similarity score of the top 3 retrieved chunks.
        retrieval_attempts: Number of retrieval passes made.

    Returns:
        VerdictCard with structured verdict.
    """
    system_prompt, user_prompt = _build_prompt(dimension, company_name, retrieved_chunks)

    _log.info(f"Analysis Agent: evaluating {dimension['name']} for {company_name}")

    if not retrieved_chunks:
        no_evidence_inst = dimension.get("no_evidence_handling", "")
        # Read the fallback from the explicit `no_evidence_rating` field rather
        # than inferring it from `no_evidence_handling` prose.  That text states
        # intent by naming the rating it forbids ("do not default to LOW"), so a
        # keyword scan over it returns precisely the wrong answer.  When the
        # field is absent, default to MEDIUM — "we could not tell" — and never
        # LOW, which would report a dimension with no evidence as low risk.
        raw_rating = dimension.get("no_evidence_rating") or config.RATING_MEDIUM
        reasoning = (
            "No evidence was retrieved for this dimension, so no model analysis "
            "was performed and this rating is a configured fallback, not a finding."
        )
        if no_evidence_inst:
            reasoning += f" Config guidance: {no_evidence_inst}"
        parsed = {
            "dimension": dimension["dimension_id"],
            "verdict": raw_rating,
            "reasoning": reasoning,
            "evidence": [],
            "flags": ["no_evidence"],
        }
    else:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "human", "content": user_prompt},
            ]
            response = _invoke_with_retry(messages, dimension["dimension_id"])
            response_text = response.content
            parsed = _parse_verdict(str(response_text), dimension["dimension_id"])
        except Exception as e:
            _log.warning(
                "Analysis model call failed for %s: %s",
                dimension["dimension_id"],
                e,
            )
            parsed = {
                "dimension": dimension["dimension_id"],
                "verdict": "MEDIUM",
                "reasoning": f"Analysis model call failed: {e}",
                "evidence": [],
                "flags": ["analysis_error"],
            }

    raw_rating = str(parsed.get("verdict", parsed.get("rating", "")) or "")
    flags = list(parsed.get("flags") or [])
    # Only a model answer is checked: the no-evidence path's configured
    # INCONCLUSIVE is already marked by its own `no_evidence` flag.
    if "no_evidence" not in flags:
        rating_flag = _rating_flag(raw_rating)
        if rating_flag and rating_flag not in flags:
            flags.append(rating_flag)

    # Build VerdictCard
    verdict: VerdictCard = {
        "dimension": dimension["dimension_id"],
        "rating": _coerce_rating(raw_rating),
        "confidence": retrieval_score,
        "reasoning": parsed.get("reasoning", ""),
        "evidence_citations": [
            {
                "text": e.get("source_passage", e.get("text", "")),
                "source_page": e.get("source_page", 0),
            }
            for e in parsed.get("evidence", [])
        ],
        "flags": flags,
        "retrieval_attempts": retrieval_attempts,
    }

    _log.info(
        f"  Verdict: {verdict['rating']} "
        f"(confidence={verdict['confidence']:.3f}, "
        f"flags={verdict['flags']})"
    )
    return verdict


def run(state: DDState) -> DDState:
    """
    Analysis Agent node function for LangGraph.

    Evaluates the current dimension using retrieved evidence,
    produces a VerdictCard, and advances the dimension index.
    """
    dim_index = state["current_dim_index"]
    dimensions = state.get("dimensions", [])
    if dim_index >= len(dimensions):
        _log.warning("Analysis Agent called with current_dim_index out of range; skipping.")
        return state

    dimension = dimensions[dim_index]
    company = state["company_name"]

    # Consume retrieved chunks from Retrieval Agent
    retrieved_chunks = state.pop("_retrieved_chunks", [])
    retrieval_score = state.pop("_retrieval_score", 0.0)
    retrieval_attempts_count = state.pop("_retrieval_attempts", 1)

    # Run analysis
    verdict = analyze_dimension(
        dimension=dimension,
        company_name=company,
        retrieved_chunks=retrieved_chunks,
        retrieval_score=retrieval_score,
        retrieval_attempts=retrieval_attempts_count,
    )

    # Accumulate in state
    state["verdicts"].append(verdict)
    state["retrieval_attempts"].append(retrieval_attempts_count)
    state["confidence_scores"].append(retrieval_score)

    # Advance dimension index
    state["current_dim_index"] = dim_index + 1

    _log.info(
        f"Dimension {dim_index + 1}/{len(dimensions)} complete: "
        f"{dimension['name']} → {verdict['rating']}"
    )
    return state
