"""
Analysis Agent for the M&A Due Diligence Analyzer.

Responsibilities:
  1. Receive retrieved chunks from the Retrieval Agent
  2. Render them as delimited, identified evidence blocks (E1, E2, ...) under
     the dimension's rubric and guidance
  3. Call the configured model (Gemini or Claude) with the dimension's JSON
     output schema
  4. Check every quoted passage against the evidence block it cites
  5. Output a structured VerdictCard and accumulate it in DDState
"""

from dataclasses import dataclass, field
import json
import logging
import re
import time
import unicodedata
from typing import Any, Callable

import anthropic
from google import genai
from google.genai import errors as genai_errors, types as genai_types
import httpx

from src import config
from src.state import DDState, VerdictCard

_log = logging.getLogger(__name__)

# Generous cap on guidance items per category: high enough that no curated
# signal in a well-formed config is silently dropped, bounded enough to catch a
# runaway/malformed field.  Token budget is not a constraint here (the model's
# context window dwarfs the full guidance block), so we do not truncate the
# per-item text — only bound the item count.
_MAX_GUIDANCE_ITEMS = 12

# Server-side refusal fallback ("default" routing, Claude only).  If a safety
# classifier declines the request, the API re-runs it on Anthropic's recommended
# model for that refusal category; the verdict then records which model answered.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Gemini finish reasons meaning the model stopped without answering.  RECITATION
# matters here: the prompts ask for verbatim quotes from filings.
_GEMINI_BLOCKED = frozenset({"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"})

# Gemini's 429 says how long to wait: "Please retry in 25.856440294s."
_RETRY_HINT = re.compile(r"retry in ([0-9.]+)s", re.IGNORECASE)
_MAX_RETRY_WAIT_SECONDS = 120.0

# Module-level singletons — avoid re-initialising a client on every dimension.
_client: anthropic.Anthropic | None = None
_gemini_client: genai.Client | None = None


def _check_llm_config() -> None:
    llm_errors = config.validate_llm_config()
    if llm_errors:
        raise RuntimeError("; ".join(llm_errors))


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _check_llm_config()
        _client = anthropic.Anthropic(
            timeout=config.LLM_TIMEOUT_SECONDS,
            # Retry is handled explicitly in _invoke_with_retry so that each
            # attempt is logged; the client's own retry would be silent.
            max_retries=0,
        )
    return _client


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _check_llm_config()
        _gemini_client = genai.Client(
            api_key=config.GOOGLE_API_KEY,
            http_options=genai_types.HttpOptions(timeout=int(config.LLM_TIMEOUT_SECONDS * 1000)),
        )
    return _gemini_client


@dataclass
class ModelReply:
    """What one analysis call returned, independent of either SDK's types."""
    text: str
    stop_reason: str              # "end_turn" | "max_tokens" | "refusal" | provider-specific
    model: str
    fallback_used: bool = False
    usage: dict = field(default_factory=dict)
    stop_detail: str = ""         # why the model stopped without answering, if known


def _is_retryable(error: Exception) -> bool:
    """Rate limits, server errors and network failures are transient; any
    other API error (bad request, auth, unknown model) will fail again, and so
    will a free-tier daily quota until it resets."""
    if isinstance(error, (anthropic.RateLimitError, anthropic.APIConnectionError, httpx.TransportError)):
        return True  # APITimeoutError is an APIConnectionError
    if isinstance(error, anthropic.APIStatusError):
        return error.status_code >= 500
    if isinstance(error, genai_errors.APIError):
        if error.code == 429:
            return "PerDay" not in str(error)
        return error.code >= 500
    return False


def _retry_delay(error: Exception, attempt: int, base_delay: float) -> float:
    """Seconds to wait before the next attempt: the server's own "retry in Xs"
    when it gives one (a retry sooner just spends the per-minute quota),
    otherwise exponential backoff."""
    hint = _RETRY_HINT.search(str(error))
    if hint:
        return min(float(hint.group(1)) + 1.0, _MAX_RETRY_WAIT_SECONDS)
    return base_delay * (2 ** (attempt - 1))


def _invoke_with_retry(call: Callable[[], Any], dimension_id: str):
    """Invoke the analysis model with bounded retries and exponential backoff.

    Mirrors `_get_embeddings_with_retry` in src/indexing/pinecone_ops.py.  Without
    this a single transient failure — a rate limit, most likely — becomes a
    verdict that reads exactly like a real one.  Non-retryable errors raise
    immediately.
    """
    max_retries = max(1, config.LLM_MAX_RETRIES)
    base_delay = max(0.0, config.LLM_RETRY_BASE_SECONDS)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return call()
        except Exception as e:
            last_error = e
            if not _is_retryable(e) or attempt == max_retries:
                break

            sleep_seconds = _retry_delay(e, attempt, base_delay)
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
        f"Analysis model call failed for {dimension_id} after {attempt} attempt(s): {last_error}"
    ) from last_error


def _usage_dict(usage: Any) -> dict:
    if usage is None:
        return {}
    out = {}
    for name in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        value = getattr(usage, name, None)
        if isinstance(value, int):
            out[name] = value
    return out


def _call_claude(
    system_prompt: str,
    user_prompt: str,
    output_schema: dict | None,
    dimension_id: str,
) -> ModelReply:
    """One request to the Anthropic API.  The reply's text is the JSON answer;
    thinking blocks are skipped (their text is empty by default and is not the
    answer)."""
    output_config: dict[str, Any] = {"effort": config.LLM_EFFORT}
    if output_schema:
        output_config["format"] = {"type": "json_schema", "schema": output_schema}

    request = {
        "model": config.LLM_MODEL,
        "max_tokens": config.LLM_MAX_TOKENS,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "output_config": output_config,
        "betas": [_FALLBACK_BETA],
        "fallbacks": "default",
    }
    response = _invoke_with_retry(lambda: _get_client().beta.messages.create(**request), dimension_id)

    text = "".join(
        getattr(block, "text", "") or ""
        for block in (response.content or [])
        if getattr(block, "type", None) == "text"
    )
    usage = getattr(response, "usage", None)
    iterations = getattr(usage, "iterations", None) or []
    fallback_used = any(getattr(it, "type", None) == "fallback_message" for it in iterations)
    return ModelReply(
        text=text,
        stop_reason=str(getattr(response, "stop_reason", "") or ""),
        model=str(getattr(response, "model", "") or ""),
        fallback_used=fallback_used,
        usage=_usage_dict(usage),
    )


def _call_gemini(
    system_prompt: str,
    user_prompt: str,
    output_schema: dict | None,
    dimension_id: str,
) -> ModelReply:
    """One request to the Gemini API, normalised to the same ModelReply.  The
    answer is read from the non-thought text parts of the first candidate."""
    request_config: dict[str, Any] = {
        "system_instruction": system_prompt,
        "temperature": config.LLM_TEMPERATURE,
        "max_output_tokens": config.LLM_MAX_TOKENS,
    }
    if output_schema:
        request_config["response_mime_type"] = "application/json"
        request_config["response_json_schema"] = output_schema

    response = _invoke_with_retry(
        lambda: _get_gemini_client().models.generate_content(
            model=config.LLM_MODEL,
            contents=user_prompt,
            config=request_config,
        ),
        dimension_id,
    )

    candidates = response.candidates or []
    parts = (candidates[0].content.parts or []) if candidates and candidates[0].content else []
    text = "".join(p.text for p in parts if p.text and not p.thought)

    finish = candidates[0].finish_reason if candidates else None
    finish_name = str(getattr(finish, "name", finish) or "")
    block = getattr(response.prompt_feedback, "block_reason", None)
    block_name = str(getattr(block, "name", block) or "")
    if block_name:
        stop_reason, stop_detail = "refusal", block_name
    elif finish_name in _GEMINI_BLOCKED:
        stop_reason, stop_detail = "refusal", finish_name
    elif finish_name == "MAX_TOKENS":
        stop_reason, stop_detail = "max_tokens", ""
    else:
        stop_reason, stop_detail = "end_turn" if finish_name == "STOP" else finish_name, ""

    meta = response.usage_metadata
    usage = {
        "input_tokens": getattr(meta, "prompt_token_count", None),
        "output_tokens": getattr(meta, "candidates_token_count", None),
        "thinking_tokens": getattr(meta, "thoughts_token_count", None),
    }
    return ModelReply(
        text=text,
        stop_reason=stop_reason,
        model=str(response.model_version or config.LLM_MODEL),
        usage={k: v for k, v in usage.items() if isinstance(v, int)},
        stop_detail=stop_detail,
    )


def _call_model(
    system_prompt: str,
    user_prompt: str,
    output_schema: dict | None,
    dimension_id: str,
) -> ModelReply:
    """One analysis request to the provider that serves LLM_MODEL."""
    if config.llm_provider() == "anthropic":
        return _call_claude(system_prompt, user_prompt, output_schema, dimension_id)
    if config.llm_provider() == "google":
        return _call_gemini(system_prompt, user_prompt, output_schema, dimension_id)
    _check_llm_config()
    raise RuntimeError(f"No provider for LLM_MODEL={config.LLM_MODEL!r}")


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

        # A boilerplate entry's quoted wording is its point, so unlike a guard
        # its "pattern" field is rendered.
        boilerplate = [
            f'- "{_collapse_ws(item["pattern"])}": {_collapse_ws(item["classification"])}. '
            f'{_collapse_ws(item["explanation"])}'
            for item in (evidence_schema.get("boilerplate_low_patterns") or [])[:_MAX_GUIDANCE_ITEMS]
        ]
        if boilerplate:
            if not lines:
                lines.append("## Dimension Guidance")
            lines.append("### Boilerplate Language")
            lines.extend(boilerplate)

        temporal_scope_note = evidence_schema.get("temporal_scope_note")
        if isinstance(temporal_scope_note, str) and temporal_scope_note.strip():
            if not lines:
                lines.append("## Dimension Guidance")
            lines.append("### Temporal Scope")
            lines.append(f"- {_collapse_ws(temporal_scope_note)}")

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


def _evidence_id(i: int) -> str:
    return f"E{i}"


def _escape_attr(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def _render_evidence(retrieved_chunks: list[dict]) -> str:
    """Render chunks as delimited evidence blocks the model cites by ID.

    The delimiters mark where filing text begins and ends; the system prompt
    says nothing inside them can act as instructions.  A closing tag inside a
    chunk is defused so a chunk cannot end its own block early.
    """
    blocks = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        text = str(chunk.get("text", "")).replace("</evidence", "<\\/evidence")
        blocks.append(
            f'<evidence id="{_evidence_id(i)}" section="{_escape_attr(_chunk_section_label(chunk))}" '
            f'page="{_escape_attr(chunk.get("source_page", ""))}" chunk="{_escape_attr(chunk.get("id", ""))}">\n'
            f"{text}\n"
            f"</evidence>"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


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

    chunks_text = _render_evidence(retrieved_chunks)

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
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
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


# ---------------------------------------------------------------------------
# Quote verification: every passage the model quotes must be in the evidence
# ---------------------------------------------------------------------------

_QUOTE_TRANSLATION = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
    " ": " ",
})
_ELLIPSIS = re.compile(r"\s*(?:\[(?:\.\.\.|\u2026)\]|\.\.\.|\u2026)\s*")
# Fragments shorter than this between ellipses carry too little to verify.
_MIN_FRAGMENT_CHARS = 12


def _normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).translate(_QUOTE_TRANSLATION)
    return " ".join(text.split()).casefold()


def _quote_in(quote: str, haystack_normalized: str) -> bool:
    """True when the quote appears in the text, ignoring whitespace, case and
    typographic quote/dash variants.  An elided quote ("A ... B") must have
    each fragment present, in order."""
    fragments = [
        _normalize_for_match(f).strip(" \"'")
        for f in _ELLIPSIS.split(str(quote))
    ]
    fragments = [f for f in fragments if f]
    if not fragments:
        return False
    position = 0
    checked_any = False
    for fragment in fragments:
        if len(fragment) < _MIN_FRAGMENT_CHARS and len(fragments) > 1:
            continue
        found = haystack_normalized.find(fragment, position)
        if found < 0:
            return False
        position = found + len(fragment)
        checked_any = True
    return checked_any


def _parse_evidence_id(value: Any, n_chunks: int) -> int | None:
    """'E3', 'e3', '3' or 'Evidence 3' -> index 2; None if absent or out of range."""
    match = re.search(r"(\d+)", str(value or ""))
    if not match:
        return None
    index = int(match.group(1)) - 1
    return index if 0 <= index < n_chunks else None


def _other_quotes(details: dict) -> list[tuple[str, str]]:
    """Quoted passages outside `evidence`: the dimension's *_quote fields and
    `source_passage` inside nested lists (D3 matters, D5 material weaknesses)."""
    quotes: list[tuple[str, str]] = []
    for key, value in details.items():
        if key == "evidence":
            continue
        if key.endswith("_quote") and isinstance(value, str) and value.strip():
            quotes.append((key, value))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    passage = item.get("source_passage")
                    if isinstance(passage, str) and passage.strip():
                        quotes.append((f"{key}[{i}].source_passage", passage))
    return quotes


def _check_quotes(details: dict, retrieved_chunks: list[dict]) -> tuple[list[dict], dict]:
    """Build evidence citations and check every quote against the evidence.

    An evidence item's quote must be in the block its evidence_id names; any
    other quote must be in some retrieved block.  Returns the citations and a
    summary {checked, verified, failed}.
    """
    normalized = [_normalize_for_match(c.get("text", "")) for c in retrieved_chunks]
    corpus = "\n".join(normalized)
    citations: list[dict] = []
    failed: list[str] = []
    checked = 0

    for n, item in enumerate(details.get("evidence") or []):
        if not isinstance(item, dict):
            continue
        quote = str(item.get("source_passage") or item.get("text") or "")
        index = _parse_evidence_id(item.get("evidence_id"), len(retrieved_chunks))
        citation: dict = {"text": quote, "source_page": item.get("source_page", 0)}
        if index is not None:
            chunk = retrieved_chunks[index]
            citation["evidence_id"] = _evidence_id(index + 1)
            citation["chunk_id"] = str(chunk.get("id", ""))
            citation["source_page"] = chunk.get("source_page", citation["source_page"])
        if quote.strip():
            checked += 1
            ok = index is not None and _quote_in(quote, normalized[index])
            citation["verified"] = ok
            if not ok:
                failed.append(f"evidence[{n}]")
        citations.append(citation)

    for label, quote in _other_quotes(details):
        checked += 1
        if not _quote_in(quote, corpus):
            failed.append(label)

    return citations, {"checked": checked, "verified": checked - len(failed), "failed": failed}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

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
    dimension_id = dimension["dimension_id"]

    _log.info(f"Analysis Agent: evaluating {dimension['name']} for {company_name}")

    reply: ModelReply | None = None
    # True only when the model returned an answer we could read; only then is
    # its rating checked and its quotes verified.
    model_answered = False

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
            "dimension": dimension_id,
            "verdict": raw_rating,
            "reasoning": reasoning,
            "evidence": [],
            "flags": ["no_evidence"],
        }
    else:
        try:
            reply = _call_model(
                system_prompt,
                user_prompt,
                dimension.get("output_schema"),
                dimension_id,
            )
        except Exception as e:
            _log.warning("Analysis model call failed for %s: %s", dimension_id, e)
            parsed = {
                "dimension": dimension_id,
                "verdict": "MEDIUM",
                "reasoning": f"Analysis model call failed: {e}",
                "evidence": [],
                "flags": ["analysis_error"],
            }
        else:
            if reply.stop_reason == "refusal":
                parsed = {
                    "dimension": dimension_id,
                    "verdict": "MEDIUM",
                    "reasoning": f"The model declined the request ({reply.stop_detail or 'refusal'}); no finding.",
                    "evidence": [],
                    "flags": ["refusal"],
                }
            elif reply.stop_reason == "max_tokens":
                parsed = {
                    "dimension": dimension_id,
                    "verdict": "MEDIUM",
                    "reasoning": "The answer was cut off at max_tokens; no finding.",
                    "evidence": [],
                    "flags": ["truncated"],
                }
            else:
                parsed = _parse_verdict(reply.text, dimension_id)
                model_answered = "parse_error" not in (parsed.get("flags") or [])

    raw_rating = str(parsed.get("verdict", parsed.get("rating", "")) or "")
    flags = [str(f) for f in (parsed.get("flags") or [])]
    if model_answered:
        rating_flag = _rating_flag(raw_rating)
        if rating_flag and rating_flag not in flags:
            flags.append(rating_flag)
    if reply is not None and reply.fallback_used and "model_fallback" not in flags:
        flags.append("model_fallback")

    citations, quote_check = _check_quotes(parsed, retrieved_chunks) if model_answered else ([], None)
    if quote_check and quote_check["failed"] and "unverified_quote" not in flags:
        flags.append("unverified_quote")

    verdict: VerdictCard = {
        "dimension": dimension_id,
        "rating": _coerce_rating(raw_rating),
        "confidence": retrieval_score,
        "reasoning": parsed.get("reasoning", ""),
        "evidence_citations": citations,
        "flags": flags,
        "retrieval_attempts": retrieval_attempts,
        "raw_verdict": raw_rating,
        "details": parsed,
        "retrieved_chunk_ids": [str(c.get("id", "")) for c in retrieved_chunks],
    }
    if reply is not None:
        verdict["served_model"] = reply.model
        verdict["usage"] = reply.usage
    if quote_check is not None:
        verdict["quote_check"] = quote_check

    _log.info(
        f"  Verdict: {verdict['rating']} (raw={raw_rating!r}, "
        f"confidence={verdict['confidence']:.3f}, "
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
