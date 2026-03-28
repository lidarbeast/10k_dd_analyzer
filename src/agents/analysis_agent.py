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

from langchain_google_genai import ChatGoogleGenerativeAI

from src import config
from src.state import DDState, VerdictCard

_log = logging.getLogger(__name__)


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
        section = chunk.get("section_type", "unknown")
        score = chunk.get("score", 0.0)
        text = chunk.get("text", "")
        chunks_text += (
            f"### Evidence {i} [{section}] (score: {score:.3f})\n"
            f"{text}\n\n"
        )

    user_prompt = template.get("user_template", "Evaluate {company_name}.\n\n{retrieved_chunks}")
    user_prompt = user_prompt.replace("{company_name}", company_name)
    user_prompt = user_prompt.replace("{retrieved_chunks}", chunks_text)

    return system_prompt, user_prompt


def _parse_verdict(response_text: str, dimension_id: str) -> dict:
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
            "verdict": "INCONCLUSIVE",
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
        retrieval_score: Max similarity score from retrieval.
        retrieval_attempts: Number of retrieval passes made.

    Returns:
        VerdictCard with structured verdict.
    """
    system_prompt, user_prompt = _build_prompt(dimension, company_name, retrieved_chunks)

    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        google_api_key=config.GOOGLE_API_KEY,
    )

    _log.info(f"Analysis Agent: evaluating {dimension['name']} for {company_name}")

    # Call LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "human", "content": user_prompt},
    ]
    response = llm.invoke(messages)
    response_text = response.content

    # Parse structured verdict
    parsed = _parse_verdict(response_text, dimension["dimension_id"])

    # Build VerdictCard
    verdict: VerdictCard = {
        "dimension": dimension["dimension_id"],
        "rating": parsed.get("verdict", parsed.get("rating", "INCONCLUSIVE")).upper(),
        "confidence": retrieval_score,
        "reasoning": parsed.get("reasoning", ""),
        "evidence_citations": [
            {
                "text": e.get("source_passage", e.get("text", "")),
                "source_page": e.get("source_page", 0),
            }
            for e in parsed.get("evidence", [])
        ],
        "flags": parsed.get("flags", []),
        "retrieval_attempts": retrieval_attempts,
    }

    # Validate rating
    if verdict["rating"] not in config.VALID_RATINGS:
        _log.warning(
            f"Invalid rating '{verdict['rating']}' for {dimension['name']}; "
            f"keeping as-is"
        )

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
    dimension = state["dimensions"][dim_index]
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
        f"Dimension {dim_index + 1}/5 complete: "
        f"{dimension['name']} → {verdict['rating']}"
    )
    return state
