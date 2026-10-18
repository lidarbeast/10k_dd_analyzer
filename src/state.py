"""
Shared state definition for the LangGraph M&A due diligence pipeline.

DDState is the TypedDict that flows through all graph nodes.
"""

from typing import NotRequired, TypedDict


class FilingChunk(TypedDict):
    """A chunk of a parsed SEC 10-K filing with section metadata."""
    text: str
    headings: list[str]
    source_page: int
    source_document: str
    label: str


class DimensionSpec(TypedDict):
    """Specification for a single due diligence dimension, loaded from config JSON."""
    dimension_id: str
    name: str
    retrieval_queries: list[str]
    alternate_queries: list[str]
    rubric: dict               # {"HIGH": {...}, "MEDIUM": {...}, "LOW": {...}}
    prompt_template: dict      # {"system": str, "user_template": str}
    target_sections: list[str]
    mandatory_10k_items: NotRequired[list[dict]]
    evidence_schema: NotRequired[dict]
    no_evidence_handling: NotRequired[str]
    no_evidence_rating: NotRequired[str]   # explicit fallback when no evidence is retrieved
    rubric_sources: NotRequired[list[str]]
    confidence_floor: NotRequired[float]
    output_schema: NotRequired[dict]       # JSON schema for the model's answer (output_config.format)


class EvidenceCitation(TypedDict):
    """A single piece of evidence backing a verdict."""
    text: str
    source_page: int
    evidence_id: NotRequired[str]   # "E3": the prompt's evidence block the quote came from
    chunk_id: NotRequired[str]      # Pinecone ID of that block, e.g. "rev-20221231_chunk_280"
    verified: NotRequired[bool]     # quote found verbatim (whitespace-normalised) in that chunk


class VerdictCard(TypedDict):
    """Structured verdict for a single due diligence dimension."""
    dimension: str             # e.g. "financial_health"
    rating: str                # LOW | MEDIUM | HIGH (INCONCLUSIVE coerces to MEDIUM; see raw_verdict)
    confidence: float          # 0.0–1.0: mean similarity score of the top 3 retrieved chunks
    reasoning: str             # One-paragraph synthesis
    evidence_citations: list[EvidenceCitation]
    flags: list[str]           # Model flags plus pipeline flags (degraded and diagnostic)
    retrieval_attempts: int    # 1 if first retrieval sufficient, 2 if retry triggered
    raw_verdict: NotRequired[str]                # the model's answer before coercion, e.g. "INCONCLUSIVE"
    details: NotRequired[dict]                   # the model's full structured answer
    retrieved_chunk_ids: NotRequired[list[str]]  # in prompt order: E1 is the first
    served_model: NotRequired[str]               # the model that actually answered
    usage: NotRequired[dict]                     # token counts for the call
    quote_check: NotRequired[dict]               # {checked, verified, failed}: quotes found in the evidence


class DDState(TypedDict):
    """Shared state flowing through the LangGraph due diligence pipeline."""
    # Input
    company_name: str
    filing_path: str

    # Document Agent output.  `filing_chunks` is not populated: chunks live in
    # Pinecone (namespace = filing stem) and in data/parsed/{stem}_processed.json.
    filing_chunks: list[FilingChunk]
    namespace: NotRequired[str]
    indexed_chunk_count: NotRequired[int]

    # Dimension loop state
    dimensions: list[DimensionSpec]
    current_dim_index: int

    # Analysis output (accumulated across dimensions)
    verdicts: list[VerdictCard]
    retrieval_attempts: list[int]
    confidence_scores: list[float]

    # Report output
    report: dict  # Final aggregated DD report

    # Scratch fields used during dimension loop (optional)
    _retrieved_chunks: NotRequired[list[dict]]
    _retrieval_score: NotRequired[float]
    _retrieval_attempts: NotRequired[int]
    _retrieval_audit: NotRequired[dict]          # current dimension only
    _retrieval_audits: NotRequired[dict]         # dimension_id -> audit, accumulated across the loop
    _warnings: NotRequired[list[str]]
