"""
Shared state definition for the LangGraph M&A due diligence pipeline.

DDState is the TypedDict that flows through all graph nodes.
"""

from typing import TypedDict


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


class EvidenceCitation(TypedDict):
    """A single piece of evidence backing a verdict."""
    text: str
    source_page: int


class VerdictCard(TypedDict):
    """Structured verdict for a single due diligence dimension."""
    dimension: str             # e.g. "financial_health"
    rating: str                # LOW | MEDIUM | HIGH
    confidence: float          # 0.0–1.0: max similarity score of top retrieved chunk
    reasoning: str             # One-paragraph synthesis
    evidence_citations: list[EvidenceCitation]
    flags: list[str]           # Specific risk signals found
    retrieval_attempts: int    # 1 if first retrieval sufficient, 2 if retry triggered


class DDState(TypedDict):
    """Shared state flowing through the LangGraph due diligence pipeline."""
    # Input
    company_name: str
    filing_path: str

    # Document Agent output
    filing_chunks: list[FilingChunk]

    # Dimension loop state
    dimensions: list[DimensionSpec]
    current_dim_index: int

    # Analysis output (accumulated across dimensions)
    verdicts: list[VerdictCard]
    retrieval_attempts: list[int]
    confidence_scores: list[float]

    # Report output
    report: dict  # Final aggregated DD report
