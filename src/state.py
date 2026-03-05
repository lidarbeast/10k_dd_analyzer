"""
Shared state definition for the LangGraph compliance audit pipeline.

AuditState is the TypedDict that flows through all graph nodes.
"""

from typing import TypedDict


class ClauseResult(TypedDict):
    """Result of analyzing a single OSFI clause against a bank policy."""
    clause_id: str
    clause_text: str
    verdict: str            # PASS | PARTIAL | FAIL
    evidence: str           # Retrieved OSFI excerpt used for grading
    policy_excerpt: str     # Matching bank policy text
    reasoning: str          # LLM explanation of the verdict
    risk_severity: str      # LOW | MEDIUM | HIGH | CRITICAL
    retrieval_score: float  # Top similarity score from Pinecone
    retrieval_attempts: int # Number of retrieval calls made (1 or 2)


class PolicyChunk(TypedDict):
    """A chunk of bank policy mapped to an OSFI clause."""
    clause_id: str
    policy_chunk: str
    retrieval_query: str


class OSFIClause(TypedDict):
    """A single OSFI guideline clause from the audit checklist."""
    clause_id: str
    clause_text: str
    requirement_summary: str


class AuditState(TypedDict):
    """Shared state flowing through the LangGraph compliance pipeline."""
    # Input
    company_name: str
    policy_chunks: list[PolicyChunk]
    osfi_checklist: list[OSFIClause]

    # Processing state
    current_clause_index: int
    retrieval_attempts: int

    # Output
    graded_results: list[ClauseResult]
    report: dict  # Final aggregated compliance report
