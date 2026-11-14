"""
Configuration module for the M&A Due Diligence Analyzer.

Loads environment variables and defines project-wide constants.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FILINGS_DIR = DATA_DIR / "filings"
# Rubrics are versioned by directory.  Once an experiment has run against a
# version, that directory is frozen; a rubric change means a new directory
# (and a new MLflow experiment), never an edit in place.
RUBRIC_VERSION = os.getenv("RUBRIC_VERSION", "v1")
DIMENSIONS_DIR = DATA_DIR / "dimensions" / RUBRIC_VERSION
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
PARSED_DIR = DATA_DIR / "parsed"
REPORTS_DIR = DATA_DIR / "reports"
# Every pipeline run: data/runs/<run>/state.json plus the committed ledger.jsonl
RUNS_DIR = DATA_DIR / "runs"

# ---------------------------------------------------------------------------
# LLM Configuration (Analysis Agent)
# ---------------------------------------------------------------------------
# The provider follows the model name: "gemini-*" calls the Google Gemini API,
# "claude-*" the Anthropic API.
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# Gemini only: Claude Sonnet 5.5 rejects a non-default temperature.  Even at 0
# repeated runs are not guaranteed to agree.
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
# Claude only: thinking depth and token spend (low | medium | high | xhigh | max).
LLM_EFFORT = os.getenv("LLM_EFFORT", "high")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "16000"))

# Retry/timeout for analysis calls.  An unguarded call turns a transient 429
# into a verdict that reads like a real one.  The timeout covers thinking at
# high effort over a full evidence block.
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "300"))
# The backoff is long because the Gemini free tier allows 5 requests a minute
# and returns 503 under load: quick retries only use up that allowance.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "4"))
LLM_RETRY_BASE_SECONDS = float(os.getenv("LLM_RETRY_BASE_SECONDS", "15.0"))

# ---------------------------------------------------------------------------
# Embedding Configuration
# ---------------------------------------------------------------------------
# Defaults are the setup the index was built with.  Query embeddings must come
# from the same model as the indexed chunks, or retrieval silently degrades.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()

# Google / Gemini embeddings (unused while the provider is local)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

# Local embeddings (sentence-transformers)
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-m3")

# Pinecone index dimension must match embedding output dimension.
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

# ---------------------------------------------------------------------------
# Pinecone Configuration (serverless free tier)
# ---------------------------------------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ma-dd-analyzer")

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "8"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
# Minimum score any retrieved chunk must meet (after retry) before passing to
# the LLM.  Below this the retrieval agent signals no-evidence so the Analysis
# Agent applies the dimension's no_evidence_handling instead of hallucinating.
RETRIEVAL_HARD_FLOOR = float(os.getenv("RETRIEVAL_HARD_FLOOR", "0.35"))
MAX_RETRIEVAL_RETRIES = 1
SECTION_BOOST_WEIGHT = float(os.getenv("SECTION_BOOST_WEIGHT", "0.2"))

# ---------------------------------------------------------------------------
# Chunking settings
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "768"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
SHORT_CHUNK_MIN_CHARS = int(os.getenv("SHORT_CHUNK_MIN_CHARS", "150"))

# ---------------------------------------------------------------------------
# Embedding retry settings
# ---------------------------------------------------------------------------
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))
EMBEDDING_RETRY_BASE_SECONDS = float(os.getenv("EMBEDDING_RETRY_BASE_SECONDS", "1.0"))

# ---------------------------------------------------------------------------
# Verdict labels (risk ratings per dimension)
# ---------------------------------------------------------------------------
RATING_LOW = "LOW"
RATING_MEDIUM = "MEDIUM"
RATING_HIGH = "HIGH"
VALID_RATINGS = {RATING_LOW, RATING_MEDIUM, RATING_HIGH}

# ---------------------------------------------------------------------------
# LangSmith tracing
# ---------------------------------------------------------------------------
# Tracing sends prompts — including filing text — to smith.langchain.com, so
# it runs only with a real key.  An empty or placeholder key ("your-...")
# turns it off here, before LangGraph reads the environment.
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "ma-dd-analyzer")
_TRACING_REQUESTED = os.getenv("LANGCHAIN_TRACING_V2", "false").strip().lower() == "true"
_TRACING_KEY_VALID = bool(LANGCHAIN_API_KEY) and not LANGCHAIN_API_KEY.startswith("your-")
LANGCHAIN_TRACING_V2 = "true" if (_TRACING_REQUESTED and _TRACING_KEY_VALID) else "false"
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGSMITH_TRACING"] = LANGCHAIN_TRACING_V2

# ---------------------------------------------------------------------------
# MLflow experiment tracking
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "dd-analyzer-eval")

# ---------------------------------------------------------------------------
# 10-K section types for metadata filtering
# ---------------------------------------------------------------------------
SECTION_TYPES = [
    "item_1",
    "item_1a",
    "item_1c",
    "item_3",
    "item_7",
    "item_8",
    "item_9a",
    "auditor_report",
    "footnotes",
]

# ---------------------------------------------------------------------------
# Test company CIKs (for EDGAR API calls)
# ---------------------------------------------------------------------------
COMPANY_CIKS = {
    "carvana": "0001690820",
    "peloton": "0001639825",
    "shopify": "0001594805",
}


def validate_config() -> list[str]:
    """Check that required configuration values are set. Returns list of errors."""
    errors = []
    if EMBEDDING_PROVIDER not in {"google", "local"}:
        errors.append("EMBEDDING_PROVIDER must be 'google' or 'local'")
    if EMBEDDING_PROVIDER == "google" and not GOOGLE_API_KEY:
        errors.append(
            "GOOGLE_API_KEY is not set. Get one at https://aistudio.google.com/apikey"
        )
    if not PINECONE_API_KEY:
        errors.append("PINECONE_API_KEY is not set. Sign up at https://www.pinecone.io/")
    return errors


def llm_provider() -> str:
    """Which API serves LLM_MODEL: "anthropic", "google" or "" if neither."""
    if LLM_MODEL.startswith("claude-"):
        return "anthropic"
    if LLM_MODEL.startswith("gemini-"):
        return "google"
    return ""


def validate_llm_config() -> list[str]:
    """Check the Analysis Agent's settings.  Returns a list of errors."""
    errors = []
    provider = llm_provider()
    if not provider:
        errors.append(f"LLM_MODEL={LLM_MODEL!r} must be a gemini-* or claude-* model")
    if provider == "google" and not GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY is not set. Get one at https://aistudio.google.com/apikey")
    if provider == "anthropic" and LLM_EFFORT not in {"low", "medium", "high", "xhigh", "max"}:
        errors.append(f"LLM_EFFORT={LLM_EFFORT!r} must be low|medium|high|xhigh|max")
    return errors
