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
DIMENSIONS_DIR = DATA_DIR / "dimensions"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
PARSED_DIR = DATA_DIR / "parsed"
REPORTS_DIR = DATA_DIR / "reports"

# ---------------------------------------------------------------------------
# LLM Configuration (Google Gemini — free tier)
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# ---------------------------------------------------------------------------
# Embedding Configuration
# ---------------------------------------------------------------------------
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "google").lower()

# Google / Gemini embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

# Local embeddings (sentence-transformers)
# Good defaults:
# - sentence-transformers/all-MiniLM-L6-v2 (384 dim, fast, CPU-friendly)
# - intfloat/e5-large-v2 (1024 dim, higher quality, heavier)
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# Pinecone index dimension must match embedding output dimension.
# For local models, you typically want to set this to the model's dim.
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "3072"))

# ---------------------------------------------------------------------------
# Pinecone Configuration (serverless free tier)
# ---------------------------------------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ma-dd-analyzer")

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
MAX_RETRIEVAL_RETRIES = 1
SECTION_BOOST_WEIGHT = float(os.getenv("SECTION_BOOST_WEIGHT", "0.2"))

# ---------------------------------------------------------------------------
# Chunking settings
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
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
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "ma-dd-analyzer")

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
