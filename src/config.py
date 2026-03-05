"""
Configuration module for the OSFI Compliance Analyzer.

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
OSFI_GUIDELINES_DIR = DATA_DIR / "osfi_guidelines"
BANK_REPORTS_DIR = DATA_DIR / "bank_reports"
PARSED_DIR = DATA_DIR / "parsed"
CHECKLIST_DIR = DATA_DIR / "checklist"

# ---------------------------------------------------------------------------
# LLM Configuration (Google Gemini - free tier)
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# ---------------------------------------------------------------------------
# Embedding Configuration (sentence-transformers - local, free)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Dimension mapping for common sentence-transformer models
EMBEDDING_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "all-MiniLM-L12-v2": 384,
}

# ---------------------------------------------------------------------------
# Pinecone Configuration (free tier)
# ---------------------------------------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "osfi-guidelines")

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
MAX_RETRIEVAL_RETRIES = 1

# ---------------------------------------------------------------------------
# Chunking settings
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ---------------------------------------------------------------------------
# Verdict labels
# ---------------------------------------------------------------------------
VERDICT_PASS = "PASS"
VERDICT_PARTIAL = "PARTIAL"
VERDICT_FAIL = "FAIL"
VALID_VERDICTS = {VERDICT_PASS, VERDICT_PARTIAL, VERDICT_FAIL}

# ---------------------------------------------------------------------------
# Risk severity levels
# ---------------------------------------------------------------------------
RISK_LOW = "LOW"
RISK_MEDIUM = "MEDIUM"
RISK_HIGH = "HIGH"
RISK_CRITICAL = "CRITICAL"


def validate_config() -> list[str]:
    """Check that required configuration values are set. Returns list of errors."""
    errors = []
    if not GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY is not set. Get one at https://aistudio.google.com/apikey")
    if not PINECONE_API_KEY:
        errors.append("PINECONE_API_KEY is not set. Sign up at https://www.pinecone.io/")
    return errors
