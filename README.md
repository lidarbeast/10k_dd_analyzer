# OSFI Compliance Analyzer

An agentic AI system that audits Canadian bank risk management frameworks against OSFI regulatory guidelines, producing structured per-clause compliance reports.

## Architecture

**3-Agent LangGraph Pipeline:**

| Agent | Role |
|-------|------|
| **Document Agent** | Parses bank policy PDF, chunks into auditable claims, maps to OSFI clauses |
| **Retrieval Agent** | Hybrid vector + keyword search against indexed OSFI corpus in Pinecone |
| **Analysis Agent** | ReAct loop: retrieves evidence, evaluates alignment, outputs structured verdict |

## Tech Stack

- **LLM**: Google Gemini (free tier)
- **Orchestration**: LangGraph (StateGraph)
- **Vector DB**: Pinecone (hybrid search)
- **Embeddings**: sentence-transformers (local, free)
- **PDF Parsing**: Docling
- **Experiment Tracking**: MLflow
- **Backend**: FastAPI
- **Frontend**: Streamlit

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env with your API keys (Gemini + Pinecone)

# 4. Validate config
python -c "from src.config import validate_config; print(validate_config())"
```

## Project Structure

```
src/
├── config.py           # Settings and environment variables
├── state.py            # AuditState TypedDict (shared pipeline state)
├── agents/             # Document, Retrieval, Analysis agents
├── graph/              # LangGraph StateGraph pipeline
├── indexing/           # PDF parsing, chunking, Pinecone ops
└── evaluation/         # Metrics and eval harness
data/
├── osfi_guidelines/    # OSFI guideline PDFs
├── bank_reports/       # Bank annual report risk sections
├── parsed/             # Docling output
└── checklist/          # OSFI E-23 audit checklist JSON
```

## Regulatory Corpus

- **OSFI E-23** — Model Risk Management (primary)
- **OSFI E-21** — Operational Risk Management
- **OSFI B-10** — Third-Party Risk Management
