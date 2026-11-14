# 10k_dd_analyzer

Agentic RAG over SEC 10-K filings. It rates financial health, going concern and internal controls, and evaluates the ratings against ground truth from external sources.

## Development setup

You need Linux, Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra cu128        # NVIDIA GPU; use --extra cpu without one
source .venv/bin/activate
pre-commit install           # ruff, lock-file and large-file checks on each commit
cp .env.example .env         # then fill in GOOGLE_API_KEY and PINECONE_API_KEY

python -m pytest -q -m "not live"
```

Always pass the torch extra to `uv sync`. Without it, uv replaces the installed torch with PyPI's build.

CI runs lint, format, lock-file and offline-test checks on every pull request. See `.github/workflows/ci.yml`.
