"""
Run records: what every pipeline run saves, and the ledger of all runs.

Each run writes its full final state to data/runs/{date}_{stem}_{sha}/state.json
and appends one line to data/runs/ledger.jsonl.  The ledger is committed; the
run directories are not.

The ledger exists because filings that may later be scored are being run
during development (decision of 2026-10-09).  When the scored set is chosen,
the ledger shows exactly which filings were seen, under which rubric and
pipeline configuration, with which results.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from src import config

LEDGER_NAME = "ledger.jsonl"


def rubric_hash(dimensions_dir: Path | None = None) -> str:
    """Short SHA-256 over every file in the rubric directory (names and bytes)."""
    root = dimensions_dir or config.DIMENSIONS_DIR
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def git_state() -> dict:
    """Commit the run used, and whether the working tree had uncommitted changes."""
    sha = _git("rev-parse", "--short", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "git_sha": sha or "unknown",
        "git_dirty": bool(status) if status is not None else None,
    }


def run_provenance() -> dict:
    """The configuration a run used, stamped on its report."""
    return {
        "rubric_version": config.RUBRIC_VERSION,
        "rubric_hash": rubric_hash(),
        "model": config.LLM_MODEL,
        "effort": config.LLM_EFFORT if config.llm_provider() == "anthropic" else None,
        "temperature": config.LLM_TEMPERATURE if config.llm_provider() == "google" else None,
        "retrieval_top_k": config.RETRIEVAL_TOP_K,
        "embedding_model": (
            config.LOCAL_EMBEDDING_MODEL if config.EMBEDDING_PROVIDER == "local" else config.EMBEDDING_MODEL
        ),
        **git_state(),
    }


def write_run(state: dict, runs_dir: Path | None = None) -> Path:
    """Save the run's full state and append its ledger line.  Returns the run directory."""
    runs_dir = runs_dir or config.RUNS_DIR
    report = state.get("report") or {}
    now = datetime.now(UTC)
    stem = report.get("namespace") or Path(state.get("filing_path", "unknown")).stem
    sha = report.get("git_sha", "unknown")

    run_dir = runs_dir / f"{now.strftime('%Y%m%dT%H%M%SZ')}_{stem}_{sha}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "state.json").write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    entry = {
        "timestamp": now.isoformat(timespec="seconds"),
        "company_name": report.get("company_name"),
        "filing": stem,
        "dimensions": list((report.get("ratings") or {}).keys()),
        "ratings": report.get("ratings"),
        "raw_verdicts": report.get("raw_verdicts"),
        "degraded_dimensions": report.get("degraded_dimensions"),
        "trustworthy": report.get("trustworthy"),
        "rubric_version": report.get("rubric_version"),
        "rubric_hash": report.get("rubric_hash"),
        "model": report.get("model"),
        "effort": report.get("effort"),
        "temperature": report.get("temperature"),
        "git_sha": report.get("git_sha"),
        "git_dirty": report.get("git_dirty"),
        "run_dir": run_dir.relative_to(config.PROJECT_ROOT).as_posix()
        if run_dir.is_relative_to(config.PROJECT_ROOT) else str(run_dir),
    }
    with (runs_dir / LEDGER_NAME).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return run_dir
