"""The run record: every run's state is saved and appended to the ledger.

The ledger is how a later choice of scored filings can see which filings
were already run during development, and under which configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.run_record import rubric_hash, run_provenance, write_run


def test_write_run_saves_state_and_appends_ledger(tmp_path) -> None:
    report = {
        "company_name": "TestCo",
        "namespace": "test-20991231",
        "ratings": {"d5_internal_controls": "LOW"},
        "raw_verdicts": {"d5_internal_controls": "LOW"},
        "degraded_dimensions": {},
        "trustworthy": True,
        **run_provenance(),
    }
    state = {"filing_path": "data/filings/test-20991231.pdf", "report": report, "verdicts": []}

    first = write_run(state, runs_dir=tmp_path)
    write_run(state, runs_dir=tmp_path)

    saved = json.loads((first / "state.json").read_text(encoding="utf-8"))
    assert saved["report"]["namespace"] == "test-20991231"
    lines = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, "The ledger appends; it never rewrites earlier runs"
    entry = json.loads(lines[0])
    assert entry["filing"] == "test-20991231"
    assert entry["ratings"] == {"d5_internal_controls": "LOW"}
    assert entry["rubric_hash"] == rubric_hash()
    assert entry["model"] == config.LLM_MODEL


def test_rubric_hash_changes_with_any_rubric_file(tmp_path) -> None:
    (tmp_path / "a.json").write_text('{"x": 1}', encoding="utf-8")
    before = rubric_hash(tmp_path)
    (tmp_path / "a.json").write_text('{"x": 2}', encoding="utf-8")
    assert rubric_hash(tmp_path) != before
