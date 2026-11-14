"""Offline checks for the indexing pipeline's bookkeeping.

Parsing, chunking and Pinecone are replaced with fakes.  What is checked is
the contract the rest of the project relies on:

  - the processed-chunk file holds exactly the chunks that were upserted
  - the namespace is cleared before the upsert, so no stale vectors survive
  - a namespace whose vector count differs from the chunk count is flagged
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src import config
from src.indexing import run_pipeline


def _chunks(n: int) -> list[dict]:
    return [
        {
            "text": f"chunk text number {i} " * 20,
            "headings": [f"Heading {i}"],
            "source_page": i + 1,
            "source_document": "fake-20991231",
            "label": "text",
        }
        for i in range(n)
    ]


@pytest.fixture
def fake_pipeline(tmp_path, monkeypatch):
    calls: list[tuple] = []
    state = {"index_count": None}

    monkeypatch.setattr(config, "PARSED_DIR", tmp_path)
    monkeypatch.setattr(run_pipeline, "parse_pdf", lambda path, write_outputs=False: object())
    monkeypatch.setattr(run_pipeline, "chunk_document", lambda doc, source_document: _chunks(5))
    monkeypatch.setattr(run_pipeline, "post_process", lambda chunks, min_chars: chunks)

    def fake_clear(namespace):
        calls.append(("clear", namespace, (tmp_path / "fake-20991231_processed.json").exists()))

    def fake_upsert(chunks, namespace=""):
        calls.append(("upsert", namespace, [c["text"] for c in chunks]))
        return len(chunks)

    def fake_wait(namespace, expected, timeout_seconds=90.0):
        return expected if state["index_count"] is None else state["index_count"]

    monkeypatch.setattr(run_pipeline, "clear_namespace", fake_clear)
    monkeypatch.setattr(run_pipeline, "upsert_chunks", fake_upsert)
    monkeypatch.setattr(run_pipeline, "wait_for_vector_count", fake_wait)

    pdf = tmp_path / "fake-20991231.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    return pdf, calls, state


def test_processed_file_matches_upserted_chunks(fake_pipeline) -> None:
    pdf, calls, _ = fake_pipeline
    rec = run_pipeline.run_for_filing(pdf)

    assert rec["status"] == "success", rec
    persisted = json.loads(Path(rec["processed_path"]).read_text(encoding="utf-8"))
    upserted_texts = next(c[2] for c in calls if c[0] == "upsert")
    assert [c["text"] for c in persisted] == upserted_texts
    assert rec["index_vector_count"] == 5


def test_namespace_cleared_after_persist_and_before_upsert(fake_pipeline) -> None:
    pdf, calls, _ = fake_pipeline
    run_pipeline.run_for_filing(pdf)

    order = [c[0] for c in calls]
    assert order == ["clear", "upsert"], order
    clear_call = calls[0]
    assert clear_call[1] == "fake-20991231"
    assert clear_call[2] is True, "chunk file must be written before the index is touched"


def test_index_count_mismatch_is_flagged(fake_pipeline) -> None:
    pdf, _, state = fake_pipeline
    state["index_count"] = 7  # e.g. stale vectors the clear did not remove
    rec = run_pipeline.run_for_filing(pdf)

    assert rec["status"] == "mismatch"
    assert rec["index_vector_count"] == 7
