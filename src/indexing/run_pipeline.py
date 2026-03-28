"""
End-to-end indexing pipeline runner.

Runs, per filing:
  parser.py (Docling) -> chunker.py (HybridChunker) -> post_process.py -> pinecone_ops.py

Runs fully in-memory and only writes per-run JSONL status logs.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

# ── Path bootstrap (allows running from any directory) ──────────────────────
sys.path.append(str(Path(__file__).parent.parent.parent))

from src import config  # noqa: E402
from src.indexing.chunker import chunk_document  # noqa: E402
from src.indexing.parser import parse_pdf  # noqa: E402
from src.indexing.pinecone_ops import upsert_chunks  # noqa: E402
from src.indexing.post_process import post_process  # noqa: E402

_log = logging.getLogger(__name__)

_REQUIRED_CHUNK_KEYS = {"text", "headings", "source_page", "source_document", "label"}


def _new_run_record(source_document: str, pdf_path: Path) -> dict:
    """Create a default per-filing run record."""
    return {
        "source_document": source_document,
        "pdf_path": str(pdf_path),
        "status": "unknown",
        "upserted": 0,
        "processed_chunks": 0,
        "failed_step": None,
        "error": None,
        "error_type": None,
        "traceback": None,
    }


def _apply_status_from_counts(
    rec: dict,
    *,
    source_document: str,
    upserted: int,
    processed_chunks: int,
    log_prefix: str = "",
) -> None:
    """Set status/upsert counters and emit consistent status logs."""
    rec["upserted"] = upserted
    rec["processed_chunks"] = processed_chunks

    if upserted < processed_chunks:
        rec["status"] = "partial"
        _log.warning(
            "%s%s: partial upsert (%s/%s)",
            log_prefix,
            source_document,
            upserted,
            processed_chunks,
        )
    elif upserted == processed_chunks:
        rec["status"] = "success"
        _log.info("%s%s: upserted %s chunks", log_prefix, source_document, upserted)
    else:
        rec["status"] = "mismatch"
        _log.warning(
            "%s%s: upserted more vectors than processed chunks (%s/%s)",
            log_prefix,
            source_document,
            upserted,
            processed_chunks,
        )


def _create_run_log_file() -> Path:
    """Create a per-run JSONL log file under logs/ and return its path."""
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Include microseconds and open with exclusive create to avoid collisions.
    for _ in range(3):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        out_path = logs_dir / f"indexing_run-{ts}.jsonl"
        try:
            with out_path.open("x", encoding="utf-8"):
                pass
            _log.info("Run log: %s", out_path)
            return out_path
        except FileExistsError:
            continue

    raise RuntimeError("Failed to create unique run log file after 3 attempts.")


def _append_run_log_record(log_path: Path, record: dict) -> None:
    """Append a single JSON record (one line) to the run log."""
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _validate_chunks(chunks: list[dict], source_document: str, stage: str) -> None:
    """Validate minimal chunk schema before downstream processing/upsert."""
    for idx, chunk in enumerate(chunks):
        missing = _REQUIRED_CHUNK_KEYS.difference(chunk.keys())
        if missing:
            raise ValueError(
                f"Invalid chunk schema for '{source_document}' at {stage} "
                f"(index={idx}): missing keys {sorted(missing)}"
            )


def run_for_filing(
    pdf_path: Path,
    *,
    namespace: str | None = None,
    min_chars: int = config.SHORT_CHUNK_MIN_CHARS,
) -> dict:
    """
    Run the full pipeline for a single PDF and return its run record.
    """
    source_document = pdf_path.stem
    namespace = namespace or source_document
    rec: dict = _new_run_record(source_document, pdf_path)

    processed_count = 0
    upserted = 0
    step_name = "parse"
    try:
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        _log.info("Parsing %s", pdf_path.name)
        parsed_doc = parse_pdf(pdf_path, write_outputs=False)

        step_name = "chunk"
        _log.info("Chunking %s", pdf_path.name)
        raw_chunks = chunk_document(parsed_doc, source_document=source_document)
        _validate_chunks(raw_chunks, source_document, stage="chunk")

        step_name = "post_process"
        _log.info("Post-processing %s", pdf_path.name)
        processed = post_process(raw_chunks, min_chars=min_chars)
        processed_count = len(processed)

        step_name = "upsert"
        _log.info(
            "Upserting %s (%s chunks) to namespace '%s'",
            pdf_path.name,
            processed_count,
            namespace,
        )
        upserted = upsert_chunks(processed, namespace=namespace)

        if upserted != processed_count:
            _log.warning(
                "Upsert count mismatch for %s: expected %s, got %s.",
                source_document,
                processed_count,
                upserted,
            )

        _apply_status_from_counts(
            rec,
            source_document=source_document,
            upserted=upserted,
            processed_chunks=processed_count,
        )
        return rec
    except Exception as e:
        rec["status"] = "failed"
        rec["upserted"] = upserted
        rec["processed_chunks"] = processed_count
        rec["failed_step"] = step_name
        rec["error"] = str(e)
        rec["error_type"] = type(e).__name__
        rec["traceback"] = traceback.format_exc()
        _log.error(
            "Pipeline failed for %s at step '%s': %s",
            source_document,
            step_name,
            e,
            exc_info=True,
        )
        return rec


def run_all_filings(
    filings_dir: Path,
    *,
    min_chars: int = config.SHORT_CHUNK_MIN_CHARS,
    run_log_path: Path | None = None,
) -> dict[str, int]:
    """Run pipeline for every PDF in filings_dir. Returns per-filing upsert counts."""
    pdfs = sorted(filings_dir.glob("*.pdf"))
    if not pdfs:
        _log.error("No PDF files found in %s", filings_dir)
        return {}

    results: dict[str, int] = {}
    if run_log_path is None:
        run_log_path = _create_run_log_file()
    for pdf in pdfs:
        source_document = pdf.stem
        rec = run_for_filing(
            pdf,
            namespace=source_document,
            min_chars=min_chars,
        )

        # Persist this filing's record before moving to the next file.
        _append_run_log_record(run_log_path, rec)

        if rec["status"] == "failed":
            results[source_document] = 0
        else:
            results[source_document] = int(rec["upserted"])

    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Run parse->chunk->post_process->upsert for filings in data/filings/."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help="Optional path to a single filing PDF. If omitted, processes all PDFs in data/filings/.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=config.SHORT_CHUNK_MIN_CHARS,
        help=(
            "Short-chunk merge threshold "
            f"(default: {config.SHORT_CHUNK_MIN_CHARS})."
        ),
    )
    args = parser.parse_args()

    errors = config.validate_config()
    if errors:
        print("Config validation failed:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    if args.input_file:
        pdf_path = Path(args.input_file)
        run_log_path = _create_run_log_file()
        rec = run_for_filing(
            pdf_path,
            min_chars=args.min_chars,
        )
        _append_run_log_record(run_log_path, rec)

        if rec["status"] == "failed":
            raise SystemExit(1)
    else:
        run_log_path = _create_run_log_file()
        results = run_all_filings(
            config.FILINGS_DIR,
            min_chars=args.min_chars,
            run_log_path=run_log_path,
        )
        total = sum(results.values())
        _log.info("Done: %s filings, %s total chunks upserted", len(results), total)
