"""
Post-processing pipeline for FilingChunks produced by chunker.py.

Run this after chunking and before embedding + upserting to Pinecone.
Produces cleaner chunk boundaries by merging short orphan chunks.

Pipeline stages (in order):
   1. Short-chunk merge  — absorb orphan chunks into adjacent same-heading chunk

Usage (CLI):
    # Process a single chunks file:
    python src/indexing/post_process.py data/parsed/shop-20241231_chunks.json

    # Process all *_chunks.json files in data/parsed/:
    python src/indexing/post_process.py
"""

import json
import logging
import sys
from pathlib import Path

# ── Path bootstrap (allows running from any directory) ──────────────────────
sys.path.append(str(Path(__file__).parent.parent.parent))

from src import config  # noqa: E402  (imported after path fix)

_log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────


# 1 token ≈ 4 characters (English prose approximation).
# Used as a cheap proxy to avoid running the full tokenizer per pair.
_MAX_TOKENS = 512
_CHARS_PER_TOKEN = 4
_MAX_CHARS = _MAX_TOKENS * _CHARS_PER_TOKEN  # ≈ 2048


# ── Stage 1: Short-chunk merge ───────────────────────────────────────────────

def merge_short_chunks(
    chunks: list[dict],
    min_chars: int = config.SHORT_CHUNK_MIN_CHARS,
) -> list[dict]:
    """
    Merge any chunk shorter than `min_chars` into the preceding chunk,
    only if they share the same headings AND the combined text stays
    within _MAX_CHARS. Short chunks with no valid predecessor are kept as-is.
    """
    before = len(chunks)
    result: list[dict] = []

    for chunk in chunks:
        if (
            result
            and len(chunk["text"]) < min_chars
            and chunk["headings"] == result[-1]["headings"]
            and len(result[-1]["text"]) + len(chunk["text"]) + 1 <= _MAX_CHARS
        ):
            result[-1]["text"] += "\n" + chunk["text"]
            _log.debug(
                f"Merged short chunk into previous "
                f"(headings={chunk['headings']}, page={chunk['source_page']})"
            )
        else:
            result.append(chunk)

    _log.debug(f"Short-chunk merge: {before} → {len(result)} chunks")
    return result


# ── Full pipeline ────────────────────────────────────────────────────────────

def post_process(
    chunks: list[dict],
    min_chars: int = config.SHORT_CHUNK_MIN_CHARS,
) -> list[dict]:
    """
    Run all post-processing stages on a list of raw FilingChunk dicts.

    Args:
        chunks:    Raw FilingChunk dicts from chunker.py.
        min_chars: Character threshold below which a chunk is considered short.

    Returns:
        List of FilingChunk dicts with cleaner chunk boundaries.
    """
    _log.info(f"post_process: starting with {len(chunks)} chunks")

    chunks = merge_short_chunks(chunks, min_chars=min_chars)

    if not chunks:
        msg = "post_process: produced 0 chunks after merging"
        _log.error(msg)
        raise RuntimeError(msg)

    _log.info(f"post_process: finished with {len(chunks)} chunks")
    return chunks


# ── CLI entry-point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Post-process FilingChunks JSON files before Pinecone upsert."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help=(
            "Path to a specific *_chunks.json file. "
            "If omitted, all *_chunks.json files in data/parsed/ are processed."
        ),
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=config.SHORT_CHUNK_MIN_CHARS,
        help=(
            "Character threshold below which a chunk is treated as short "
            f"(default: {config.SHORT_CHUNK_MIN_CHARS})."
        ),
    )
    args = parser.parse_args()

    parsed_dir = config.PARSED_DIR

    if args.input_file:
        chunk_files = [Path(args.input_file)]
    else:
        chunk_files = sorted(parsed_dir.glob("*_chunks.json"))

    if not chunk_files:
        _log.error("No *_chunks.json files found.")
        sys.exit(1)

    for chunk_path in chunk_files:
        _log.info(f"Processing {chunk_path.name} ...")

        with open(chunk_path, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)

        processed = post_process(raw_chunks, min_chars=args.min_chars)


        # Write output alongside the input file
        stem = chunk_path.stem.replace("_chunks", "")
        out_path = chunk_path.parent / f"{stem}_processed.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)

        _log.info(
            f"  ✓ {chunk_path.name}: {len(raw_chunks)} → {len(processed)} chunks "
            f"→ saved to {out_path.name}"
        )
