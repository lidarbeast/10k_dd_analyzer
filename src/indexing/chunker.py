import json
import logging
import sys
from pathlib import Path

# Add project root to python path so we can import from src regardless of where we run the script
sys.path.append(str(Path(__file__).parent.parent.parent))

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc.document import DoclingDocument
from docling_core.transforms.chunker import BaseChunk
from transformers import AutoTokenizer

from src import config
from src.state import FilingChunk

_log = logging.getLogger(__name__)


def _build_chunker(max_tokens: int) -> HybridChunker:
    """Build a chunker aligned to the active embedding tokenizer when local."""
    if config.EMBEDDING_PROVIDER == "local":
        try:
            hf_tokenizer = AutoTokenizer.from_pretrained(config.LOCAL_EMBEDDING_MODEL)
            tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer, max_tokens=max_tokens)
            _log.info(
                "HybridChunker tokenizer aligned to local embedding model '%s' "
                "(tokenizer max=%s, chunk max=%s)",
                config.LOCAL_EMBEDDING_MODEL,
                getattr(hf_tokenizer, "model_max_length", "unknown"),
                max_tokens,
            )
            return HybridChunker(tokenizer=tokenizer)
        except Exception as e:
            _log.warning(
                "Failed to load tokenizer for local model '%s': %s. "
                "Falling back to HybridChunker default tokenizer.",
                config.LOCAL_EMBEDDING_MODEL,
                e,
            )

    return HybridChunker(max_tokens=max_tokens)

# ── Chunking logic ──────────────────────────────────────────

def chunk_document(
    parsed_doc_dict: dict,
    source_document: str | None = None,
    max_tokens: int = config.CHUNK_SIZE,
) -> list[FilingChunk]:
    """
    Chunk a Docling parsed document into FilingChunk entries using HybridChunker.

    Args:
        parsed_doc_dict: The native Docling JSON output dictionary.
        source_document: The source document identifier (e.g. 'shop-20241231').
        max_tokens: Maximum tokens per chunk.

    Returns:
        list of FilingChunk dicts ready for embedding + upserting.
    """
    try:
        # Load the DoclingDocument directly from the JSON dictionary
        doc = DoclingDocument.model_validate(parsed_doc_dict)
    except Exception as e:
        msg = (
            f"Failed to load DoclingDocument for '{source_document or 'unknown'}' "
            f"from parsed dictionary: {e}"
        )
        _log.error(msg)
        raise RuntimeError(msg) from e

    # Max tokens controls chunk size and retrieval granularity.
    # When using local embeddings, align chunk tokenizer with embedding tokenizer.
    chunker = _build_chunker(max_tokens=max_tokens)
    
    chunk_iter = chunker.chunk(dl_doc=doc)
    doc_chunks: list[BaseChunk] = list(chunk_iter)

    if not doc_chunks:
        msg = f"Native chunker produced 0 chunks for '{source_document or 'unknown'}'"
        _log.error(msg)
        raise RuntimeError(msg)

    chunks: list[FilingChunk] = []

    for chunk in doc_chunks:
        # Get contextualized text (includes heading context)
        text = chunker.contextualize(chunk)
        
        # Determine the source page natively from chunk meta
        first_page = 0
        label = "unknown"
        
        if chunk.meta.doc_items and len(chunk.meta.doc_items) > 0:
            item = chunk.meta.doc_items[0]
            if hasattr(item, "prov") and item.prov and len(item.prov) > 0:
                first_page = item.prov[0].page_no
            if hasattr(item, "label"):
                label = item.label

        
        # Keep all headings natively returned by Docling
        headings = list(chunk.meta.headings) if chunk.meta.headings else []

        chunks.append(
            FilingChunk(
                text=text,
                headings=headings,
                source_page=first_page,
                source_document=source_document or "unknown",
                label=label,
            )
        )

    _log.info(f"{source_document}: produced {len(chunks)} chunks using HybridChunker")
    return chunks

# ── CLI entry-point ──────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    parser = argparse.ArgumentParser(description="Chunk parsed filing JSONs.")
    parser.add_argument(
        "input_file", 
        nargs="?", 
        help="Path to a specific *_parsed.json file to chunk. If not provided, chunks all *_parsed.json files in data/parsed/."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=config.CHUNK_SIZE,
        help=f"Maximum tokens per chunk (default: {config.CHUNK_SIZE}).",
    )
    args = parser.parse_args()

    parsed_dir = config.PARSED_DIR
    
    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.exists():
            _log.error(f"Input file not found: {input_path}")
            exit(1)
        json_files = [input_path]
    else:
        json_files = sorted(parsed_dir.glob("*_parsed.json"))

    if not json_files:
        _log.error("No parsed JSON files found.")
    else:
        for json_path in json_files:
            _log.info(f"Chunking {json_path.name}...")
            source_doc = json_path.stem
            if source_doc.endswith("_parsed"):
                source_doc = source_doc[: -len("_parsed")]
            
            with open(json_path, "r", encoding="utf-8") as f:
                parsed_doc_dict = json.load(f)

            if parsed_doc_dict.get("source_document"):
                source_doc = parsed_doc_dict["source_document"]

            doc_chunks = chunk_document(
                parsed_doc_dict,
                source_document=source_doc,
                max_tokens=args.max_tokens,
            )
            _log.info(f"  ✓ {source_doc}: {len(doc_chunks)} chunks")

            output_path = json_path.parent / f"{source_doc}_chunks.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(doc_chunks, f, indent=2, ensure_ascii=False)
            _log.info(f"  → saved {len(doc_chunks)} chunks to {output_path.name}")
