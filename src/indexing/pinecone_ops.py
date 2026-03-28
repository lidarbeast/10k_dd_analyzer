"""
Pinecone operations for the M&A Due Diligence Analyzer.

Uses Google text-embedding-004 (768-dim) for dense embeddings and
Pinecone serverless for vector storage with heading-based metadata.
"""

import json
import logging
import sys
import time

from pathlib import Path

# ── Path bootstrap (allows running from any directory) ──────────────────────
sys.path.append(str(Path(__file__).parent.parent.parent))

from google import genai                  # noqa: E402
from pinecone import Pinecone, ServerlessSpec  # noqa: E402

from src import config                    # noqa: E402

_log = logging.getLogger(__name__)

# ── Module-level singletons ──────────────────────────────────
_pinecone_client: Pinecone | None = None
_genai_client: genai.Client | None = None
_local_embedder = None


def _get_genai_client() -> genai.Client:
    """Return a lazily-initialised google.genai Client."""
    global _genai_client
    if _genai_client is None:
        if not config.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY is not set. "
                "Get one at https://aistudio.google.com/apikey"
            )
        _genai_client = genai.Client(api_key=config.GOOGLE_API_KEY)
    return _genai_client


# ── Embeddings ───────────────────────────────────────────────

def _get_local_embedder():
    """Return a lazily-initialised sentence-transformers embedder."""
    global _local_embedder
    if _local_embedder is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install dependencies (pip install -r requirements.txt) "
                "or set EMBEDDING_PROVIDER=google."
            ) from e
        _local_embedder = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
    return _local_embedder


def _embedding_dimension() -> int:
    """Resolve embedding dimension strictly from configuration."""
    dim = int(config.EMBEDDING_DIMENSION)
    if dim <= 0:
        raise ValueError("EMBEDDING_DIMENSION must be a positive integer")
    return dim


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate dense embeddings using configured provider.

    Providers:
      - google: Gemini embedding API (batched, rate-limited)
      - local: sentence-transformers (CPU/GPU local)

    Args:
        texts: List of text strings to embed.

    Returns:
        List of float vectors, one per input text.
    """
    if config.EMBEDDING_PROVIDER == "local":
        embedder = _get_local_embedder()
        vectors = embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    client = _get_genai_client()

    batch_size = 100
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=batch,
            config={"task_type": "RETRIEVAL_DOCUMENT"},
        )
        all_embeddings.extend(e.values for e in result.embeddings)

        # Respect rate limits on free tier
        if i + batch_size < len(texts):
            time.sleep(0.5)

    return all_embeddings


def _get_embeddings_with_retry(texts: list[str]) -> list[list[float]]:
    """Generate embeddings with bounded retries and exponential backoff."""
    max_retries = max(1, config.EMBEDDING_MAX_RETRIES)
    base_delay = max(0.0, config.EMBEDDING_RETRY_BASE_SECONDS)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return get_embeddings(texts)
        except Exception as e:
            last_error = e
            if attempt == max_retries:
                break

            sleep_seconds = base_delay * (2 ** (attempt - 1))
            _log.warning(
                "Embedding batch failed (attempt %s/%s): %s. Retrying in %.1fs",
                attempt,
                max_retries,
                e,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(
        f"Failed to generate embeddings after {max_retries} attempt(s)."
    ) from last_error


def get_query_embedding(query: str) -> list[float]:
    """
    Generate a single query embedding using configured provider.
    """
    if config.EMBEDDING_PROVIDER == "local":
        embedder = _get_local_embedder()
        v = embedder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return v.tolist()

    client = _get_genai_client()

    result = client.models.embed_content(
        model=config.EMBEDDING_MODEL,
        contents=query,
        config={"task_type": "RETRIEVAL_QUERY"},
    )
    return result.embeddings[0].values


# ── Pinecone index management ────────────────────────────────


def init_pinecone():
    """
    Initialize the Pinecone client and create the index if needed.

    Returns:
        A Pinecone Index handle for the configured index name.
    """
    global _pinecone_client

    if _pinecone_client is None:
        if not config.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is not set in .env")
        _pinecone_client = Pinecone(api_key=config.PINECONE_API_KEY)

    index_name = config.PINECONE_INDEX_NAME

    dim = _embedding_dimension()

    if index_name not in _pinecone_client.list_indexes().names():
        _log.info(
            f"Creating Pinecone index '{index_name}' "
            f"(dim={dim}, metric=cosine)..."
        )
        _pinecone_client.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=config.PINECONE_ENVIRONMENT,
            ),
        )
        _log.info(f"Index '{index_name}' created.")
    else:
        _log.info(f"Index '{index_name}' already exists.")

        # Fail fast with a clear message if an existing index has the wrong schema.
        try:
            index_info = _pinecone_client.describe_index(index_name)
            existing_dim = getattr(index_info, "dimension", None)
            if existing_dim is not None and int(existing_dim) != dim:
                raise RuntimeError(
                    f"Pinecone index '{index_name}' has dimension={existing_dim}, "
                    f"but EMBEDDING_DIMENSION={dim}. Delete/recreate the index "
                    "or change EMBEDDING_DIMENSION to match."
                )
        except RuntimeError:
            raise
        except Exception:
            # Don't block if describe_index shape changes or is unavailable.
            _log.debug("Could not validate existing Pinecone index dimension.")

    return _pinecone_client.Index(index_name)


# ── Upsert ───────────────────────────────────────────────────


def upsert_chunks(chunks: list[dict], namespace: str = "") -> int:
    """
    Embed and upsert filing chunks to Pinecone.

    Args:
        chunks: List of FilingChunk-like dicts from the chunker.
        namespace: Pinecone namespace (e.g. source_document name).

    Returns:
        Number of vectors upserted.
    """
    if not chunks:
        _log.warning("No chunks to upsert.")
        return 0

    index = init_pinecone()
    source_doc = chunks[0].get("source_document", "unknown")
    _log.info(
        f"Upserting {len(chunks)} chunks from '{source_doc}' "
        f"to namespace '{namespace}'..."
    )

    batch_size = 100
    total_upserted = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]

        # Generate dense embeddings
        embeddings = _get_embeddings_with_retry(texts)
        if len(embeddings) != len(texts):
            raise RuntimeError(
                "Embedding count mismatch: "
                f"expected {len(texts)}, got {len(embeddings)}"
            )

        # Build Pinecone vectors
        vectors = []
        for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            chunk_id = f"{source_doc}_chunk_{i + j}"

            metadata = {
                "source_document": chunk.get("source_document", ""),
                "headings": chunk.get("headings", []),
                "source_page": chunk.get("source_page", 0),
                # Store full text — avg chunks ~1400 chars, max ~3200 chars,
                # well within Pinecone's 40 KB per-vector metadata limit.
                "text": chunk["text"],
            }

            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": metadata,
            })

        upsert_response = index.upsert(vectors=vectors, namespace=namespace)

        # Pinecone SDK typically returns UpsertResponse.upserted_count.
        # Support camelCase dict payloads as a defensive fallback.
        acknowledged = getattr(upsert_response, "upserted_count", None)
        if acknowledged is None and isinstance(upsert_response, dict):
            acknowledged = upsert_response.get("upserted_count")
            if acknowledged is None:
                acknowledged = upsert_response.get("upsertedCount")

        if acknowledged is None:
            acknowledged = len(vectors)
            _log.warning(
                "  Batch %s: upsert response missing acknowledged count; "
                "falling back to submitted vector count=%s",
                i // batch_size + 1,
                acknowledged,
            )
        elif acknowledged != len(vectors):
            _log.warning(
                "  Batch %s: submitted %s vectors, Pinecone acknowledged %s",
                i // batch_size + 1,
                len(vectors),
                acknowledged,
            )

        total_upserted += int(acknowledged)
        _log.info(
            "  Batch %s: %s vectors acknowledged by Pinecone",
            i // batch_size + 1,
            int(acknowledged),
        )

    _log.info(f"Upserted {total_upserted} chunks from '{source_doc}'.")
    return total_upserted


# ── Query ────────────────────────────────────────────────────


def query_index(
    query: str,
    section_types: list[str] | None = None,
    top_k: int | None = None,
    namespace: str = "",
) -> list[dict]:
    """
    Query the Pinecone index with optional section-type score boosting.

    Args:
        query: Natural language query string.
        section_types: Optional list of preferred section types.
            Matching chunks receive a +SECTION_BOOST_WEIGHT score boost
            (soft filter — non-matching chunks are NOT excluded).
        top_k: Number of results to return (default from config).
        namespace: Pinecone namespace to search.

        Returns:
        List of dicts sorted by score, each with:
            text, score, headings, source_document, source_page
    """
    if top_k is None:
        top_k = config.RETRIEVAL_TOP_K

    index = init_pinecone()
    query_embedding = get_query_embedding(query)

    # Over-fetch when boosting so we can re-rank
    fetch_k = top_k * 2 if section_types else top_k

    results = index.query(
        vector=query_embedding,
        top_k=fetch_k,
        include_metadata=True,
        namespace=namespace,
    )

    # Parse matching results
    parsed = []
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        score = match["score"]

        # Note: Section-type boosting removed as extraction is deferred

        parsed.append({
            "text": meta.get("text", ""),
            "score": score,
            "headings": meta.get("headings", []),
            "source_document": meta.get("source_document", ""),
            "source_page": meta.get("source_page", 0),
        })

    # Re-sort by score and trim to top_k
    parsed.sort(key=lambda x: x["score"], reverse=True)
    return parsed[:top_k]


# ── Full index build ─────────────────────────────────────────


def build_index(input_path: Path | None = None) -> None:
    """
    Run the full indexing pipeline: load post-processed chunks and upsert to Pinecone.

    Reads from *_processed.json files produced by post_process.py.
    Each filing is upserted into its own Pinecone namespace (source_document name).

    Args:
        input_path: Path to a specific *_processed.json file, or None to
            process all *_processed.json files in data/parsed/.
    """
    if input_path is not None:
        processed_files = [input_path]
    else:
        processed_files = sorted(config.PARSED_DIR.glob("*_processed.json"))

    if not processed_files:
        _log.error(
            "No *_processed.json files found in data/parsed/. "
            "Run post_process.py first."
        )
        return

    for path in processed_files:
        _log.info(f"Loading {path.name}...")
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not chunks:
            _log.warning(f"  {path.name} is empty, skipping.")
            continue

        source_doc = chunks[0].get("source_document", path.stem.replace("_processed", ""))
        _log.info(f"  {source_doc}: {len(chunks)} chunks → upserting to namespace '{source_doc}'")
        upsert_chunks(chunks, namespace=source_doc)

    _log.info("Index build complete.")


# ── CLI entry-point ──────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Embed and upsert *_processed.json filing chunks to Pinecone."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help=(
            "Path to a specific *_processed.json file. "
            "If omitted, all *_processed.json files in data/parsed/ are indexed."
        ),
    )
    args = parser.parse_args()

    errors = config.validate_config()
    if errors:
        print("Config validation failed:")
        for error in errors:
            print(f"  - {error}")
    else:
        build_index(Path(args.input_file) if args.input_file else None)
