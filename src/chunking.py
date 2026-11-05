"""
Phase 3 — Chunking
Split raw Wikipedia articles into overlapping token-based chunks
and save them as JSON to data/processed_chunks/.
"""

import json
import logging
from pathlib import Path

import tiktoken

from .config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TIKTOKEN_ENCODING,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filename → display title mapping
# ---------------------------------------------------------------------------
_FILENAME_TO_TITLE: dict[str, str] = {
    "artificial_intelligence": "Artificial intelligence",
    "machine_learning": "Machine learning",
    "deep_learning": "Deep learning",
    "neural_network": "Neural network",
    "supervised_learning": "Supervised learning",
    "unsupervised_learning": "Unsupervised learning",
    "reinforcement_learning": "Reinforcement learning",
    "data_science": "Data science",
    "big_data": "Big data",
    "statistics": "Statistics",
}

_TITLE_TO_URL: dict[str, str] = {
    "Artificial intelligence": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "Machine learning":        "https://en.wikipedia.org/wiki/Machine_learning",
    "Deep learning":           "https://en.wikipedia.org/wiki/Deep_learning",
    "Neural network":          "https://en.wikipedia.org/wiki/Neural_network",
    "Supervised learning":     "https://en.wikipedia.org/wiki/Supervised_learning",
    "Unsupervised learning":   "https://en.wikipedia.org/wiki/Unsupervised_learning",
    "Reinforcement learning":  "https://en.wikipedia.org/wiki/Reinforcement_learning",
    "Data science":            "https://en.wikipedia.org/wiki/Data_science",
    "Big data":                "https://en.wikipedia.org/wiki/Big_data",
    "Statistics":              "https://en.wikipedia.org/wiki/Statistics",
}


def _get_tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding(TIKTOKEN_ENCODING)


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    source: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split *text* into overlapping chunks measured in tokens.

    Each chunk dict:
        chunk_id      : e.g. "machine_lea_004"
        text          : decoded text
        source        : display title ("Machine learning")
        url           : Wikipedia article URL
        token_count   : number of tokens in this chunk
        start_token   : token offset in the full document
        chunk_index   : position of this chunk within the article (0-based)
        total_chunks  : total number of chunks in the article (filled in post-hoc)
    """
    tokenizer = _get_tokenizer()
    tokens = tokenizer.encode(text)

    # Build a short safe prefix for chunk IDs (max 12 chars)
    prefix = source.lower().replace(" ", "_")[:12]

    chunks: list[dict] = []
    start = 0
    idx = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_str = tokenizer.decode(chunk_tokens).strip()

        # Skip empty chunks (can happen near end of document)
        if chunk_text_str:
            chunks.append(
                {
                    "chunk_id": f"{prefix}_{idx:03d}",
                    "text": chunk_text_str,
                    "source": source,
                    "url": _TITLE_TO_URL.get(source, ""),
                    "token_count": len(chunk_tokens),
                    "start_token": start,
                    "chunk_index": idx,
                    "total_chunks": 0,  # back-filled below
                }
            )
            idx += 1

        if end == len(tokens):
            break

        start = end - overlap  # step back by overlap

    # Back-fill total_chunks now that we know the final count
    for chunk in chunks:
        chunk["total_chunks"] = len(chunks)

    return chunks


# ---------------------------------------------------------------------------
# File-level helpers
# ---------------------------------------------------------------------------

def _stem_to_title(stem: str) -> str:
    """Map a file stem like 'big_data' to its display title."""
    return _FILENAME_TO_TITLE.get(stem, stem.replace("_", " ").title())


def process_article(filepath: Path) -> list[dict]:
    """Read one raw .txt file and return its list of chunk dicts."""
    text = filepath.read_text(encoding="utf-8")
    source = _stem_to_title(filepath.stem)
    chunks = chunk_text(text, source)
    logger.info(f"  {filepath.name} → {len(chunks)} chunks")
    return chunks


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_all(force: bool = False) -> list[dict]:
    """
    Process every .txt file in data/raw/ into chunks.

    Saves:
      - data/processed_chunks/<stem>_chunks.json  (per-article)
      - data/processed_chunks/all_chunks.json     (combined)

    Args:
        force: re-chunk even if all_chunks.json already exists.

    Returns:
        The full list of chunk dicts.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = PROCESSED_DATA_DIR / "all_chunks.json"

    if not force and combined_path.exists():
        logger.info("Chunks already exist — loading from disk (use force=True to re-chunk)")
        return load_chunks()

    raw_files = sorted(RAW_DATA_DIR.glob("*.txt"))
    if not raw_files:
        raise FileNotFoundError(
            f"No .txt files found in {RAW_DATA_DIR}. "
            "Run data collection first."
        )

    logger.info(f"Chunking {len(raw_files)} article(s)…")
    all_chunks: list[dict] = []

    for filepath in raw_files:
        chunks = process_article(filepath)
        all_chunks.extend(chunks)

        per_article_path = PROCESSED_DATA_DIR / (filepath.stem + "_chunks.json")
        per_article_path.write_text(
            json.dumps(chunks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    combined_path.write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(
        f"Chunking complete — {len(all_chunks)} total chunks "
        f"saved to {PROCESSED_DATA_DIR}"
    )
    return all_chunks


def load_chunks() -> list[dict]:
    """Load all chunks from disk (requires process_all to have been run)."""
    combined_path = PROCESSED_DATA_DIR / "all_chunks.json"
    if not combined_path.exists():
        raise FileNotFoundError(
            f"No chunks found at {combined_path}. Run the ingestion pipeline first."
        )
    with combined_path.open(encoding="utf-8") as fh:
        return json.load(fh)
