"""
Phase 6 — Retrieval
Embeds a user query and fetches the top-k most relevant chunks
from the ChromaDB vector store.
"""

import logging
from typing import Optional

from .config import TOP_K
from .embeddings import get_provider
from .vector_store import get_store

logger = logging.getLogger(__name__)


class Retriever:
    """
    Query pipeline:
        user query (str)
            → embedding vector
            → ChromaDB cosine search
            → ranked list of chunk dicts
    """

    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k
        self._embedder = get_provider()
        self._store = get_store()

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        """
        Retrieve the most relevant chunks for *query*.

        Args:
            query: natural-language question.
            k:     number of chunks to return (overrides self.top_k).

        Returns:
            List of dicts with keys: chunk_id, text, source, similarity.
            Sorted descending by similarity.
        """
        k = k or self.top_k

        if not self._store.is_populated():
            raise RuntimeError(
                "Vector store is empty. Run 'python main.py ingest' first."
            )

        query_vec = self._embedder.embed(query)
        results = self._store.search(query_vec, k=k)

        logger.info(f"Query: '{query[:60]}' → retrieved {len(results)} chunks")
        for i, r in enumerate(results, 1):
            logger.debug(
                f"  {i}. [{r['source']}] sim={r['similarity']:.3f}  "
                f"{r['text'][:80]}…"
            )

        return results


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """Return the shared Retriever instance (lazy-initialised)."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
