"""
Phase 5 — Vector Database
ChromaDB-backed persistent vector store with cosine similarity.
"""

import logging
from typing import Optional

import chromadb

from .config import CHROMA_DB_DIR, COLLECTION_NAME

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Thin wrapper around a ChromaDB persistent collection.

    Uses cosine similarity (hnsw:space = cosine) so that
    similarity = 1 − returned_distance.
    """

    def __init__(self, persist_directory: str | None = None):
        self._path = persist_directory or str(CHROMA_DB_DIR)
        self._client: chromadb.PersistentClient | None = None
        self._collection = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Connect to (or create) the ChromaDB collection."""
        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self._path)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB ready at '{self._path}' — "
            f"collection '{COLLECTION_NAME}' has {self._collection.count()} docs"
        )

    def reset(self) -> None:
        """Drop and recreate the collection (wipes all stored data)."""
        if self._client is None:
            self.initialize()
        try:
            self._client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection reset — all documents removed.")

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        batch_size: int = 100,
    ) -> None:
        """
        Persist chunk dicts together with their pre-computed embeddings.

        Args:
            chunks:     list of chunk dicts (must have chunk_id, text, source).
            embeddings: parallel list of embedding vectors.
            batch_size: number of records per ChromaDB upsert call.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length."
            )

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "source":       c["source"],
                "url":          c.get("url", ""),
                "token_count":  c.get("token_count", 0),
                "start_token":  c.get("start_token", 0),
                "chunk_index":  c.get("chunk_index", 0),
                "total_chunks": c.get("total_chunks", 0),
            }
            for c in chunks
        ]

        for i in range(0, len(ids), batch_size):
            sl = slice(i, i + batch_size)
            self._collection.upsert(
                ids=ids[sl],
                embeddings=embeddings[sl],
                documents=documents[sl],
                metadatas=metadatas[sl],
            )

        logger.info(f"Stored {len(chunks)} chunks in ChromaDB.")

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[dict]:
        """
        Return the top-k most similar chunks.

        Each result dict:
            chunk_id    : str
            text        : str
            source      : str
            similarity  : float in [0, 1]  (1 = identical)
        """
        n = min(k, self._collection.count())
        if n == 0:
            return []

        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        results: list[dict] = []
        for idx in range(len(raw["ids"][0])):
            distance = raw["distances"][0][idx]
            meta = raw["metadatas"][0][idx]
            results.append(
                {
                    "chunk_id":    raw["ids"][0][idx],
                    "text":        raw["documents"][0][idx],
                    "source":      meta["source"],
                    "url":         meta.get("url", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "total_chunks":meta.get("total_chunks", 0),
                    "token_count": meta.get("token_count", 0),
                    "similarity":  round(1.0 - distance, 4),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count(self) -> int:
        if self._collection is None:
            return 0
        return self._collection.count()

    def is_populated(self) -> bool:
        return self.count() > 0


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    """Return the shared VectorStore instance (lazy-initialised)."""
    global _store
    if _store is None:
        _store = VectorStore()
        _store.initialize()
    return _store
