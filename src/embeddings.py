"""
Phase 4 — Embeddings
Converts text chunks into dense vector embeddings.
Supports sentence-transformers (local, default) and OpenAI.
"""

import logging
from typing import Optional

from .config import (
    EMBEDDING_PROVIDER,
    SENTENCE_TRANSFORMER_MODEL,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """
    Unified embedding interface.

    Providers:
      - "sentence-transformers"  local model, no API key required (default)
      - "openai"                 requires OPENAI_API_KEY env var
    """

    def __init__(self, provider: str = EMBEDDING_PROVIDER):
        self.provider = provider
        self._model = None
        self._client = None
        self._dim: int = 0
        self._setup()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        if self.provider == "sentence-transformers":
            self._setup_st()
        elif self.provider == "openai":
            self._setup_openai()
        elif self.provider == "anthropic":
            logger.warning(
                "Anthropic does not expose a public embedding API. "
                "Falling back to sentence-transformers."
            )
            self.provider = "sentence-transformers"
            self._setup_st()
        else:
            raise ValueError(
                f"Unknown EMBEDDING_PROVIDER '{self.provider}'. "
                "Choose 'sentence-transformers' or 'openai'."
            )

    def _setup_st(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        logger.info(f"Loading sentence-transformer model: {SENTENCE_TRANSFORMER_MODEL}")
        self._model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        # get_embedding_dimension is the current API; fall back for older versions
        get_dim = getattr(
            self._model,
            "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        self._dim = get_dim()
        logger.info(f"Model ready — embedding dim: {self._dim}")

    def _setup_openai(self) -> None:
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Export it or switch to EMBEDDING_PROVIDER=sentence-transformers."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai not installed. Run: pip install openai") from exc

        self._client = OpenAI(api_key=OPENAI_API_KEY)
        self._dim = 1536  # text-embedding-3-small
        logger.info("OpenAI embedding provider ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        """Embed a single string. Returns a flat list of floats."""
        return self.embed_batch([text])[0]

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Embed a list of strings.
        Returns a list of float lists (one per input text).
        """
        if not texts:
            return []

        if self.provider == "sentence-transformers":
            import numpy as np

            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,   # cosine-ready
                convert_to_numpy=True,
            )
            return embeddings.tolist()

        elif self.provider == "openai":
            all_embeddings: list[list[float]] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = self._client.embeddings.create(
                    model=OPENAI_EMBEDDING_MODEL,
                    input=batch,
                )
                all_embeddings.extend(item.embedding for item in response.data)
            return all_embeddings

        raise RuntimeError(f"Provider '{self.provider}' not handled in embed_batch.")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_provider: Optional[EmbeddingProvider] = None


def get_provider() -> EmbeddingProvider:
    """Return the shared EmbeddingProvider instance (lazy-initialised)."""
    global _provider
    if _provider is None:
        _provider = EmbeddingProvider()
    return _provider
