"""
tests/test_app.py — WikiTruth AI
Basic smoke tests for app imports and pipeline initialization.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestImports:
    def test_import_config(self):
        from src.config import APP_NAME, TOPICS, CHUNK_SIZE
        assert APP_NAME == "WikiTruth AI"
        assert len(TOPICS) > 0
        assert CHUNK_SIZE > 0

    def test_import_grounding(self):
        from src.grounding import GroundingStatus, build_prompt, validate_context
        assert GroundingStatus.GROUNDED is not None

    def test_import_wikipedia_retriever(self):
        from src.wikipedia_retriever import get_article_url, get_source_links
        url = get_article_url("Test article")
        assert "wikipedia.org" in url

    def test_import_utils(self):
        from src.utils import format_similarity_bar, truncate_text, compute_hallucination_rate
        bar = format_similarity_bar(0.5)
        assert isinstance(bar, str)

    def test_import_image_generator(self):
        from src.image_generator import generate_image, generate_image_matplotlib
        # Should import without error
        assert callable(generate_image)
        assert callable(generate_image_matplotlib)

    def test_import_pipeline(self):
        from pipeline import RAGPipeline
        # Should instantiate without error (even without DB)
        # Note: DB connection happens lazily
        pipeline = RAGPipeline()
        assert hasattr(pipeline, 'ingest')
        assert hasattr(pipeline, 'query')
        assert hasattr(pipeline, 'query_ungrounded')

    def test_grounding_status_enum(self):
        from src.grounding import GroundingStatus
        statuses = list(GroundingStatus)
        assert len(statuses) == 4

    def test_config_topics(self):
        from src.config import TOPICS
        assert "Machine learning" in TOPICS
        assert "Deep learning" in TOPICS
        assert len(TOPICS) == 10


class TestImageGeneratorFallback:
    def test_matplotlib_fallback_returns_bytes(self):
        from src.image_generator import generate_image_matplotlib
        result = generate_image_matplotlib("Machine learning")
        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 1000  # Should be a real PNG

    def test_matplotlib_fallback_known_topic(self):
        from src.image_generator import generate_image_matplotlib
        result = generate_image_matplotlib("Deep learning")
        assert result is not None

    def test_matplotlib_fallback_unknown_topic(self):
        from src.image_generator import generate_image_matplotlib
        result = generate_image_matplotlib("Quantum entanglement in black holes")
        # Should still work with generic keywords
        assert result is not None
