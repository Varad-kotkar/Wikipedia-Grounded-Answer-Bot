"""
tests/test_retrieval.py — WikiTruth AI
Tests for the retrieval pipeline components.
Note: ChromaDB-dependent tests require prior ingestion.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.wikipedia_retriever import get_article_url, get_source_links, format_source_citation


class TestWikipediaURLs:
    def test_basic_url_construction(self):
        url = get_article_url("Machine learning")
        assert url == "https://en.wikipedia.org/wiki/Machine_learning"

    def test_url_encoding_special_chars(self):
        url = get_article_url("Artificial intelligence")
        assert "Artificial_intelligence" in url
        assert url.startswith("https://en.wikipedia.org/wiki/")

    def test_single_word(self):
        url = get_article_url("Statistics")
        assert url == "https://en.wikipedia.org/wiki/Statistics"

    def test_url_has_no_spaces(self):
        url = get_article_url("Deep learning")
        assert " " not in url


class TestSourceLinks:
    def setup_method(self):
        self.chunks = [
            {"source": "Machine learning", "text": "ML text", "similarity": 0.9, "url": ""},
            {"source": "Machine learning", "text": "ML text 2", "similarity": 0.8, "url": ""},
            {"source": "Deep learning", "text": "DL text", "similarity": 0.7, "url": ""},
        ]

    def test_returns_unique_sources(self):
        links = get_source_links(self.chunks)
        titles = [l["title"] for l in links]
        assert len(titles) == len(set(titles))

    def test_chunk_count_correct(self):
        links = get_source_links(self.chunks)
        ml_link = next(l for l in links if l["title"] == "Machine learning")
        assert ml_link["chunk_count"] == 2

    def test_sorted_by_chunk_count(self):
        links = get_source_links(self.chunks)
        # Machine learning has 2 chunks, should be first
        assert links[0]["title"] == "Machine learning"

    def test_url_constructed_when_missing(self):
        links = get_source_links(self.chunks)
        for link in links:
            assert link["url"].startswith("https://en.wikipedia.org/wiki/")

    def test_empty_chunks(self):
        links = get_source_links([])
        assert links == []

    def test_format_citation_returns_string(self):
        citation = format_source_citation(self.chunks)
        assert isinstance(citation, str)
        assert "Machine learning" in citation


class TestUtils:
    def setup_method(self):
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.utils import format_similarity_bar, truncate_text
        self.format_bar = format_similarity_bar
        self.truncate = truncate_text

    def test_similarity_bar_format(self):
        bar = self.format_bar(0.75)
        assert "█" in bar
        assert "░" in bar
        assert "0.750" in bar

    def test_similarity_bar_full(self):
        bar = self.format_bar(1.0)
        assert "░" not in bar

    def test_similarity_bar_empty(self):
        bar = self.format_bar(0.0)
        assert "█" not in bar

    def test_truncate_short_text(self):
        text = "Hello world"
        assert self.truncate(text, 100) == text

    def test_truncate_long_text(self):
        text = "A" * 300
        result = self.truncate(text, 200)
        assert len(result) <= 200
        assert result.endswith("...")
