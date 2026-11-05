"""
tests/test_grounding.py — WikiTruth AI
Tests for grounding prompts, context validation, and grounding status detection.
"""

import pytest
sys_path_fix = None

try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
except Exception:
    pass

from src.grounding import (
    build_prompt,
    validate_context,
    detect_grounding_status,
    get_status_badge,
    GroundingStatus,
    ZERO_SHOT_TEMPLATE,
    FEW_SHOT_TEMPLATE,
)


# Sample chunks for testing
SAMPLE_CHUNKS = [
    {
        "source": "Machine learning",
        "text": "Machine learning is a subset of artificial intelligence that gives computers the ability to learn from data without being explicitly programmed.",
        "similarity": 0.85,
        "url": "https://en.wikipedia.org/wiki/Machine_learning",
    },
    {
        "source": "Deep learning",
        "text": "Deep learning uses multi-layered neural networks to learn representations from raw data automatically.",
        "similarity": 0.72,
        "url": "https://en.wikipedia.org/wiki/Deep_learning",
    },
]


class TestBuildPrompt:
    def test_zero_shot_prompt_contains_question(self):
        _, prompt = build_prompt("What is ML?", SAMPLE_CHUNKS, style="zero_shot")
        assert "What is ML?" in prompt

    def test_zero_shot_prompt_contains_sources(self):
        _, prompt = build_prompt("What is ML?", SAMPLE_CHUNKS, style="zero_shot")
        assert "Machine learning" in prompt
        assert "Deep learning" in prompt

    def test_few_shot_prompt_contains_examples(self):
        _, prompt = build_prompt("What is ML?", SAMPLE_CHUNKS, style="few_shot")
        assert "EXAMPLE" in prompt

    def test_few_shot_prompt_contains_question(self):
        _, prompt = build_prompt("What is ML?", SAMPLE_CHUNKS, style="few_shot")
        assert "What is ML?" in prompt

    def test_zero_shot_is_different_from_few_shot(self):
        _, zero_prompt = build_prompt("What is ML?", SAMPLE_CHUNKS, style="zero_shot")
        _, few_prompt = build_prompt("What is ML?", SAMPLE_CHUNKS, style="few_shot")
        assert zero_prompt != few_prompt

    def test_empty_chunks(self):
        _, prompt = build_prompt("What is ML?", [], style="zero_shot")
        assert "What is ML?" in prompt

    def test_returns_tuple_of_two_strings(self):
        result = build_prompt("Q?", SAMPLE_CHUNKS)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(s, str) for s in result)


class TestValidateContext:
    def test_high_similarity_is_valid(self):
        is_valid, avg_sim = validate_context(SAMPLE_CHUNKS, threshold=0.25)
        assert is_valid is True
        assert avg_sim > 0.25

    def test_low_similarity_is_invalid(self):
        low_chunks = [
            {"similarity": 0.10, "source": "X", "text": "irrelevant"},
        ]
        is_valid, avg_sim = validate_context(low_chunks, threshold=0.25)
        assert is_valid is False

    def test_empty_chunks_is_invalid(self):
        is_valid, avg_sim = validate_context([], threshold=0.25)
        assert is_valid is False
        assert avg_sim == 0.0

    def test_returns_avg_similarity(self):
        _, avg = validate_context(SAMPLE_CHUNKS)
        expected = (0.85 + 0.72) / 2
        assert abs(avg - expected) < 0.01


class TestDetectGroundingStatus:
    def test_grounded_answer(self):
        answer = "Machine learning is a subset of AI [Source 1]."
        status = detect_grounding_status(answer, SAMPLE_CHUNKS)
        assert status == GroundingStatus.GROUNDED

    def test_unsupported_cannot_answer(self):
        answer = "I cannot answer this question based on the available Wikipedia sources."
        status = detect_grounding_status(answer, SAMPLE_CHUNKS)
        assert status == GroundingStatus.UNSUPPORTED

    def test_unsupported_dont_know(self):
        answer = "I don't know the answer to this question."
        status = detect_grounding_status(answer, SAMPLE_CHUNKS)
        assert status == GroundingStatus.UNSUPPORTED

    def test_low_confidence_empty_chunks(self):
        answer = "Some answer."
        status = detect_grounding_status(answer, [])
        assert status == GroundingStatus.LOW_CONFIDENCE

    def test_low_confidence_low_sim_chunks(self):
        low_chunks = [{"similarity": 0.05, "source": "X", "text": "y"}]
        answer = "Some answer."
        status = detect_grounding_status(answer, low_chunks)
        assert status == GroundingStatus.LOW_CONFIDENCE


class TestStatusBadge:
    def test_grounded_badge(self):
        badge = get_status_badge(GroundingStatus.GROUNDED)
        assert badge["color"] == "green"
        assert "✅" in badge["icon"]

    def test_unsupported_badge(self):
        badge = get_status_badge(GroundingStatus.UNSUPPORTED)
        assert badge["color"] == "red"

    def test_low_confidence_badge(self):
        badge = get_status_badge(GroundingStatus.LOW_CONFIDENCE)
        assert badge["color"] == "gray"

    def test_all_statuses_have_badges(self):
        for status in GroundingStatus:
            badge = get_status_badge(status)
            assert "label" in badge
            assert "color" in badge
            assert "icon" in badge
