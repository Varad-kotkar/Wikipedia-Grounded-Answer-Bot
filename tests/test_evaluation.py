"""
tests/test_evaluation.py — WikiTruth AI
Tests for evaluation scoring logic and metric calculation.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluate import _auto_score, _compute_metrics
from src.utils import compute_metrics_summary, compute_hallucination_rate


class TestAutoScore:
    def test_correct_answer(self):
        keywords = ["algorithm", "data", "learn", "training"]
        answer = "Machine learning is an algorithm that learns from training data."
        score = _auto_score(answer, keywords)
        assert score == 2  # 100% keyword match → 2

    def test_partial_answer(self):
        keywords = ["algorithm", "data", "learn", "training", "neural", "model", "subset"]
        answer = "ML uses data and algorithms."
        score = _auto_score(answer, keywords)
        # 2/7 = 28.6% → score 1
        assert score in (1, 2)  # depends on exact matches

    def test_wrong_answer(self):
        keywords = ["algorithm", "data", "learn", "training", "neural", "model", "subset"]
        answer = "The sky is blue."
        score = _auto_score(answer, keywords)
        assert score == 0

    def test_i_dont_know_scores_zero(self):
        answer = "I don't know the answer to this question."
        score = _auto_score(answer, ["algorithm", "data"])
        assert score == 0

    def test_i_cannot_answer_scores_zero(self):
        answer = "I cannot answer this question based on the available Wikipedia sources."
        score = _auto_score(answer, ["algorithm", "data"])
        assert score == 0

    def test_empty_keywords(self):
        answer = "Some answer"
        score = _auto_score(answer, [])
        assert score == 1  # default when no keywords

    def test_case_insensitive(self):
        keywords = ["MACHINE", "LEARNING"]
        answer = "Machine learning is important."
        score = _auto_score(answer, keywords)
        assert score == 2


class TestComputeMetrics:
    def setup_method(self):
        self.results = [
            {"id": "q1", "difficulty": "easy", "score": 2, "answer": "Correct answer", "avg_retrieval_similarity": 0.8},
            {"id": "q2", "difficulty": "easy", "score": 1, "answer": "Partial answer", "avg_retrieval_similarity": 0.6},
            {"id": "q3", "difficulty": "medium", "score": 0, "answer": "Wrong answer", "avg_retrieval_similarity": 0.3},
            {"id": "q4", "difficulty": "medium", "score": 2, "answer": "Correct answer", "avg_retrieval_similarity": 0.9},
            {"id": "q5", "difficulty": "hard", "score": 0, "answer": "I don't know", "avg_retrieval_similarity": 0.1},
        ]

    def test_total_questions(self):
        metrics = _compute_metrics(self.results)
        assert metrics["total_questions"] == 5

    def test_correct_count(self):
        metrics = _compute_metrics(self.results)
        assert metrics["correct_2"] == 2

    def test_partial_count(self):
        metrics = _compute_metrics(self.results)
        assert metrics["partial_1"] == 1

    def test_wrong_count(self):
        metrics = _compute_metrics(self.results)
        assert metrics["wrong_0"] == 2

    def test_accuracy_pct(self):
        metrics = _compute_metrics(self.results)
        # 2 correct out of 5 = 40%
        assert metrics["overall_accuracy_pct"] == 40.0

    def test_empty_results(self):
        metrics = _compute_metrics([])
        assert metrics == {}

    def test_by_difficulty_exists(self):
        metrics = _compute_metrics(self.results)
        assert "by_difficulty" in metrics
        assert "easy" in metrics["by_difficulty"]


class TestHallucinationRate:
    def test_no_hallucinations(self):
        results = [
            {"score": 2, "answer": "Correct answer"},
            {"score": 1, "answer": "Partial answer"},
        ]
        rate = compute_hallucination_rate(results)
        assert rate == 0.0

    def test_hallucinated_answers(self):
        results = [
            {"score": 0, "answer": "Completely wrong fabricated answer"},
            {"score": 2, "answer": "Correct answer"},
        ]
        rate = compute_hallucination_rate(results)
        assert rate == 50.0

    def test_refusal_not_hallucination(self):
        results = [
            {"score": 0, "answer": "I cannot answer this question"},
            {"score": 2, "answer": "Correct answer"},
        ]
        rate = compute_hallucination_rate(results)
        # Refusal is NOT hallucination
        assert rate == 0.0

    def test_empty_results(self):
        rate = compute_hallucination_rate([])
        assert rate == 0.0
