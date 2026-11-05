"""
src/utils.py — WikiTruth AI
Utility functions for export, formatting, and metric calculation.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


def export_to_csv(results: List[Dict], path: str) -> None:
    """
    Export evaluation results to CSV format.
    Creates parent directories if they don't exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if not results:
        return
    
    # Flatten nested fields for CSV
    flat_results = []
    for r in results:
        flat = {
            "id": r.get("id", ""),
            "question": r.get("question", ""),
            "difficulty": r.get("difficulty", ""),
            "category": r.get("category", ""),
            "grounded_answer": r.get("grounded_answer", r.get("answer", "")),
            "ungrounded_answer": r.get("ungrounded_answer", ""),
            "expected_answer": r.get("expected_answer", ""),
            "grounded_score": r.get("grounded_score", r.get("score", "")),
            "ungrounded_score": r.get("ungrounded_score", ""),
            "avg_retrieval_similarity": r.get("avg_retrieval_similarity", 0),
            "grounding_status": r.get("grounding_status", ""),
        }
        flat_results.append(flat)
    
    fieldnames = list(flat_results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_results)


def export_to_json(data: Any, path: str) -> None:
    """
    Export any data structure to JSON format.
    Creates parent directories if they don't exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: str) -> Any:
    """Load a JSON file. Returns None if file doesn't exist."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def compute_hallucination_rate(results: List[Dict], mode: str = "grounded") -> float:
    """
    Compute hallucination rate.
    Hallucination = answer containing unsupported claims (score 0 and not 'I cannot answer')
    
    Args:
        results: list of result dicts
        mode: 'grounded' or 'ungrounded'
    """
    if not results:
        return 0.0
    
    score_key = f"{mode}_score" if f"{mode}_score" in results[0] else "score"
    answer_key = f"{mode}_answer" if f"{mode}_answer" in results[0] else "answer"
    
    hallucinated = 0
    for r in results:
        answer = r.get(answer_key, "").lower()
        score = r.get(score_key, 1)
        # Hallucination: score 0 AND the model didn't refuse
        refusal_phrases = ["i cannot answer", "i don't know", "i do not know"]
        is_refusal = any(phrase in answer for phrase in refusal_phrases)
        if score == 0 and not is_refusal:
            hallucinated += 1
    
    return round(hallucinated / len(results) * 100, 1)


def format_similarity_bar(score: float, width: int = 20) -> str:
    """
    Create a text-based similarity bar.
    Example: 0.75 → '███████████████░░░░░ 0.75'
    """
    filled = int(score * width)
    empty = width - filled
    bar = "█" * filled + "░" * empty
    return f"{bar} {score:.3f}"


def truncate_text(text: str, max_chars: int = 200, suffix: str = "...") -> str:
    """Smart text truncation at word boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars - len(suffix)]
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]
    return truncated + suffix


def get_timestamp() -> str:
    """Return ISO 8601 UTC timestamp string."""
    return datetime.utcnow().isoformat() + "Z"


def compute_metrics_summary(results: List[Dict], mode: str = "grounded") -> Dict:
    """
    Compute comprehensive metrics from evaluation results.
    Works for both 'grounded' and 'ungrounded' modes.
    """
    if not results:
        return {}
    
    score_key = f"{mode}_score" if f"{mode}_score" in (results[0] if results else {}) else "score"
    
    total = len(results)
    scores = [r.get(score_key, 0) for r in results]
    correct = sum(1 for s in scores if s == 2)
    partial = sum(1 for s in scores if s == 1)
    wrong = sum(1 for s in scores if s == 0)
    
    accuracy = round(correct / total * 100, 1)
    error_rate = round(wrong / total * 100, 1)
    partial_rate = round(partial / total * 100, 1)
    hallucination_rate = compute_hallucination_rate(results, mode)
    
    # By difficulty
    by_difficulty = {}
    for diff in ("easy", "medium", "hard", "synthesis"):
        subset = [r for r in results if r.get("difficulty") == diff]
        if subset:
            subset_scores = [r.get(score_key, 0) for r in subset]
            by_difficulty[diff] = {
                "count": len(subset),
                "correct": sum(1 for s in subset_scores if s == 2),
                "accuracy_pct": round(sum(1 for s in subset_scores if s == 2) / len(subset) * 100, 1),
                "avg_score": round(sum(subset_scores) / len(subset), 3),
            }
    
    return {
        "mode": mode,
        "total": total,
        "correct": correct,
        "partial": partial,
        "wrong": wrong,
        "accuracy_pct": accuracy,
        "error_rate_pct": error_rate,
        "partial_rate_pct": partial_rate,
        "hallucination_rate_pct": hallucination_rate,
        "weighted_score_pct": round(sum(scores) / (total * 2) * 100, 1),
        "by_difficulty": by_difficulty,
    }
