"""
Phase 9 & 10 — Evaluation Harness — WikiTruth AI
Runs test questions through the RAG pipeline and scores answers.
Supports dual-track evaluation: Grounded (Mode B) vs Ungrounded (Mode A).

Scoring scale:
  2 = Correct (accurate and complete)
  1 = Partially correct (right direction, missing key details)
  0 = Incorrect / "I don't know" for an answerable question
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union

# Allow running directly from the project root
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline import RAGPipeline
from src.config import EVAL_RESULTS_DIR
from src.utils import (
    export_to_csv,
    export_to_json,
    compute_metrics_summary,
    compute_hallucination_rate,
    get_timestamp,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def auto_score(answer: str, expected_keywords: list) -> int:
    """
    Keyword-based automatic scoring.

    Counts how many expected keywords appear in the answer (case-insensitive).
    Thresholds:
      ≥ 50 % of keywords → 2
      ≥ 25 % of keywords → 1
      < 25 %             → 0
    Returns 0 if the answer is "I don't know" or refusal (case-insensitive).
    """
    if not expected_keywords:
        return 1  # can't score without keywords

    ans_lower = answer.lower()

    refusal_phrases = [
        "i don't know",
        "i do not know",
        "i cannot answer",
        "cannot answer this question",
        "no relevant passages",
    ]
    if any(phrase in ans_lower for phrase in refusal_phrases):
        return 0

    hits = sum(1 for kw in expected_keywords if kw.lower() in ans_lower)
    ratio = hits / len(expected_keywords)

    if ratio >= 0.5:
        return 2
    elif ratio >= 0.25:
        return 1
    return 0


_auto_score = auto_score


def manual_score(question: str, answer: str, expected: str) -> int:
    """
    Interactive manual scoring. Prompts the evaluator and returns 0/1/2.
    """
    print("\n" + "─" * 60)
    print(f"EXPECTED ANSWER:\n{expected}")
    print("─" * 60)
    print(f"SYSTEM ANSWER:\n{answer}")
    print("─" * 60)

    while True:
        try:
            raw = input("Score [0=wrong / 1=partial / 2=correct] (or 's' to skip): ").strip()
            if raw.lower() == "s":
                return -1  # sentinel for skipped
            score = int(raw)
            if score in (0, 1, 2):
                return score
            print("  Please enter 0, 1, or 2.")
        except (ValueError, KeyboardInterrupt):
            print("\n  Skipping this question.")
            return -1


_manual_score = manual_score


# ---------------------------------------------------------------------------
# Metrics calculation & printing
# ---------------------------------------------------------------------------

def _compute_metrics(results: list) -> dict:
    if not results:
        return {}

    total = len(results)
    total_score = sum(r.get("score", r.get("grounded_score", 0)) for r in results)
    max_score = total * 2

    # Accuracy: treat score 2 as correct
    scores = [r.get("score", r.get("grounded_score", 0)) for r in results]
    correct = sum(1 for s in scores if s == 2)
    partial = sum(1 for s in scores if s == 1)
    wrong = sum(1 for s in scores if s == 0)

    # Refusal count
    idk_count = sum(
        1 for r in results
        if any(phrase in r.get("answer", "").lower() for phrase in ["i don't know", "i do not know", "i cannot answer"])
    )

    # By difficulty
    by_difficulty: dict[str, dict] = {}
    for difficulty in ("easy", "medium", "hard", "synthesis"):
        subset = [r for r in results if r.get("difficulty") == difficulty]
        if subset:
            sub_scores = [r.get("score", r.get("grounded_score", 0)) for r in subset]
            avg = sum(sub_scores) / (len(subset) * 2)
            by_difficulty[difficulty] = {
                "count": len(subset),
                "avg_score": round(avg, 3),
                "accuracy_pct": round(
                    sum(1 for s in sub_scores if s == 2) / len(subset) * 100, 1
                ),
            }

    # Average retrieval similarity
    sims = [r.get("avg_retrieval_similarity", 0) for r in results if r.get("avg_retrieval_similarity", 0) > 0]
    avg_sim = round(sum(sims) / len(sims), 4) if sims else 0.0

    return {
        "total_questions": total,
        "scored_questions": total,
        "total_score": total_score,
        "max_possible_score": max_score,
        "overall_accuracy_pct": round(correct / total * 100, 1),
        "weighted_score_pct": round(total_score / max_score * 100, 1),
        "correct_2": correct,
        "partial_1": partial,
        "wrong_0": wrong,
        "idk_responses": idk_count,
        "hallucination_proxy_pct": round((total - idk_count - correct) / total * 100, 1),
        "avg_retrieval_similarity": avg_sim,
        "by_difficulty": by_difficulty,
    }


def _print_summary(metrics: dict, skipped: int = 0, title: str = "EVALUATION SUMMARY") -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(title)
    print(bar)
    print(f"  Questions scored  : {metrics.get('scored_questions', 0)}")
    if skipped:
        print(f"  Questions skipped : {skipped}")
    print(f"  Correct   (2/2)   : {metrics.get('correct_2', 0)}")
    print(f"  Partial   (1/2)   : {metrics.get('partial_1', 0)}")
    print(f"  Wrong     (0/2)   : {metrics.get('wrong_0', 0)}")
    print(f"  'I don't know'    : {metrics.get('idk_responses', 0)}")
    print()
    print(f"  Overall accuracy  : {metrics.get('overall_accuracy_pct', 0):.1f}%  (score=2)")
    print(f"  Weighted score    : {metrics.get('weighted_score_pct', 0):.1f}%  (incl. partials)")
    print(f"  Avg retrieval sim : {metrics.get('avg_retrieval_similarity', 0):.4f}")
    print()
    print("  By difficulty:")
    for lvl, stats in metrics.get("by_difficulty", {}).items():
        print(
            f"    {lvl:10s} — {stats['count']:2d} Qs, "
            f"accuracy {stats['accuracy_pct']:5.1f}%,  "
            f"avg_score {stats['avg_score']:.3f}"
        )
    print(bar)

    target = 70.0
    achieved = metrics.get("overall_accuracy_pct", 0)
    status = "✓ PASSED" if achieved >= target else "✗ BELOW TARGET"
    print(f"\n  Target: >{target}% accuracy  →  {status} ({achieved:.1f}%)")
    print(bar + "\n")


def _load_questions_list(questions: Union[str, Path, list]) -> list:
    """Helper to ensure questions is a list of question dicts."""
    if isinstance(questions, (str, Path)):
        p = Path(questions)
        if not p.exists():
            raise FileNotFoundError(f"Questions file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("questions", data)
    return list(questions)


# ---------------------------------------------------------------------------
# Individual evaluation tracks
# ---------------------------------------------------------------------------

def run_ungrounded_evaluation(
    questions: Union[list, str, Path],
    top_k: int = 5,
    auto_score: bool = True,
) -> Tuple[list, dict]:
    """
    Mode A: Evaluate answers generated by LLM WITHOUT Wikipedia retrieval.
    """
    q_list = _load_questions_list(questions)
    pipeline = RAGPipeline()
    results = []

    print(f"\n{'='*60}")
    print(f"UNGROUNDED EVALUATION (Mode A) — {len(q_list)} questions")
    print(f"Mode: {'auto keyword-scoring' if auto_score else 'interactive manual scoring'}")
    print(f"{'='*60}\n")

    for i, q in enumerate(q_list, 1):
        print(f"[{i:02d}/{len(q_list)}] ({q['difficulty'].upper():10s}) {q['question']}")
        try:
            res = pipeline.query_ungrounded(q["question"])
            answer = res.get("answer", "")
        except Exception as exc:
            logger.error(f"Ungrounded pipeline error for question {q['id']}: {exc}")
            answer = f"[ERROR: {exc}]"

        if auto_score:
            score = _auto_score(answer, q.get("expected_keywords", []))
            label = {2: "✓ Correct", 1: "~ Partial", 0: "✗ Wrong"}[score]
            print(f"         Answer: {answer[:120]}…")
            print(f"         Score : {score}/2  {label}\n")
        else:
            score = _manual_score(q["question"], answer, q.get("expected_answer", "N/A"))
            if score == -1:
                score = 0

        results.append({
            "id": q["id"],
            "question": q["question"],
            "difficulty": q.get("difficulty", ""),
            "category": q.get("category", ""),
            "answer": answer,
            "ungrounded_answer": answer,
            "expected_answer": q.get("expected_answer", ""),
            "score": score,
            "ungrounded_score": score,
            "grounding_status": "ungrounded",
            "avg_retrieval_similarity": 0.0,
            "retrieval_sources": [],
        })

    metrics = _compute_metrics(results)
    _print_summary(metrics, title="UNGROUNDED (MODE A) SUMMARY")
    return results, metrics


def run_grounded_evaluation(
    questions: Union[list, str, Path],
    top_k: int = 5,
    auto_score: bool = True,
) -> Tuple[list, dict]:
    """
    Mode B: Evaluate answers generated by RAG with Wikipedia context retrieval.
    """
    q_list = _load_questions_list(questions)
    pipeline = RAGPipeline()
    results = []
    skipped = 0

    print(f"\n{'='*60}")
    print(f"GROUNDED EVALUATION (Mode B) — {len(q_list)} questions")
    print(f"Mode: {'auto keyword-scoring' if auto_score else 'interactive manual scoring'}")
    print(f"{'='*60}\n")

    for i, q in enumerate(q_list, 1):
        print(f"[{i:02d}/{len(q_list)}] ({q['difficulty'].upper():10s}) {q['question']}")
        try:
            res = pipeline.query(q["question"], top_k=top_k)
            answer = res.get("answer", "")
        except Exception as exc:
            logger.error(f"Grounded pipeline error for question {q['id']}: {exc}")
            answer = f"[ERROR: {exc}]"
            res = {"answer": answer, "context_chunks": [], "grounding_status": "unsupported"}

        if auto_score:
            score = _auto_score(answer, q.get("expected_keywords", []))
            label = {2: "✓ Correct", 1: "~ Partial", 0: "✗ Wrong"}[score]
            print(f"         Answer: {answer[:120]}…")
            print(f"         Score : {score}/2  {label}\n")
        else:
            score = _manual_score(q["question"], answer, q.get("expected_answer", "N/A"))
            if score == -1:
                skipped += 1
                continue

        chunks = res.get("context_chunks", [])
        retrieval_sources = [c.get("source", "") for c in chunks]
        retrieval_sims = [c.get("similarity", 0.0) for c in chunks]
        avg_sim = round(sum(retrieval_sims) / len(retrieval_sims), 4) if retrieval_sims else 0.0

        results.append({
            "id": q["id"],
            "question": q["question"],
            "difficulty": q.get("difficulty", ""),
            "category": q.get("category", ""),
            "answer": answer,
            "grounded_answer": answer,
            "expected_answer": q.get("expected_answer", ""),
            "score": score,
            "grounded_score": score,
            "grounding_status": res.get("grounding_status", "grounded"),
            "retrieval_sources": retrieval_sources,
            "avg_retrieval_similarity": avg_sim,
            "context_chunks": chunks,
        })

    metrics = _compute_metrics(results)
    _print_summary(metrics, skipped=skipped, title="GROUNDED (MODE B) SUMMARY")
    return results, metrics


def run_full_dual_evaluation(
    questions_file: str = "evaluation/test_questions.json",
    auto_score: bool = True,
    top_k: int = 5,
    output_dir: Optional[Union[str, Path]] = None,
) -> dict:
    """
    Run full dual-track evaluation:
      - Mode A: Ungrounded (raw LLM)
      - Mode B: Grounded (RAG with Wikipedia passages)
    Saves results to both CSV and JSON.
    """
    q_list = _load_questions_list(questions_file)

    # 1. Run Ungrounded
    ungrounded_results, ungrounded_metrics = run_ungrounded_evaluation(
        q_list, top_k=top_k, auto_score=auto_score
    )

    # 2. Run Grounded
    grounded_results, grounded_metrics = run_grounded_evaluation(
        q_list, top_k=top_k, auto_score=auto_score
    )

    # 3. Combine dual results
    dual_results = []
    ungrounded_map = {r["id"]: r for r in ungrounded_results}
    for g in grounded_results:
        u = ungrounded_map.get(g["id"], {})
        combined = {
            "id": g["id"],
            "question": g["question"],
            "difficulty": g.get("difficulty", ""),
            "category": g.get("category", ""),
            "expected_answer": g.get("expected_answer", ""),
            "grounded_answer": g.get("grounded_answer", g.get("answer", "")),
            "ungrounded_answer": u.get("ungrounded_answer", u.get("answer", "")),
            "grounded_score": g.get("grounded_score", g.get("score", 0)),
            "ungrounded_score": u.get("ungrounded_score", u.get("score", 0)),
            "avg_retrieval_similarity": g.get("avg_retrieval_similarity", 0.0),
            "grounding_status": g.get("grounding_status", "grounded"),
            "retrieval_sources": g.get("retrieval_sources", []),
        }
        dual_results.append(combined)

    # 4. Summaries
    grounded_summary = compute_metrics_summary(dual_results, mode="grounded")
    ungrounded_summary = compute_metrics_summary(dual_results, mode="ungrounded")

    out_dir = Path(output_dir) if output_dir else EVAL_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_file = out_dir / f"dual_eval_{timestamp_str}.csv"
    json_file = out_dir / f"dual_eval_{timestamp_str}.json"
    latest_csv = out_dir / "dual_eval_latest.csv"
    latest_json = out_dir / "dual_eval_latest.json"

    export_to_csv(dual_results, str(csv_file))
    export_to_csv(dual_results, str(latest_csv))

    dual_output = {
        "timestamp": get_timestamp(),
        "config": {
            "auto_score": auto_score,
            "top_k": top_k,
            "questions_file": str(questions_file),
        },
        "grounded_metrics": grounded_metrics,
        "ungrounded_metrics": ungrounded_metrics,
        "summary": {
            "grounded": grounded_summary,
            "ungrounded": ungrounded_summary,
        },
        "results": dual_results,
    }

    export_to_json(dual_output, str(json_file))
    export_to_json(dual_output, str(latest_json))

    print(f"\nDual evaluation results saved to:\n  {csv_file}\n  {json_file}")
    return dual_output


# ---------------------------------------------------------------------------
# Backward-compatible run_evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    questions_file: str = "evaluation/test_questions.json",
    auto_score: bool = False,
    top_k: int = 5,
    output_file: Optional[str] = None,
) -> dict:
    """
    Run the full evaluation harness (Mode B grounded evaluation).
    Maintains full backward compatibility.
    """
    results, metrics = run_grounded_evaluation(
        questions=questions_file,
        top_k=top_k,
        auto_score=auto_score,
    )

    if output_file:
        out = {
            "timestamp": datetime.utcnow().isoformat(),
            "config": {
                "auto_score": auto_score,
                "top_k": top_k,
                "questions_file": questions_file,
            },
            "metrics": metrics,
            "results": results,
        }
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nResults saved to: {output_file}")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WikiTruth AI Evaluation Harness")
    parser.add_argument("--auto", action="store_true", help="Use automatic keyword scoring")
    parser.add_argument("--dual", action="store_true", help="Run dual-track evaluation (grounded + ungrounded)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--questions", type=str, default="evaluation/test_questions.json", help="Questions JSON path")
    parser.add_argument("--output", type=str, default=None, help="Output file or dir")

    args = parser.parse_args()
    if args.dual:
        run_full_dual_evaluation(
            questions_file=args.questions,
            auto_score=args.auto,
            top_k=args.top_k,
            output_dir=args.output,
        )
    else:
        run_evaluation(
            questions_file=args.questions,
            auto_score=args.auto,
            top_k=args.top_k,
            output_file=args.output,
        )
