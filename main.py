#!/usr/bin/env python3
"""
main.py — CLI entry point for the Wikipedia RAG system.

Usage:
  python main.py ingest                           # download, chunk, embed, store
  python main.py ingest --reset                   # wipe DB and re-ingest
  python main.py query "What is deep learning?"   # ask a question
  python main.py query "..." --top-k 3            # retrieve 3 chunks
  python main.py evaluate                         # interactive evaluation
  python main.py evaluate --auto                  # auto keyword-scoring
"""

import argparse
import logging
import sys

# Pretty log format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> None:
    from pipeline import RAGPipeline
    pipeline = RAGPipeline()
    pipeline.ingest(reset=args.reset, force_rechunk=args.force_rechunk)


def cmd_query(args: argparse.Namespace) -> None:
    from pipeline import RAGPipeline
    pipeline = RAGPipeline()
    result = pipeline.query(args.question, top_k=args.top_k)
    pipeline.display_result(result, verbose=not args.quiet)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from evaluation.evaluate import run_evaluation
    run_evaluation(
        questions_file=args.questions_file,
        auto_score=args.auto,
        top_k=args.top_k,
        output_file=args.output,
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Wikipedia RAG System — AI/ML Knowledge Base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # ── ingest ────────────────────────────────────────────────────────────
    p_ingest = sub.add_parser(
        "ingest",
        help="Download Wikipedia articles, chunk, embed, and store in ChromaDB.",
    )
    p_ingest.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the vector store before re-inserting (forces full re-embedding).",
    )
    p_ingest.add_argument(
        "--force-rechunk",
        action="store_true",
        help="Re-chunk raw files even if chunks already exist on disk.",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # ── query ─────────────────────────────────────────────────────────────
    p_query = sub.add_parser(
        "query",
        help="Ask a question and get a grounded answer.",
    )
    p_query.add_argument("question", type=str, help="Question to answer.")
    p_query.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="K",
        help="Number of chunks to retrieve (default: 5).",
    )
    p_query.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the answer, not the retrieved context.",
    )
    p_query.set_defaults(func=cmd_query)

    # ── evaluate ──────────────────────────────────────────────────────────
    p_eval = sub.add_parser(
        "evaluate",
        help="Run the evaluation harness on the test question set.",
    )
    p_eval.add_argument(
        "--questions-file",
        type=str,
        default="evaluation/test_questions.json",
        metavar="FILE",
        help="Path to the test-questions JSON file.",
    )
    p_eval.add_argument(
        "--auto",
        action="store_true",
        help="Use automatic keyword-based scoring instead of interactive scoring.",
    )
    p_eval.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="K",
        help="Chunks to retrieve per question (default: 5).",
    )
    p_eval.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Save evaluation results to a JSON file.",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
