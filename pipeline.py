
"""
pipeline.py — WikiTruth AI

RAGPipeline orchestrates ingestion (download → chunk → embed → store)
and querying (embed → retrieve → generate).

Supports:
  - Grounded mode (Mode B): Wikipedia retrieval + LLM generation
  - Ungrounded mode (Mode A): Direct LLM without retrieval
  - Prompt style comparison: zero_shot vs few_shot
  - Ingestion progress callbacks for Streamlit UI
"""

import logging
from tqdm import tqdm

from src.config import TOPICS
from src.data_collection import collect_all
from src.chunking import process_all, load_chunks
from src.embeddings import get_provider
from src.vector_store import get_store
from src.retrieval import get_retriever
from src.generation import get_generator

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    WikiTruth AI — End-to-end RAG pipeline with separate ingest and query paths.
    """

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        topics: list[str] | None = None,
        reset: bool = False,
        force_rechunk: bool = False,
        progress_callback=None,
    ) -> None:
        """
        Run the full ingestion pipeline:
          1. Download Wikipedia articles
          2. Chunk them into token-bounded segments
          3. Embed every chunk
          4. Persist embeddings + metadata to ChromaDB

        Args:
            topics: override the default TOPICS list.
            reset: wipe the vector store before inserting.
            force_rechunk: re-chunk even if chunks already exist on disk.
            progress_callback: optional function receiving (progress, message).
        """

        def report_progress(progress: float, message: str) -> None:
            if progress_callback:
                progress_callback(progress, message)

        logger.info("=" * 60)
        logger.info("WikiTruth AI — INGESTION PIPELINE START")
        logger.info("=" * 60)

        # Step 1: collect raw Wikipedia text
        logger.info("Step 1/4 — Downloading Wikipedia articles…")
        report_progress(0.10, "Downloading Wikipedia articles...")
        collect_all(topics=topics or TOPICS)

        # Step 2: chunk into token segments
        logger.info("Step 2/4 — Chunking articles…")
        report_progress(0.35, "Processing and chunking articles...")
        chunks = process_all(force=force_rechunk)
        logger.info(f"  {len(chunks)} chunks ready")

        if not chunks:
            raise RuntimeError(
                "No text chunks were created. Check Wikipedia downloads "
                "and the data processing pipeline."
            )

        # Step 3: initialise / optionally reset vector store
        logger.info("Step 3/4 — Initialising vector store…")
        report_progress(0.50, "Initializing vector database...")

        store = get_store()

        if reset:
            store.reset()
            logger.info("  Vector store reset.")

        if store.is_populated():
            count = store.count()

            logger.info(
                f"  Vector store already contains {count} documents. "
                "Skipping embedding step (use --reset to re-embed)."
            )

            report_progress(
                1.0,
                f"Knowledge base already contains {count} indexed documents."
            )

            logger.info("=" * 60)
            logger.info("INGESTION COMPLETE (skipped re-embedding)")
            logger.info("=" * 60)
            return

        # Step 4: embed + store
        logger.info("Step 4/4 — Generating embeddings and storing…")
        report_progress(0.60, "Generating embeddings...")

        embedder = get_provider()
        texts = [chunk["text"] for chunk in chunks]

        batch_size = 64
        all_embeddings: list[list[float]] = []

        total_batches = max(
            1,
            (len(texts) + batch_size - 1) // batch_size
        )

        for batch_num, i in enumerate(
            tqdm(
                range(0, len(texts), batch_size),
                desc="Embedding",
                unit="batch"
            )
        ):
            batch = texts[i:i + batch_size]

            all_embeddings.extend(
                embedder.embed_batch(batch)
            )

            progress = 0.60 + (
                0.35 * (batch_num + 1) / total_batches
            )

            report_progress(
                progress,
                f"Generating embeddings: batch "
                f"{batch_num + 1}/{total_batches}"
            )

        # Store embeddings and metadata in ChromaDB
        store.add_chunks(chunks, all_embeddings)

        document_count = store.count()

        report_progress(
            1.0,
            f"Knowledge base ready: {document_count} documents indexed."
        )

        logger.info("=" * 60)
        logger.info(
            f"INGESTION COMPLETE — {document_count} documents in vector store"
        )
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Grounded querying (Mode B)
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int = 5,
        prompt_style: str = "zero_shot"
    ) -> dict:
        """
        Mode B: Run the full grounded query pipeline.
        embed question → retrieve top-k chunks → LLM generate grounded answer.

        Returns a result dict with keys:
            question, answer, context_chunks, provider, model,
            grounding_status, prompt_style
        """

        retriever = get_retriever()
        generator = get_generator()

        chunks = retriever.retrieve(question, k=top_k)

        result = generator.generate(
            question,
            chunks,
            prompt_style=prompt_style
        )

        return result

    # ------------------------------------------------------------------
    # Ungrounded querying (Mode A)
    # ------------------------------------------------------------------

    def query_ungrounded(self, question: str) -> dict:
        """
        Mode A: Answer question using LLM WITHOUT Wikipedia retrieval.
        Used for grounded vs ungrounded comparison (U5 evaluation).

        Returns:
            dict with keys: question, answer, provider, model,
            grounding_status
        """

        generator = get_generator()

        return generator.generate_ungrounded(question)

    # ------------------------------------------------------------------
    # Prompt style comparison
    # ------------------------------------------------------------------

    def query_with_prompt_style(
        self,
        question: str,
        prompt_style: str = "zero_shot",
        top_k: int = 5
    ) -> dict:
        """
        Query with a specific prompting style.

        Args:
            question: natural-language question
            prompt_style: 'zero_shot' or 'few_shot'
            top_k: number of chunks to retrieve

        Returns:
            result dict (same as query())
        """

        return self.query(
            question,
            top_k=top_k,
            prompt_style=prompt_style
        )

    # ------------------------------------------------------------------
    # Comparison helper
    # ------------------------------------------------------------------

    def compare_grounded_ungrounded(
        self,
        question: str,
        top_k: int = 5
    ) -> dict:
        """
        Run both grounded and ungrounded modes and return both results.

        Returns:
            {
                'question': str,
                'grounded': dict,
                'ungrounded': dict,
            }
        """

        grounded = self.query(
            question,
            top_k=top_k
        )

        ungrounded = self.query_ungrounded(question)

        return {
            "question": question,
            "grounded": grounded,
            "ungrounded": ungrounded,
        }

    def compare_prompt_styles(
        self,
        question: str,
        top_k: int = 5
    ) -> dict:
        """
        Compare zero-shot vs few-shot grounding prompts for the same question.

        Returns:
            {
                'question': str,
                'zero_shot': dict,
                'few_shot': dict,
            }
        """

        zero_shot = self.query(
            question,
            top_k=top_k,
            prompt_style="zero_shot"
        )

        few_shot = self.query(
            question,
            top_k=top_k,
            prompt_style="few_shot"
        )

        return {
            "question": question,
            "zero_shot": zero_shot,
            "few_shot": few_shot,
        }

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    @staticmethod
    def display_result(
        result: dict,
        verbose: bool = True
    ) -> None:
        """Pretty-print a query result to stdout."""

        bar = "=" * 60
        thin = "-" * 60

        print(f"\n{bar}")
        print(f"WikiTruth AI — QUESTION: {result['question']}")
        print(bar)

        print(
            f"\nANSWER [{result.get('model', result.get('provider', '?'))}]:"
        )

        print(result["answer"])

        if verbose:
            print(f"\n{thin}")
            print("RETRIEVED CONTEXT:")

            for i, chunk in enumerate(
                result.get("context_chunks", []),
                1
            ):
                sim = chunk.get("similarity", "?")

                print(
                    f"\n  [{i}] {chunk['source']} "
                    f"(similarity: {sim})\n"
                    f"      {chunk['text'][:250]}…"
                )

        print(f"{bar}\n")
