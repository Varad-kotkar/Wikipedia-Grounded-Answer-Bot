"""
src/generation.py — WikiTruth AI
LLM-based answer generation with grounding prompt support.

Provider fallback chain:
  1. Anthropic Claude (if ANTHROPIC_API_KEY is set)
  2. OpenAI GPT (if OPENAI_API_KEY is set)
  3. Local Flan-T5 (no API key required)

Supports zero-shot grounding, few-shot grounding, and ungrounded comparison.
"""

import logging
from typing import Optional, List, Dict

from .config import (
    LLM_PROVIDER,
    ANTHROPIC_MODEL,
    OPENAI_CHAT_MODEL,
    LOCAL_LLM_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
)
from .grounding import build_prompt, detect_grounding_status, GroundingStatus

logger = logging.getLogger(__name__)

_UNGROUNDED_PROMPT = """\
Answer the following question to the best of your ability.

Question: {question}

Answer:"""

# Flan-T5 models commonly have a 512-token encoder input limit.
# Keep this conservative and preserve the question before adding evidence.
LOCAL_INPUT_TOKEN_LIMIT = 512


class Generator:
    """WikiTruth AI generator with hosted-provider and local-model fallback."""

    def __init__(self, provider: str = LLM_PROVIDER):
        self.provider = provider
        self._client = None
        self._local_pipeline = None
        self._setup()

    def _setup(self) -> None:
        if self.provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                if OPENAI_API_KEY:
                    logger.warning(
                        "ANTHROPIC_API_KEY not set; falling back to OpenAI."
                    )
                    self.provider = "openai"
                    self._setup_openai()
                else:
                    logger.warning(
                        "No API keys found; falling back to local Flan-T5."
                    )
                    self.provider = "local"
                    self._setup_local()
            else:
                self._setup_anthropic()

        elif self.provider == "openai":
            if not OPENAI_API_KEY:
                logger.warning(
                    "OPENAI_API_KEY not set; falling back to local Flan-T5."
                )
                self.provider = "local"
                self._setup_local()
            else:
                self._setup_openai()

        elif self.provider == "local":
            self._setup_local()

        else:
            logger.warning(
                "Unknown provider '%s'; falling back to local Flan-T5.",
                self.provider,
            )
            self.provider = "local"
            self._setup_local()

    def _setup_anthropic(self) -> None:
        try:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=ANTHROPIC_API_KEY
            )
            logger.info(
                "Generator: Anthropic Claude (%s)",
                ANTHROPIC_MODEL,
            )

        except ImportError:
            logger.exception(
                "anthropic package unavailable; falling back to local Flan-T5."
            )
            self.provider = "local"
            self._setup_local()

    def _setup_openai(self) -> None:
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info(
                "Generator: OpenAI (%s)",
                OPENAI_CHAT_MODEL,
            )

        except ImportError:
            logger.exception(
                "openai package unavailable; falling back to local Flan-T5."
            )
            self.provider = "local"
            self._setup_local()

    def _setup_local(self) -> None:
        """Defer loading Flan-T5 until the first generation request."""
        self.provider = "local"
        logger.info(
            "Generator: Local Flan-T5 (%s)",
            LOCAL_LLM_MODEL,
        )

    def _get_local_pipeline(self):
        """Load the Flan-T5 text2text-generation pipeline once."""
        if self._local_pipeline is None:
            try:
                from transformers import pipeline as hf_pipeline

            except ImportError as exc:
                raise ImportError(
                    "transformers is not installed. "
                    "Install requirements.txt."
                ) from exc

            logger.info(
                "Loading local Flan-T5 model: %s",
                LOCAL_LLM_MODEL,
            )

            # Flan-T5 is an encoder-decoder model.
            # text2text-generation is the correct pipeline
            # when using Transformers 4.x.
            self._local_pipeline = hf_pipeline(
                "text2text-generation",
                model=LOCAL_LLM_MODEL,
            )

            logger.info(
                "Local Flan-T5 model loaded successfully."
            )

        return self._local_pipeline

    def _prepare_local_input(self, prompt: str) -> str:
        """
        Token-truncate local inputs safely.

        The question is deliberately placed first in grounded prompts
        so truncation cannot discard it after long retrieved passages.
        """
        pipe = self._get_local_pipeline()
        tokenizer = pipe.tokenizer

        encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=LOCAL_INPUT_TOKEN_LIMIT,
            return_tensors=None,
        )

        input_ids = encoded["input_ids"]

        return tokenizer.decode(
            input_ids,
            skip_special_tokens=True,
        )

    @staticmethod
    def _build_local_grounded_prompt(
        question: str,
        chunks: List[Dict],
        prompt_style: str,
    ) -> str:
        """
        Build a compact Flan-T5 prompt with the question first.

        This avoids sending the very long hosted-model prompt,
        including lengthy few-shot examples, to the local model.
        """

        if prompt_style == "few_shot":
            instruction = (
                "Answer only from the evidence. "
                "If the evidence does not answer the question, "
                "say that the Wikipedia sources do not contain "
                "enough information. Cite evidence as [Source N]."
            )

        else:
            instruction = (
                "Use only the Wikipedia evidence below. "
                "Answer the question directly in clear sentences "
                "and cite factual claims as [Source N]. "
                "Do not add facts from memory. "
                "If the evidence does not answer the question, "
                "say that the sources do not contain enough information."
            )

        # Question first: truncation from the right preserves it.
        parts = [
            instruction,
            f"Question: {question}",
            "Wikipedia evidence:",
        ]

        for index, chunk in enumerate(chunks, start=1):
            source = str(
                chunk.get("source", "Wikipedia")
            )

            text = str(
                chunk.get("text", "")
            ).strip()

            if text:
                parts.append(
                    f"[Source {index}: {source}] {text}"
                )

        parts.append("Answer:")

        return "\n".join(parts)

    def _call_llm(self, prompt: str) -> str:
        """Send a prompt to the selected provider and return generated text."""

        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            return response.content[0].text.strip()

        if self.provider == "openai":
            response = self._client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

            content = response.choices[0].message.content

            return (content or "").strip()

        if self.provider == "local":
            pipe = self._get_local_pipeline()

            safe_prompt = self._prepare_local_input(prompt)

            outputs = pipe(
                safe_prompt,
                max_new_tokens=MAX_TOKENS,
                do_sample=False,
            )

            if not outputs or "generated_text" not in outputs[0]:
                raise RuntimeError(
                    "Flan-T5 returned an empty generation result."
                )

            return str(
                outputs[0]["generated_text"]
            ).strip()

        raise RuntimeError(
            f"Unknown provider: {self.provider}"
        )

    def generate(
        self,
        question: str,
        chunks: List[Dict],
        prompt_style: str = "zero_shot",
    ) -> dict:
        """Generate an answer grounded in retrieved Wikipedia passages."""

        if not chunks:
            return {
                "question": question,
                "answer": (
                    "I cannot answer this question based on the "
                    "available Wikipedia sources. "
                    "No relevant passages were retrieved."
                ),
                "context_chunks": [],
                "provider": self.provider,
                "model": self._get_model_name(),
                "grounding_status": GroundingStatus.UNSUPPORTED.value,
                "prompt_style": prompt_style,
            }

        if self.provider == "local":
            # Use a compact question-first prompt for Flan-T5.
            full_prompt = self._build_local_grounded_prompt(
                question,
                chunks,
                prompt_style,
            )

        else:
            # Hosted models use the original grounding prompt.
            _, full_prompt = build_prompt(
                question,
                chunks,
                style=prompt_style,
            )

        try:
            answer = self._call_llm(
                full_prompt
            ).strip()

            if not answer:
                raise RuntimeError(
                    "The language model returned an empty answer."
                )

        except Exception as exc:
            logger.exception(
                "Grounded LLM generation failed."
            )

            answer = f"[Generation error: {exc}]"

        status = detect_grounding_status(
            answer,
            chunks,
        )

        return {
            "question": question,
            "answer": answer,
            "context_chunks": chunks,
            "provider": self.provider,
            "model": self._get_model_name(),
            "grounding_status": status.value,
            "prompt_style": prompt_style,
        }

    def generate_ungrounded(
        self,
        question: str,
    ) -> dict:
        """Generate without retrieval for grounded-vs-ungrounded comparison."""

        prompt = _UNGROUNDED_PROMPT.format(
            question=question
        )

        try:
            answer = self._call_llm(
                prompt
            ).strip()

            if not answer:
                raise RuntimeError(
                    "The language model returned an empty answer."
                )

        except Exception as exc:
            logger.exception(
                "Ungrounded LLM generation failed."
            )

            answer = f"[Generation error: {exc}]"

        return {
            "question": question,
            "answer": answer,
            "context_chunks": [],
            "provider": self.provider,
            "model": self._get_model_name(),
            "grounding_status": "ungrounded",
            "prompt_style": "none",
        }

    def _get_model_name(self) -> str:
        """Return the configured model identifier."""

        if self.provider == "anthropic":
            return ANTHROPIC_MODEL

        if self.provider == "openai":
            return OPENAI_CHAT_MODEL

        if self.provider == "local":
            return LOCAL_LLM_MODEL

        return "unknown"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_generator: Optional[Generator] = None


def get_generator() -> Generator:
    """Return the shared Generator instance (lazy-initialised)."""

    global _generator

    if _generator is None:
        _generator = Generator()

    return _generator


def reset_generator() -> None:
    """Reset the generator singleton (useful for testing)."""

    global _generator
    _generator = None
