"""
src/generation.py — WikiTruth AI
LLM-based answer generation with grounding prompt support.

Provider fallback chain (auto-detected):
  1. Anthropic Claude (if ANTHROPIC_API_KEY set)
  2. OpenAI GPT (if OPENAI_API_KEY set)
  3. Local Flan-T5 (always available, no API key needed)

Supports:
  - Zero-shot grounding (default)
  - Few-shot grounding
  - Ungrounded mode (for comparison)
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


# Ungrounded prompt — raw LLM, no context
_UNGROUNDED_PROMPT = """\
Answer the following question to the best of your ability.

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class Generator:
    """
    WikiTruth AI answer generator.
    
    Wraps multiple LLM backends with automatic fallback.
    Provider resolution (auto):
        1. ANTHROPIC_API_KEY set → Anthropic Claude
        2. OPENAI_API_KEY set    → OpenAI GPT
        3. Neither set           → Local Flan-T5 (transformers)
    """

    def __init__(self, provider: str = LLM_PROVIDER):
        self.provider = provider
        self._client = None
        self._local_pipeline = None
        self._setup()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        if self.provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                if OPENAI_API_KEY:
                    logger.warning("ANTHROPIC_API_KEY not set — falling back to OpenAI.")
                    self.provider = "openai"
                    self._setup_openai()
                else:
                    logger.warning("No API keys found — falling back to local Flan-T5.")
                    self.provider = "local"
                    self._setup_local()
            else:
                self._setup_anthropic()

        elif self.provider == "openai":
            if not OPENAI_API_KEY:
                logger.warning("OPENAI_API_KEY not set — falling back to local Flan-T5.")
                self.provider = "local"
                self._setup_local()
            else:
                self._setup_openai()

        elif self.provider == "local":
            self._setup_local()

        else:
            logger.warning(f"Unknown provider '{self.provider}' — falling back to local Flan-T5.")
            self.provider = "local"
            self._setup_local()

    def _setup_anthropic(self) -> None:
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            logger.info(f"Generator: Anthropic Claude ({ANTHROPIC_MODEL})")
        except ImportError:
            logger.warning("anthropic not installed — falling back to local Flan-T5")
            self.provider = "local"
            self._setup_local()

    def _setup_openai(self) -> None:
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info(f"Generator: OpenAI ({OPENAI_CHAT_MODEL})")
        except ImportError:
            logger.warning("openai not installed — falling back to local Flan-T5")
            self.provider = "local"
            self._setup_local()

    def _setup_local(self) -> None:
        """Initialize the local Flan-T5 pipeline. Loads model on first call."""
        logger.info(f"Generator: Local Flan-T5 ({LOCAL_LLM_MODEL}) — no API key needed")
        # Defer actual model loading until first generate() call
        self.provider = "local"

    def _get_local_pipeline(self):
        """Lazy-load the local pipeline."""
        if self._local_pipeline is None:
            try:
                from transformers import pipeline as hf_pipeline
                logger.info(f"Loading local model: {LOCAL_LLM_MODEL} (first run may download ~250MB)")
                self._local_pipeline = hf_pipeline(
                    "text2text-generation",
                    model=LOCAL_LLM_MODEL,
                    max_new_tokens=MAX_TOKENS,
                    do_sample=False,
                )
                logger.info("Local model loaded successfully")
            except ImportError:
                raise ImportError(
                    "transformers not installed. Run: pip install transformers"
                )
        return self._local_pipeline

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Send prompt to configured LLM and return response text."""
        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        elif self.provider == "openai":
            response = self._client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content

        elif self.provider == "local":
            pipe = self._get_local_pipeline()
            # Flan-T5 works best with shorter prompts; extract the core Q&A
            # Truncate prompt to avoid exceeding model context window
            truncated_prompt = prompt[:2048] if len(prompt) > 2048 else prompt
            outputs = pipe(truncated_prompt)
            return outputs[0]["generated_text"]

        raise RuntimeError(f"Unknown provider: {self.provider}")

    def generate(self, question: str, chunks: List[Dict], prompt_style: str = "zero_shot") -> dict:
        """
        Generate a grounded answer using retrieved Wikipedia chunks.

        Args:
            question: user's question
            chunks: retrieved context chunks
            prompt_style: 'zero_shot' or 'few_shot'

        Returns:
            dict with keys: question, answer, context_chunks, provider, model,
                            grounding_status, prompt_style
        """
        # Validate context
        if not chunks:
            return {
                "question": question,
                "answer": "I cannot answer this question based on the available Wikipedia sources. No relevant passages were retrieved.",
                "context_chunks": [],
                "provider": self.provider,
                "model": self._get_model_name(),
                "grounding_status": GroundingStatus.UNSUPPORTED.value,
                "prompt_style": prompt_style,
            }

        # Build grounding prompt
        _, full_prompt = build_prompt(question, chunks, style=prompt_style)

        try:
            answer = self._call_llm(full_prompt).strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = "I cannot answer this question based on the available Wikipedia sources."

        # Detect grounding status
        status = detect_grounding_status(answer, chunks)

        return {
            "question": question,
            "answer": answer,
            "context_chunks": chunks,
            "provider": self.provider,
            "model": self._get_model_name(),
            "grounding_status": status.value,
            "prompt_style": prompt_style,
        }

    def generate_ungrounded(self, question: str) -> dict:
        """
        Generate an answer WITHOUT retrieval (raw LLM knowledge).
        Used for Mode A comparison in evaluation.

        Returns:
            dict with keys: question, answer, provider, model, grounding_status
        """
        prompt = _UNGROUNDED_PROMPT.format(question=question)
        
        try:
            answer = self._call_llm(prompt).strip()
        except Exception as e:
            logger.error(f"Ungrounded generation failed: {e}")
            answer = f"[Generation error: {e}]"

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
        """Return current model identifier string."""
        if self.provider == "anthropic":
            return ANTHROPIC_MODEL
        elif self.provider == "openai":
            return OPENAI_CHAT_MODEL
        elif self.provider == "local":
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
