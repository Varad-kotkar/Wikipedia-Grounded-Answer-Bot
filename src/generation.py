
"""
WikiTruth AI - LLM answer generation with grounding support.

Provider fallback:
1. Anthropic Claude, when configured
2. OpenAI, when configured
3. Local Flan-T5, without an API key

The local model is loaded lazily and runs on CPU.
"""

import logging
from typing import Optional, List, Dict

import torch
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

LOCAL_INPUT_TOKEN_LIMIT = 512
LOCAL_MAX_NEW_TOKENS = 180

_UNGROUNDED_PROMPT = """\
Answer the following question using your general knowledge.
Question: {question}
Answer:
"""


class Generator:
    """Generate grounded and ungrounded answers."""

    def __init__(self, provider: str = LLM_PROVIDER):
        self.provider = provider
        self._client = None
        self._local_model = None
        self._local_tokenizer = None
        self._setup()

    def _setup(self) -> None:
        if self.provider == "anthropic":
            if ANTHROPIC_API_KEY:
                self._setup_anthropic()
            elif OPENAI_API_KEY:
                logger.warning("Anthropic key missing; using OpenAI.")
                self.provider = "openai"
                self._setup_openai()
            else:
                logger.warning("No API key configured; using local Flan-T5.")
                self.provider = "local"
                self._setup_local()

        elif self.provider == "openai":
            if OPENAI_API_KEY:
                self._setup_openai()
            else:
                logger.warning("OpenAI key missing; using local Flan-T5.")
                self.provider = "local"
                self._setup_local()

        elif self.provider == "local":
            self._setup_local()

        else:
            logger.warning("Unknown provider '%s'; using local Flan-T5.", self.provider)
            self.provider = "local"
            self._setup_local()

    def _setup_anthropic(self) -> None:
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            logger.info("Using Anthropic model: %s", ANTHROPIC_MODEL)
        except ImportError:
            logger.exception("Anthropic package unavailable.")
            self.provider = "local"
            self._setup_local()

    def _setup_openai(self) -> None:
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("Using OpenAI model: %s", OPENAI_CHAT_MODEL)
        except ImportError:
            logger.exception("OpenAI package unavailable.")
            self.provider = "local"
            self._setup_local()

    def _setup_local(self) -> None:
        """Do not download the model until a local answer is requested."""
        self.provider = "local"
        logger.info("Local model selected: %s", LOCAL_LLM_MODEL)

    def _load_local_model(self) -> None:
        """Load Flan-T5 explicitly in CPU float32 mode."""

        if self._local_model is not None:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError as exc:
            raise RuntimeError(
                "Transformers is unavailable. Check requirements.txt."
            ) from exc

        logger.info("Loading local model: %s", LOCAL_LLM_MODEL)

        self._local_tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_LLM_MODEL
        )

        self._local_model = AutoModelForSeq2SeqLM.from_pretrained(
            LOCAL_LLM_MODEL,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
        )

        self._local_model.to("cpu")
        self._local_model.eval()

        logger.info("Local model loaded successfully on CPU.")

    @staticmethod
    def _build_local_grounded_prompt(
        question: str,
        chunks: List[Dict],
        prompt_style: str,
    ) -> str:
        """Build a compact evidence-first prompt for Flan-T5."""

        if prompt_style == "few_shot":
            instruction = """\
Answer the question using only the evidence.
If the evidence does not contain the answer, say:
"The available Wikipedia sources do not contain enough information."

Example 1:
Evidence: [Source 1] A neural network is a computing system inspired by biological neural networks.
Question: What is a neural network?
Answer: A neural network is a computing system inspired by biological neural networks. [Source 1]

Example 2:
Evidence: [Source 1] Supervised learning uses labelled training data.
Question: Does supervised learning use labelled data?
Answer: Yes. Supervised learning uses labelled training data. [Source 1]

Now answer the actual question.
"""
        else:
            instruction = """\
Answer the question using only the Wikipedia evidence.
Do not use outside knowledge or invent facts.
Cite factual claims using [Source N].
If the evidence is insufficient, say:
"The available Wikipedia sources do not contain enough information."
"""

        parts = [
            instruction,
            f"Question: {question}",
            "Wikipedia evidence:",
        ]

        for index, chunk in enumerate(chunks, start=1):
            source = str(chunk.get("source", "Wikipedia")).strip()
            content = str(chunk.get("text", "")).strip()

            if content:
                parts.append(
                    f"[Source {index}: {source}] {content}"
                )

        parts.append("Answer:")
        return "\n\n".join(parts)

    def _call_local(self, prompt: str) -> str:
        """Generate text directly with model.generate()."""

        self._load_local_model()

        encoded = self._local_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=LOCAL_INPUT_TOKEN_LIMIT,
        )

        encoded = {
            key: value.to("cpu")
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            output_ids = self._local_model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                max_new_tokens=LOCAL_MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=2,
                early_stopping=True,
            )

        answer = self._local_tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        ).strip()

        if not answer:
            raise RuntimeError("Flan-T5 returned an empty answer.")

        return answer

    def _call_llm(self, prompt: str) -> str:
        """Call the selected language model."""

        if self.provider == "local":
            return self._call_local(prompt)

        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()

        if self.provider == "openai":
            response = self._client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return (response.choices[0].message.content or "").strip()

        raise RuntimeError(f"Unknown provider: {self.provider}")

    def generate(
        self,
        question: str,
        chunks: List[Dict],
        prompt_style: str = "zero_shot",
    ) -> dict:
        """Generate an answer using retrieved evidence."""

        if not chunks:
            return {
                "question": question,
                "answer": (
                    "I cannot answer this question based on the available "
                    "Wikipedia sources. No relevant passages were retrieved."
                ),
                "context_chunks": [],
                "provider": self.provider,
                "model": self._get_model_name(),
                "grounding_status": GroundingStatus.UNSUPPORTED.value,
                "prompt_style": prompt_style,
                "generation_error": False,
            }

        if self.provider == "local":
            prompt = self._build_local_grounded_prompt(
                question, chunks, prompt_style
            )
        else:
            _, prompt = build_prompt(
                question, chunks, style=prompt_style
            )

        try:
            answer = self._call_llm(prompt).strip()

            if not answer:
                raise RuntimeError("The model returned an empty answer.")

            status = detect_grounding_status(answer, chunks)
            generation_error = False

        except Exception as exc:
            logger.exception("Grounded generation failed.")
            answer = f"[Generation error: {type(exc).__name__}: {exc}]"
            status = GroundingStatus.UNSUPPORTED
            generation_error = True

        return {
            "question": question,
            "answer": answer,
            "context_chunks": chunks,
            "provider": self.provider,
            "model": self._get_model_name(),
            "grounding_status": status.value,
            "prompt_style": prompt_style,
            "generation_error": generation_error,
        }

    def generate_ungrounded(self, question: str) -> dict:
        """Generate an answer without retrieval."""

        prompt = _UNGROUNDED_PROMPT.format(question=question)

        try:
            answer = self._call_llm(prompt).strip()
            if not answer:
                raise RuntimeError("The model returned an empty answer.")
            generation_error = False

        except Exception as exc:
            logger.exception("Ungrounded generation failed.")
            answer = f"[Generation error: {type(exc).__name__}: {exc}]"
            generation_error = True

        return {
            "question": question,
            "answer": answer,
            "context_chunks": [],
            "provider": self.provider,
            "model": self._get_model_name(),
            "grounding_status": "ungrounded",
            "prompt_style": "none",
            "generation_error": generation_error,
        }

    def _get_model_name(self) -> str:
        if self.provider == "anthropic":
            return ANTHROPIC_MODEL
        if self.provider == "openai":
            return OPENAI_CHAT_MODEL
        if self.provider == "local":
            return LOCAL_LLM_MODEL
        return "unknown"


_generator: Optional[Generator] = None


def get_generator() -> Generator:
    """Return the shared generator instance."""
    global _generator
    if _generator is None:
        _generator = Generator()
    return _generator


def reset_generator() -> None:
    """Reset the shared generator instance."""
    global _generator
    _generator = None
