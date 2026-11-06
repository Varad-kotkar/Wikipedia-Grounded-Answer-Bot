"""
WikiTruth AI generation.

Grounded RAG prefers the configured LLM (or local Flan-T5). If text generation
fails, it returns a concise extractive response made only from retrieved
Wikipedia passages. The retrieved passages are always returned to the UI.

Ungrounded mode is separate and never accesses ChromaDB or retrieved context.
"""

import logging
import re
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

from .grounding import (
    build_prompt,
    detect_grounding_status,
    GroundingStatus,
)

logger = logging.getLogger(__name__)

LOCAL_INPUT_TOKEN_LIMIT = 512
LOCAL_MAX_NEW_TOKENS = 160


class Generator:

    def __init__(self, provider: str = LLM_PROVIDER):
        self.provider = provider
        self._client = None
        self._local_model = None
        self._local_tokenizer = None
        self._local_load_error = None

        self._setup()

    def _setup(self):

        if self.provider == "anthropic" and ANTHROPIC_API_KEY:
            self._setup_anthropic()

        elif self.provider in ("anthropic", "openai") and OPENAI_API_KEY:
            self.provider = "openai"
            self._setup_openai()

        else:
            self.provider = "local"
            logger.info("Using local Flan-T5 fallback.")

    def _setup_anthropic(self):

        try:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=ANTHROPIC_API_KEY
            )

        except Exception as exc:
            logger.warning(
                "Anthropic setup failed; using local model: %s",
                exc
            )
            self.provider = "local"

    def _setup_openai(self):

        try:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=OPENAI_API_KEY
            )

        except Exception as exc:
            logger.warning(
                "OpenAI setup failed; using local model: %s",
                exc
            )
            self.provider = "local"

    def _load_local_model(self):

        if self._local_model is not None:
            return

        if self._local_load_error:
            raise RuntimeError(self._local_load_error)

        try:
            import torch

            from transformers import (
                AutoTokenizer,
                AutoModelForSeq2SeqLM
            )

            self._local_tokenizer = AutoTokenizer.from_pretrained(
                LOCAL_LLM_MODEL
            )

            self._local_model = AutoModelForSeq2SeqLM.from_pretrained(
                LOCAL_LLM_MODEL,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
            )

            self._local_model.eval()

        except Exception as exc:

            self._local_load_error = (
                f"{type(exc).__name__}: {exc}"
            )

            raise RuntimeError(
                self._local_load_error
            ) from exc

    def _call_local(self, prompt: str) -> str:

        import torch

        self._load_local_model()

        encoded = self._local_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=LOCAL_INPUT_TOKEN_LIMIT,
        )

        with torch.inference_mode():

            output_ids = self._local_model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                max_new_tokens=LOCAL_MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=2,
            )

        answer = self._local_tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        ).strip()

        if not answer:
            raise RuntimeError(
                "Flan-T5 returned an empty answer."
            )

        return answer

    def _call_llm(self, prompt: str) -> str:

        if self.provider == "local":
            return self._call_local(prompt)

        if self.provider == "anthropic":

            response = self._client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            return response.content[0].text.strip()

        if self.provider == "openai":

            response = self._client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

            return (
                response.choices[0].message.content or ""
            ).strip()

        raise RuntimeError(
            f"Unsupported provider: {self.provider}"
        )

    @staticmethod
    def _extractive_grounded_answer(
        question: str,
        chunks: List[Dict]
    ) -> str:

        """
        Fallback response based only on sentences
        in retrieved Wikipedia evidence.
        """

        stop = {
            "what", "which", "when", "where", "who",
            "does", "from", "with", "that", "this",
            "the", "and", "for", "are", "was",
            "how", "why", "about", "into", "its",
            "their",
        }

        terms = {
            word.lower()
            for word in re.findall(
                r"[a-zA-Z][a-zA-Z0-9-]{2,}",
                question
            )
            if word.lower() not in stop
        }

        scored = []

        for index, chunk in enumerate(chunks[:5], 1):

            body = re.sub(
                r"\s+",
                " ",
                str(chunk.get("text", ""))
            ).strip()

            sentences = re.split(
                r"(?<=[.!?])\s+",
                body
            )

            for sentence in sentences:

                sentence = sentence.strip()

                if len(sentence) < 35:
                    continue

                words = {
                    word.lower()
                    for word in re.findall(
                        r"[a-zA-Z][a-zA-Z0-9-]{2,}",
                        sentence
                    )
                }

                overlap = len(words & terms)

                if overlap:

                    scored.append((
                        overlap,
                        sentence,
                        index,
                        chunk.get("source", "Wikipedia")
                    ))

        scored.sort(
            key=lambda item: item[0],
            reverse=True
        )

        selected = []
        seen = set()

        for _, sentence, index, source in scored:

            key = sentence.lower()

            if key in seen:
                continue

            seen.add(key)

            selected.append(
                f"{sentence} [Source {index}: {source}]"
            )

            if len(selected) >= 3:
                break

        if selected:

            return (
                "Based on the retrieved Wikipedia passages:\n\n"
                + "\n\n".join(selected)
            )

        first = str(
            chunks[0].get("text", "")
        ).strip()

        if first:

            first = re.sub(
                r"\s+",
                " ",
                first
            )

            return (
                f"Retrieved evidence [Source 1: "
                f"{chunks[0].get('source', 'Wikipedia')}]: "
                f"{first[:650]}"
            )

        return (
            "I cannot answer this question based on "
            "the available Wikipedia sources."
        )

    @staticmethod
    def _retrieval_is_weak(chunks: List[Dict]) -> bool:

        similarities = [
            float(chunk.get("similarity", 0.0) or 0.0)
            for chunk in chunks
        ]

        return (
            not similarities
            or max(similarities) < 0.20
        )

    def generate(
        self,
        question: str,
        chunks: List[Dict],
        prompt_style: str = "zero_shot",
    ) -> dict:

        """Generate an answer grounded in retrieved evidence."""

        if (
            not chunks
            or self._retrieval_is_weak(chunks)
        ):

            return {
                "question": question,
                "answer": (
                    "I cannot answer this question based on "
                    "the available Wikipedia sources."
                ),
                "context_chunks": chunks or [],
                "provider": self.provider,
                "model": self._get_model_name(),
                "grounding_status": GroundingStatus.UNSUPPORTED.value,
                "prompt_style": prompt_style,
                "generation_error": False,
            }

        if self.provider == "local":

            context = []

            for index, chunk in enumerate(chunks[:3], 1):

                content = re.sub(
                    r"\s+",
                    " ",
                    str(chunk.get("text", ""))
                ).strip()[:650]

                context.append(
                    f"[Source {index}: "
                    f"{chunk.get('source', 'Wikipedia')}] "
                    f"{content}"
                )

            example = ""

            if prompt_style == "few_shot":

                example = (
                    "Example: Evidence: [Source 1] Water freezes "
                    "at zero degrees Celsius. "
                    "Question: At what temperature does water freeze? "
                    "Answer: Water freezes at zero degrees Celsius "
                    "[Source 1].\n"
                )

            prompt = (
                "Answer only using the evidence. "
                "Do not add outside facts. "
                "Cite factual statements as [Source N].\n"
                + example
                + f"Question: {question}\nEvidence:\n"
                + "\n".join(context)
                + "\nAnswer:"
            )

        else:

            _, prompt = build_prompt(
                question,
                chunks[:5],
                style=prompt_style
            )

        try:

            answer = self._call_llm(prompt).strip()

            if not answer:
                raise RuntimeError(
                    "The model returned an empty answer."
                )

            status = detect_grounding_status(
                answer,
                chunks
            )

            provider_label = self.provider
            generation_error = False

        except Exception:

            logger.exception(
                "Grounded generation failed; "
                "using extractive RAG fallback."
            )

            answer = self._extractive_grounded_answer(
                question,
                chunks
            )

            status = GroundingStatus.PARTIAL
            provider_label = "extractive-rag-fallback"
            generation_error = False

        return {
            "question": question,
            "answer": answer,
            "context_chunks": chunks,
            "provider": provider_label,
            "model": self._get_model_name(),
            "grounding_status": status.value,
            "prompt_style": prompt_style,
            "generation_error": generation_error,
        }

    def generate_ungrounded(self, question: str) -> dict:

        """
        Ungrounded mode:
        no retrieval and no Wikipedia source context.
        """

        prompt = (
            "Answer this question from general pretrained knowledge. "
            "Do not claim that you used external sources.\n"
            f"Question: {question}\nAnswer:"
        )

        try:

            answer = self._call_llm(prompt).strip()

            if not answer:
                raise RuntimeError(
                    "The model returned an empty answer."
                )

            error = False
            provider_label = self.provider

        except Exception as exc:

            logger.exception(
                "Ungrounded model generation failed."
            )

            answer = (
                "Ungrounded generation is unavailable because "
                "the configured language model failed "
                f"({type(exc).__name__})."
            )

            error = True
            provider_label = "generation-unavailable"

        return {
            "question": question,
            "answer": answer,
            "context_chunks": [],
            "provider": provider_label,
            "model": self._get_model_name(),
            "grounding_status": "ungrounded",
            "prompt_style": "none",
            "generation_error": error,
        }

    def _get_model_name(self):

        if self.provider == "anthropic":
            return ANTHROPIC_MODEL

        if self.provider == "openai":
            return OPENAI_CHAT_MODEL

        return LOCAL_LLM_MODEL


_generator: Optional[Generator] = None


def get_generator() -> Generator:

    global _generator

    if _generator is None:
        _generator = Generator()

    return _generator


def reset_generator() -> None:

    global _generator
    _generator = None
