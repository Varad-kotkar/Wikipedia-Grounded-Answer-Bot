"""
src/grounding.py — WikiTruth AI
Grounding prompts, context validation, and grounding status detection.
Implements zero-shot and few-shot grounding prompts for U1 prompt engineering.
"""

from enum import Enum
from typing import List, Dict, Tuple


class GroundingStatus(Enum):
    GROUNDED = "grounded"          # answer fully supported by context
    PARTIAL = "partial"            # answer partially supported
    UNSUPPORTED = "unsupported"    # answer not supported / LLM refused
    LOW_CONFIDENCE = "low_confidence"  # retrieval similarity too low


# --- ZERO-SHOT GROUNDING PROMPT ---
ZERO_SHOT_SYSTEM = """You are WikiTruth AI, a strict fact-checker and question-answering assistant.
You answer questions EXCLUSIVELY using the Wikipedia passages provided below.

RULES YOU MUST FOLLOW:
1. Use ONLY information from the provided Wikipedia passages.
2. NEVER use your pre-trained knowledge to fill in missing information.
3. NEVER invent facts, statistics, dates, names, or references.
4. If the passages do not contain the answer, respond EXACTLY:
   "I cannot answer this question based on the available Wikipedia sources."
5. Cite the source number [Source N] for each factual claim you make.
6. Clearly separate directly supported facts from inferences."""

ZERO_SHOT_TEMPLATE = """\
{system}

Wikipedia Passages:
{context}

Question: {question}

Answer (cite sources using [Source N] notation):"""


# --- FEW-SHOT GROUNDING PROMPT ---
FEW_SHOT_EXAMPLES = """
EXAMPLE 1 — Supported question:
Passage: [Source 1: Machine learning] Machine learning is a subset of artificial
intelligence that gives computers the ability to learn from data without being
explicitly programmed.
Q: What is machine learning?
A: According to Wikipedia [Source 1], machine learning is a subset of artificial
intelligence that enables computers to learn from data without explicit programming.

EXAMPLE 2 — Unsupported question:
Passage: [Source 1: Machine learning] Machine learning algorithms include
supervised, unsupervised, and reinforcement learning paradigms.
Q: Who invented the Python programming language?
A: I cannot answer this question based on the available Wikipedia sources.
The provided passages cover machine learning but contain no information about
the Python programming language's inventor.

EXAMPLE 3 — Partially supported question:
Passage: [Source 1: Deep learning] Deep learning uses multi-layered neural
networks. It has applications in image recognition and natural language processing.
Q: What are ALL the applications of deep learning?
A: Based on [Source 1], deep learning is applied in image recognition and natural
language processing. However, the available Wikipedia passages do not provide a
complete list of all applications."""

FEW_SHOT_SYSTEM = """You are WikiTruth AI, a strict fact-checker and question-answering assistant.
You answer questions EXCLUSIVELY using the Wikipedia passages provided below.

RULES YOU MUST FOLLOW:
1. Use ONLY information from the provided Wikipedia passages.
2. NEVER use your pre-trained knowledge to fill in missing information.
3. NEVER invent facts, statistics, dates, names, or references.
4. If the passages do not contain the answer, respond EXACTLY:
   "I cannot answer this question based on the available Wikipedia sources."
5. Cite the source number [Source N] for each factual claim you make.
6. Follow the examples below for the required response format."""

FEW_SHOT_TEMPLATE = """\
{system}

{examples}

NOW ANSWER THE FOLLOWING:
Wikipedia Passages:
{context}

Question: {question}

Answer (cite sources using [Source N] notation):"""


def build_prompt(question: str, chunks: List[Dict], style: str = "zero_shot") -> Tuple[str, str]:
    """
    Build the full grounding prompt for the LLM.

    Args:
        question: user's natural-language question
        chunks: list of retrieved chunk dicts (must have 'source' and 'text' keys)
        style: 'zero_shot' or 'few_shot'

    Returns:
        (system_prompt, user_prompt) tuple
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[Source {i}: {chunk['source']}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    if style == "few_shot":
        user_prompt = FEW_SHOT_TEMPLATE.format(
            system=FEW_SHOT_SYSTEM,
            examples=FEW_SHOT_EXAMPLES,
            context=context,
            question=question,
        )
        return FEW_SHOT_SYSTEM, user_prompt
    else:  # zero_shot (default)
        user_prompt = ZERO_SHOT_TEMPLATE.format(
            system=ZERO_SHOT_SYSTEM,
            context=context,
            question=question,
        )
        return ZERO_SHOT_SYSTEM, user_prompt


def validate_context(chunks: List[Dict], threshold: float = 0.25) -> Tuple[bool, float]:
    """
    Check if retrieved context is of sufficient quality.

    Returns:
        (is_valid, avg_similarity)
    """
    if not chunks:
        return False, 0.0
    sims = [c.get("similarity", 0.0) for c in chunks]
    avg_sim = sum(sims) / len(sims)
    return avg_sim >= threshold, round(avg_sim, 4)


def detect_grounding_status(answer: str, chunks: List[Dict]) -> GroundingStatus:
    """
    Determine the grounding status of an answer.

    Returns:
        GroundingStatus enum value
    """
    answer_lower = answer.lower()
    
    # Explicit refusal
    refusal_phrases = [
        "i cannot answer",
        "i don't know",
        "i do not know",
        "not in the available",
        "not found in",
        "cannot be answered",
        "no information",
    ]
    if any(phrase in answer_lower for phrase in refusal_phrases):
        return GroundingStatus.UNSUPPORTED

    # Check if sources are cited
    has_citations = "[source" in answer_lower or "[src" in answer_lower
    
    # Check partial phrases
    partial_phrases = [
        "however, the available",
        "does not provide a complete",
        "partially",
        "not fully covered",
    ]
    has_partial = any(phrase in answer_lower for phrase in partial_phrases)
    
    if not chunks:
        return GroundingStatus.LOW_CONFIDENCE
    
    avg_sim = sum(c.get("similarity", 0.0) for c in chunks) / len(chunks)
    if avg_sim < 0.20:
        return GroundingStatus.LOW_CONFIDENCE

    if has_partial:
        return GroundingStatus.PARTIAL
    
    return GroundingStatus.GROUNDED


def get_status_badge(status: GroundingStatus) -> dict:
    """
    Return display properties for a grounding status badge.
    Returns dict with 'label', 'color', 'icon'.
    """
    badges = {
        GroundingStatus.GROUNDED: {
            "label": "Fully Grounded",
            "color": "green",
            "icon": "✅",
        },
        GroundingStatus.PARTIAL: {
            "label": "Partially Grounded",
            "color": "orange",
            "icon": "⚠️",
        },
        GroundingStatus.UNSUPPORTED: {
            "label": "Not Supported",
            "color": "red",
            "icon": "❌",
        },
        GroundingStatus.LOW_CONFIDENCE: {
            "label": "Low Confidence",
            "color": "gray",
            "icon": "🔍",
        },
    }
    return badges.get(status, badges[GroundingStatus.LOW_CONFIDENCE])
