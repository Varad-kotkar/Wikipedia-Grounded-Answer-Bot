"""
Phase 2 — Data Collection
Fetch Wikipedia articles, clean the text, and save to data/raw/.
"""

import re
import time
import logging
from pathlib import Path

import wikipedia

from .config import RAW_DATA_DIR, TOPICS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Light-touch Wikipedia text cleaning:
      - strip citation markers like [1], [23]
      - strip [edit] section markers
      - collapse 3+ consecutive blank lines to 2
      - collapse multiple spaces
    """
    text = re.sub(r"\[\d+\]", "", text)          # remove [123] citations
    text = re.sub(r"\[edit\]", "", text)          # remove [edit] markers
    text = re.sub(r"\[citation needed\]", "", text)
    text = re.sub(r" {2,}", " ", text)            # collapse multiple spaces
    text = re.sub(r"\n{3,}", "\n\n", text)        # max 2 blank lines
    return text.strip()


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def title_to_filename(title: str) -> str:
    """Convert 'Artificial intelligence' -> 'artificial_intelligence.txt'."""
    return title.lower().replace(" ", "_") + ".txt"


def fetch_article(title: str, retries: int = 3, retry_delay: float = 3.0) -> tuple[str, str]:
    """
    Fetch a Wikipedia article by title.
    Returns (resolved_title, cleaned_content).
    On failure returns (title, '').
    Retries on transient errors (e.g. JSON parse failures from the API).
    """
    wikipedia.set_lang("en")

    for attempt in range(1, retries + 1):
        try:
            page = wikipedia.page(title, auto_suggest=False)
            content = clean_text(page.content)
            logger.info(f"Fetched '{page.title}' — {len(content):,} chars")
            return page.title, content

        except wikipedia.exceptions.DisambiguationError as exc:
            fallback = exc.options[0]
            logger.warning(f"Disambiguation for '{title}' → trying '{fallback}'")
            page = wikipedia.page(fallback, auto_suggest=False)
            content = clean_text(page.content)
            return page.title, content

        except wikipedia.exceptions.PageError:
            logger.error(f"Page not found: '{title}'")
            return title, ""

        except Exception as exc:  # noqa: BLE001
            if attempt < retries:
                logger.warning(
                    f"Transient error fetching '{title}' (attempt {attempt}/{retries}): {exc} "
                    f"— retrying in {retry_delay}s…"
                )
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to fetch '{title}' after {retries} attempts: {exc}")
                return title, ""


# ---------------------------------------------------------------------------
# Batch collection
# ---------------------------------------------------------------------------

def collect_all(
    topics: list[str] | None = None,
    delay: float = 1.5,
    skip_existing: bool = True,
) -> None:
    """
    Download all Wikipedia articles and save them to data/raw/.

    Args:
        topics: list of Wikipedia article titles (defaults to TOPICS).
        delay: seconds to wait between requests (be polite to Wikipedia).
        skip_existing: if True, skip articles already on disk.
    """
    if topics is None:
        topics = TOPICS

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for topic in topics:
        filepath = RAW_DATA_DIR / title_to_filename(topic)

        if skip_existing and filepath.exists():
            logger.info(f"Skipping '{topic}' (already downloaded)")
            continue

        resolved_title, content = fetch_article(topic)

        if content:
            filepath.write_text(content, encoding="utf-8")
            logger.info(f"Saved → {filepath.name}")
        else:
            logger.warning(f"Empty content for '{topic}' — skipping save")

        time.sleep(delay)

    saved = list(RAW_DATA_DIR.glob("*.txt"))
    logger.info(f"Data collection complete. {len(saved)} files in {RAW_DATA_DIR}")
