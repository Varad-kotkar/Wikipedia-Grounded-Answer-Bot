
"""
WikiTruth AI - Wikipedia data collection.

Fetches article extracts from the official MediaWiki API,
cleans the text, and saves it to data/raw/.
"""

import logging
import re
import time
from pathlib import Path
from typing import Optional

import requests

from .config import RAW_DATA_DIR, TOPICS

logger = logging.getLogger(__name__)

API_URL = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": (
        "WikiTruthAI/1.0 "
        "(educational university project; Wikipedia API client)"
    )
}

REQUEST_TIMEOUT = 25


def clean_text(text: str) -> str:
    """Remove common Wikipedia formatting artefacts."""

    if not text:
        return ""

    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[edit\]", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\[citation needed\]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def title_to_filename(title: str) -> str:
    """Convert an article title into a safe filename."""

    filename = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        title.strip().lower(),
    ).strip("_")

    return f"{filename}.txt"


def _request_article(
    title: str,
    session: requests.Session,
) -> tuple[Optional[str], Optional[str], bool]:
    """
    Fetch one article.

    Returns:
        (resolved_title, extract, is_disambiguation)
    """

    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
        "prop": "extracts|pageprops",
        "explaintext": "1",
        "exsectionformat": "plain",
        "titles": title,
    }

    response = session.get(
        API_URL,
        params=params,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    pages = data.get("query", {}).get("pages", [])

    if not pages:
        return None, None, False

    page = pages[0]

    if page.get("missing"):
        logger.warning("Wikipedia page not found: %s", title)
        return None, None, False

    page_title = page.get("title", title)
    page_props = page.get("pageprops", {})

    is_disambiguation = "disambiguation" in page_props

    extract = clean_text(page.get("extract", ""))

    if not extract:
        logger.warning("Wikipedia returned an empty extract: %s", page_title)
        return page_title, None, is_disambiguation

    return page_title, extract, is_disambiguation


def fetch_article(
    title: str,
    retries: int = 3,
    retry_delay: float = 3.0,
) -> tuple[str, str]:
    """
    Fetch an article with retries.

    Returns:
        (resolved_title, cleaned_content)

    A failed or ambiguous article returns an empty content string.
    """

    session = requests.Session()

    for attempt in range(1, retries + 1):
        try:
            resolved_title, content, is_disambiguation = (
                _request_article(title, session)
            )

            if is_disambiguation:
                logger.warning(
                    "Skipping ambiguous Wikipedia page: %s",
                    resolved_title or title,
                )
                return title, ""

            if content:
                logger.info(
                    "Fetched '%s' - %s characters",
                    resolved_title,
                    len(content),
                )
                return resolved_title or title, content

            return title, ""

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
            ValueError,
        ) as exc:
            if attempt < retries:
                wait_time = retry_delay * attempt

                logger.warning(
                    "Fetch failed for '%s' (attempt %s/%s): %s. "
                    "Retrying in %s seconds.",
                    title,
                    attempt,
                    retries,
                    exc,
                    wait_time,
                )

                time.sleep(wait_time)

            else:
                logger.error(
                    "Could not fetch '%s' after %s attempts: %s",
                    title,
                    retries,
                    exc,
                )

        except Exception:
            logger.exception("Unexpected error fetching '%s'.", title)
            break

    return title, ""


def collect_all(
    topics: Optional[list[str]] = None,
    delay: float = 1.5,
    skip_existing: bool = True,
) -> None:
    """
    Download configured Wikipedia articles into data/raw/.

    Args:
        topics: Article titles. Defaults to TOPICS from config.py.
        delay: Delay between article requests.
        skip_existing: Skip files that already contain usable text.
    """

    if topics is None:
        topics = TOPICS

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0
    skipped = 0

    for topic in topics:
        filepath = RAW_DATA_DIR / title_to_filename(topic)

        if skip_existing and filepath.exists():
            try:
                existing_text = filepath.read_text(
                    encoding="utf-8"
                ).strip()

                if len(existing_text) > 100:
                    logger.info(
                        "Skipping '%s': existing file is usable.",
                        topic,
                    )
                    skipped += 1
                    continue

            except OSError:
                logger.warning(
                    "Could not read existing file: %s",
                    filepath,
                )

        resolved_title, content = fetch_article(topic)

        if content and len(content) > 100:
            try:
                filepath.write_text(
                    content,
                    encoding="utf-8",
                )

                logger.info(
                    "Saved '%s' to %s",
                    resolved_title,
                    filepath,
                )
                successful += 1

            except OSError:
                logger.exception(
                    "Could not save article: %s",
                    filepath,
                )
                failed += 1

        else:
            logger.warning(
                "No usable content collected for '%s'.",
                topic,
            )
            failed += 1

        time.sleep(max(0.0, delay))

    total_files = len(list(RAW_DATA_DIR.glob("*.txt")))

    logger.info(
        "Collection finished. Successful: %s | Failed: %s | "
        "Skipped: %s | Files on disk: %s",
        successful,
        failed,
        skipped,
        total_files,
    )
