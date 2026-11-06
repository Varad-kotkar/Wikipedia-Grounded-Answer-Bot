"""
Robust Wikipedia article collection with API JSON and HTML fallback.
"""

import logging
import re
import time
from typing import Optional
from urllib.parse import quote

import requests

from .config import RAW_DATA_DIR, TOPICS

logger = logging.getLogger(__name__)

API_URL = "https://en.wikipedia.org/w/api.php"
REST_URL = "https://en.wikipedia.org/api/rest_v1/page/html/{}"

HEADERS = {
    "User-Agent": (
        "WikiTruthAI/1.0 (educational project; "
        "https://github.com/Varad-kotkar/Wikipedia-Grounded-Answer-Bot)"
    )
}


def clean_text(text: str) -> str:

    if not text:
        return ""

    try:
        from bs4 import BeautifulSoup

        if (
            "<html" in text[:500].lower()
            or "<!doctype" in text[:500].lower()
        ):
            text = BeautifulSoup(
                text,
                "html.parser"
            ).get_text(" ", strip=True)

    except Exception:
        pass

    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[edit\]", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\[citation needed\]",
        "",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def title_to_filename(title: str) -> str:

    safe = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        title.lower()
    ).strip("_")

    return safe + ".txt"


def _fetch_api(
    title: str,
    session: requests.Session
) -> tuple[str, str]:

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
        timeout=20
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "content-type",
        ""
    ).lower()

    if "json" not in content_type:
        raise ValueError(
            "Wikipedia API returned non-JSON content."
        )

    payload = response.json()

    pages = payload.get(
        "query", {}
    ).get("pages", [])

    if not pages or pages[0].get("missing"):
        return title, ""

    page = pages[0]

    if "disambiguation" in page.get("pageprops", {}):
        return title, ""

    return (
        page.get("title", title),
        clean_text(page.get("extract", ""))
    )


def _fetch_html(
    title: str,
    session: requests.Session
) -> tuple[str, str]:

    url = REST_URL.format(
        quote(
            title.replace(" ", "_"),
            safe="_()"
        )
    )

    response = session.get(
        url,
        headers={
            **HEADERS,
            "Accept": "text/html"
        },
        timeout=25
    )

    response.raise_for_status()

    return title, clean_text(response.text)


def fetch_article(
    title: str,
    retries: int = 3,
    retry_delay: float = 2.0
) -> tuple[str, str]:

    with requests.Session() as session:

        for attempt in range(1, retries + 1):

            try:

                resolved, content = _fetch_api(
                    title,
                    session
                )

                if len(content) < 300:

                    logger.warning(
                        "Short API extract for %s; "
                        "trying HTML endpoint.",
                        title
                    )

                    resolved, content = _fetch_html(
                        title,
                        session
                    )

                if len(content) >= 300:

                    logger.info(
                        "Fetched %s (%d characters)",
                        resolved,
                        len(content)
                    )

                    return resolved, content

                raise ValueError(
                    "No usable article text returned by Wikipedia."
                )

            except Exception as exc:

                if attempt >= retries:

                    logger.error(
                        "Could not collect %s after %d attempts: %s",
                        title,
                        retries,
                        exc
                    )

                    break

                pause = retry_delay * attempt

                logger.warning(
                    "Wikipedia request %d/%d for %s failed: %s; "
                    "retrying in %.1fs",
                    attempt,
                    retries,
                    title,
                    exc,
                    pause
                )

                time.sleep(pause)

    return title, ""


def collect_all(
    topics: Optional[list[str]] = None,
    delay: float = 1.0,
    skip_existing: bool = True
) -> None:

    topics = TOPICS if topics is None else topics

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    downloaded = 0
    reused = 0
    failed = 0

    for topic in topics:

        path = RAW_DATA_DIR / title_to_filename(topic)

        # Reuse previously collected valid articles.
        if skip_existing and path.exists():

            try:

                cached = path.read_text(
                    encoding="utf-8"
                ).strip()

                if len(cached) >= 300:

                    logger.info(
                        "Reusing saved Wikipedia article: %s",
                        topic
                    )

                    reused += 1
                    continue

            except OSError:

                logger.exception(
                    "Could not read cached file: %s",
                    path
                )

        title, content = fetch_article(topic)

        if len(content) >= 300:

            path.write_text(
                content,
                encoding="utf-8"
            )

            logger.info(
                "Saved %s (%d characters)",
                title,
                len(content)
            )

            downloaded += 1

        else:

            failed += 1

            logger.error(
                "No usable article was saved for %s",
                topic
            )

        time.sleep(max(0.0, delay))

    total_files = len(
        list(RAW_DATA_DIR.glob("*.txt"))
    )

    logger.info(
        "Collection complete: downloaded=%d reused=%d "
        "failed=%d raw_files=%d",
        downloaded,
        reused,
        failed,
        total_files
    )
