"""
src/wikipedia_retriever.py — WikiTruth AI
Builds Wikipedia article URLs and extracts source links from retrieved chunks.
"""

import urllib.parse
from typing import List, Dict


WIKIPEDIA_BASE_URL = "https://en.wikipedia.org/wiki/"


def get_article_url(title: str) -> str:
    """
    Convert a Wikipedia article title to its canonical URL.
    Example: "Machine learning" → "https://en.wikipedia.org/wiki/Machine_learning"
    """
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    return WIKIPEDIA_BASE_URL + encoded


def get_source_links(chunks: List[Dict]) -> List[Dict]:
    """
    Extract unique sources from retrieved chunks and build Wikipedia URLs.

    Args:
        chunks: list of chunk dicts with 'source' and optionally 'url' keys

    Returns:
        List of dicts: [{'title': str, 'url': str, 'chunk_count': int}]
    """
    seen: Dict[str, Dict] = {}
    for chunk in chunks:
        source = chunk.get("source", "")
        if not source:
            continue
        if source not in seen:
            # Use stored URL if available, else construct it
            url = chunk.get("url", "") or get_article_url(source)
            seen[source] = {"title": source, "url": url, "chunk_count": 0}
        seen[source]["chunk_count"] += 1
    
    # Sort by chunk count (most relevant first)
    return sorted(seen.values(), key=lambda x: x["chunk_count"], reverse=True)


def format_source_citation(chunks: List[Dict]) -> str:
    """
    Format source citations as a markdown string with clickable links.
    """
    sources = get_source_links(chunks)
    if not sources:
        return "*No sources available.*"
    
    lines = ["**Sources:**"]
    for i, src in enumerate(sources, 1):
        lines.append(f"{i}. [{src['title']}]({src['url']}) — {src['chunk_count']} passage(s) retrieved")
    return "\n".join(lines)
