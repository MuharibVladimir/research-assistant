"""Recursive text splitter used before embedding / caching.

Tavily returns ~3 snippets per query, joined they easily exceed 2-3k chars
of English text. A single embedding call captures the gist but washes out
individual facts. By splitting into overlapping chunks (~800 chars, 100
overlap on paragraph/sentence boundaries), each chunk becomes its own
retrievable unit — the vector search finds the exact fragment that
matched, not a blurred average.

Using `langchain_text_splitters.RecursiveCharacterTextSplitter` — the
LangChain-standard splitter: tries paragraph → newline → sentence →
word boundaries in that order, falls back to hard char chunks only
as a last resort.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


@lru_cache(maxsize=1)
def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Return a process-wide singleton splitter configured from settings."""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        keep_separator=True,
    )
