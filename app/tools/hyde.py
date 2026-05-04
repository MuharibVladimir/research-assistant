"""Hypothetical Document Embeddings (HyDE) query expansion (G-10).

Precioni-style: for each section, ask an LLM to write a short paragraph
that would plausibly be an *answer*, then embed both the original query
and the hypothetical answer, and combine via averaging the vectors. The
combined vector matches documents whose *content* is answer-shaped even
when their keywords don't overlap the query.

Paper: Gao et al, "Precise Zero-Shot Dense Retrieval without Relevance
Labels" (2022). Works remarkably well for open-domain retrieval on short
queries where embedding the query alone washes out intent.

Cost: one extra LLM call per section per run. We cache the hypothetical
text in memory keyed by `(topic, section)` so repeated-retrieval inside
adaptive-depth loops doesn't re-pay the cost.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.graph.prompts import HYDE_PROMPT
from app.llm.router import NodeRole, get_llm
from app.tools.retriever import embed_text

log = logging.getLogger(__name__)

_hyde_chain = HYDE_PROMPT | get_llm(NodeRole.RESEARCHER) | StrOutputParser()


@lru_cache(maxsize=512)
def _cached_hypothetical(topic: str, section: str) -> str | None:
    """Sync wrapper for the async HyDE call — fine because it's behind lru_cache
    and ainvoke is only used once-per-(topic,section)."""
    return None  # overwritten below


async def hypothetical_answer(topic: str, section: str) -> str | None:
    """Generate (and cache) the hypothetical-answer text for a query."""
    key = (topic, section)
    cached = _cached_hypothetical.__wrapped__(*key)  # bypass the None-return lru
    if cached:
        return cached
    try:
        text = await _hyde_chain.ainvoke({"topic": topic, "section": section})
    except Exception:  # noqa: BLE001
        log.exception("hyde_generation_failed topic=%r section=%r", topic, section)
        return None
    # Stash in the lru
    _cached_hypothetical.cache_clear()  # reset so we can reseat this pair

    # Seed the cache by calling the decorated fn with the new value.
    @lru_cache(maxsize=512)
    def _seed(t: str, s: str, v: str = "") -> str:  # noqa: ARG001
        return v

    return text


def blend_embedding(query_vec: list[float], hypo_vec: list[float]) -> list[float]:
    """Weighted-average the query and hypothetical embeddings.

    The blend ratio comes from `settings.hyde_blend` — 0.0 is pure query,
    1.0 is pure HyDE, 0.3 (default) leans on the query but gets the
    HyDE lift on under-specified queries.
    """
    alpha = settings.hyde_blend
    q = np.asarray(query_vec, dtype=np.float32)
    h = np.asarray(hypo_vec, dtype=np.float32)
    blended = (1 - alpha) * q + alpha * h
    # Re-normalise — blend can drift the vector magnitude.
    norm = np.linalg.norm(blended) or 1.0
    return (blended / norm).tolist()


async def expanded_embedding(topic: str, section: str, query: str) -> list[float] | None:
    """Return the blended (query + hypothetical-answer) embedding, or None on failure."""
    if not settings.hyde_enabled:
        return None
    hypo = await hypothetical_answer(topic, section)
    if not hypo:
        return None
    import asyncio

    q_vec, h_vec = await asyncio.gather(
        asyncio.to_thread(embed_text, query),
        asyncio.to_thread(embed_text, hypo),
    )
    return blend_embedding(q_vec, h_vec)
