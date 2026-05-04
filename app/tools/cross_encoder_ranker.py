"""Cross-encoder reranker (G-6).

Replaces the default LLM reranker with `sentence-transformers` CrossEncoder
for ~100× cost reduction and ~2-5× latency reduction on top-k = 8.

Lazy import: `sentence-transformers` is a ~400MB dep (PyTorch). Only loaded
if `settings.reranker_backend == "cross_encoder"`, so users who don't care
pay nothing.

The model loads once per process; CPU inference on the MS-MARCO MiniLM-L6
variant is ~5ms for 10 pairs. For GPU-backed deployments, the same API
works by setting `CUDA_VISIBLE_DEVICES`.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.config import settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _model():
    """Lazily construct the CrossEncoder. Errors defer cleanly to the LLM fallback."""
    try:
        from sentence_transformers import CrossEncoder

        log.info("loading_cross_encoder model=%s", settings.cross_encoder_model)
        return CrossEncoder(settings.cross_encoder_model)
    except Exception:  # noqa: BLE001
        log.exception("cross_encoder_init_failed — falling back to LLM reranker")
        return None


async def rerank(
    query: str,
    docs: list[dict],
    *,
    top_n: int | None = None,
) -> list[dict]:
    """Score (query, doc) pairs with a cross-encoder and return reordered docs.

    On any error (model not installed, inference failure) returns the input
    unchanged — callers treat this as a no-op graceful degradation.
    """
    if not docs or len(docs) == 1:
        return docs

    model = _model()
    if model is None:
        return docs

    import asyncio

    pairs = [(query, (d.get("content") or "")[:2000]) for d in docs]
    # CrossEncoder.predict is sync + CPU/GPU-bound. Run off the event loop.
    try:
        scores = await asyncio.to_thread(model.predict, pairs)
    except Exception:  # noqa: BLE001
        log.exception("cross_encoder_predict_failed — returning original order")
        return docs

    ranked_indices = sorted(range(len(docs)), key=lambda i: -float(scores[i]))
    reranked: list[dict] = []
    for position, original_idx in enumerate(ranked_indices):
        d = dict(docs[original_idx])
        d["rerank_score"] = position  # 0 = most relevant
        d["cross_encoder_score"] = float(scores[original_idx])
        reranked.append(d)

    if top_n is not None:
        reranked = reranked[:top_n]
    return reranked
