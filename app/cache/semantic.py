"""Semantic cache for finished reports.

If a user asks "LangGraph 2024" and later "LangGraph in 2024", the vector
similarity of the topic embeddings is > 0.95 and we can return the cached
report instead of re-running the whole graph. Saves:

  * 6 nodes × N sections × 2 LLM calls per section ≈ dozens of API calls
  * Full graph latency (10-30s) collapses to one embedding + one DB read
  * User gets faster response for semantically equivalent questions

Invalidation: TTL-based. Reports older than `cache_ttl_days` are skipped
in the lookup, preventing stale answers on time-sensitive topics
(financial data, current events).

Bypass knobs:
  * `settings.semantic_cache_threshold = 0.95` — hit threshold.
  * `settings.semantic_cache_ttl_days = 7`    — stricter than doc cache TTL
                                                  since reports are more
                                                  opinionated than chunks.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, text

from app.config import settings
from app.models.db import TopicCache
from app.models.engine import SessionLocal
from app.tools.retriever import EMBEDDING_DIMS, embed_text

log = logging.getLogger(__name__)


@dataclass
class CachedReport:
    topic: str
    final_report: str
    citations: list[dict]
    similarity: float
    cache_id: str


# Per-user scoping (C-3): a hit only matches when `api_key_hash` is the
# same as the caller's, OR both are NULL (dev-mode / legacy). Without this,
# any user could pull any other user's cached report via a near-identical
# topic embedding.
_LOOKUP_SQL = text(
    """
    SELECT id, topic, final_report, citations_json,
           1 - (topic_vec <=> :vec) AS similarity
    FROM topic_cache
    WHERE topic_vec IS NOT NULL
      AND created_at > now() - make_interval(days => :ttl_days)
      AND (
            (:key_hash IS NULL AND api_key_hash IS NULL)
         OR (api_key_hash = :key_hash)
      )
    ORDER BY topic_vec <=> :vec
    LIMIT 1
    """
).bindparams(
    bindparam("vec", type_=Vector(EMBEDDING_DIMS)),
    bindparam("ttl_days"),
    bindparam("key_hash"),
)


def lookup(topic: str, api_key_hash: str | None = None) -> CachedReport | None:
    """Return a cached report if one exists with similarity above threshold.

    Scoped by `api_key_hash` — only returns rows this caller created, or
    anonymous rows when the caller itself is anonymous.
    """
    threshold = settings.semantic_cache_threshold
    ttl = settings.semantic_cache_ttl_days
    vec = embed_text(topic)

    with SessionLocal() as db:
        row = db.execute(
            _LOOKUP_SQL,
            {"vec": vec, "ttl_days": ttl, "key_hash": api_key_hash},
        ).fetchone()
        if row is None or float(row.similarity) < threshold:
            return None
        # Count the hit (best-effort — don't let telemetry failures break the cache).
        try:
            db.execute(
                text("UPDATE topic_cache SET hit_count = hit_count + 1 WHERE id = :id"),
                {"id": row.id},
            )
            db.commit()
        except Exception:  # noqa: BLE001
            log.exception("semantic_cache_hit_counter_failed")

    citations = json.loads(row.citations_json) if row.citations_json else []
    return CachedReport(
        topic=row.topic,
        final_report=row.final_report,
        citations=citations,
        similarity=float(row.similarity),
        cache_id=str(row.id),
    )


# Size limits (H-5): keep reports cached but bound memory growth.
_MAX_REPORT_CHARS = 100_000
_MAX_CITATIONS = 20


def store(
    topic: str,
    final_report: str,
    citations: list[dict],
    api_key_hash: str | None = None,
) -> None:
    """Persist a finished report keyed by its topic embedding.

    Size-capped (H-5) so one caller can't OOM the table. `api_key_hash`
    scopes the entry (C-3); passing None stores it as anonymous / global.
    """
    if not final_report or len(final_report) > _MAX_REPORT_CHARS:
        return
    citations = (citations or [])[:_MAX_CITATIONS]
    vec = embed_text(topic)
    with SessionLocal() as db:
        db.add(
            TopicCache(
                id=uuid.uuid4(),
                topic=topic[:500],
                final_report=final_report,
                citations_json=json.dumps(citations),
                topic_vec=vec,
                api_key_hash=api_key_hash,
            )
        )
        db.commit()


def prune_old(retention_days: int | None = None) -> int:
    """Delete topic_cache rows older than the retention window.

    TTL filters at *read* time, but rows still sit on disk and slow index
    maintenance. Run this periodically (cron / release command) to keep
    the table bounded.
    """
    if retention_days is None:
        retention_days = settings.semantic_cache_ttl_days * 2
    with SessionLocal() as db:
        result = db.execute(
            text("DELETE FROM topic_cache WHERE created_at < now() - make_interval(days => :days)"),
            {"days": retention_days},
        )
        db.commit()
    return result.rowcount or 0
