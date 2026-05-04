"""In-flight request deduplication (G-4).

When two callers submit near-identical topics within a short window, we
want to avoid running the graph twice. This module keeps a small in-memory
map of `(api_key_hash, topic_embedding) → Future[thread_id]`. The second
caller subscribes to the existing future and reuses the winner's thread_id.

Design choices:
  * **Process-local** — the futures map lives in this module's namespace,
    so dedup only works within a single API worker. That's fine for a
    single-pod deployment and gives us 90%+ of the win with ~30 lines.
  * **TTL-limited** — entries auto-expire after `settings.dedup_window_seconds`
    so the map stays small.
  * **Scoped by api_key_hash** — one user's dedup never pulls another
    user's thread_id. Same guarantee as C-3.

Multi-replica migration path (when it matters)
----------------------------------------------
At Kubernetes scale with 2+ replicas behind a load balancer, in-process
dedup becomes a coin flip: two concurrent duplicate requests only collide
if the LB routes them to the same pod. On N replicas the hit rate drops
roughly as 1/N. To keep the win at multi-replica scale, migrate to a
Redis-backed version:

  1. Replace the in-process map with a Redis key per `(hash, vec_bucket)`
     where `vec_bucket` is a coarse quantisation of the topic embedding
     (e.g. first 8 components rounded to 2 decimals — a cheap ANN bucket).
  2. The leader writes `SET rl:dedup:<bucket> <thread_id> EX 30 NX`. The
     returned thread_id (if someone else's key already exists) tells the
     caller to await the existing run.
  3. Leaders publish completion via `PUBLISH dedup:done:<bucket> <thread_id>`;
     followers `SUBSCRIBE` until they get the message or time out.
  4. The TTL + PUBLISH combo gives the same semantics as `asyncio.Future`
     does locally, with at-least-once delivery that's acceptable here (a
     follower that misses the pubsub just falls through to its own graph
     run — worst case we paid for two runs, exactly what we have today).

Implementation effort at that scale is ~100 LOC + `redis.asyncio.pubsub`.
Not done in this version because single-pod is the current deployment
target; the interface (`claim_or_wait` / `release` / `fail`) is already
shaped so that swap is a file-replace, not a refactor.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from app.config import settings
from app.tools.retriever import embed_text

log = logging.getLogger(__name__)


@dataclass
class _Entry:
    caller_hash: str | None
    topic: str
    embedding: list[float]
    created_at: float
    future: asyncio.Future[str] = field(default_factory=asyncio.Future)


_ENTRIES: list[_Entry] = []
_LOCK = asyncio.Lock()


def _cosine(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
    return float(np.dot(va, vb) / denom)


def _gc_expired(now: float) -> None:
    ttl = settings.dedup_window_seconds
    kept: list[_Entry] = []
    for e in _ENTRIES:
        if now - e.created_at > ttl:
            if not e.future.done():
                # Unblock any lingering waiters with a sentinel so they fall
                # through to the "run it yourself" path.
                e.future.set_exception(asyncio.CancelledError())
            continue
        kept.append(e)
    _ENTRIES[:] = kept


async def claim_or_wait(
    topic: str,
    caller_hash: str | None,
    threshold: float | None = None,
) -> tuple[asyncio.Future[str], bool]:
    """Reserve an in-flight slot for `topic` or attach to an existing one.

    Returns `(future, is_leader)`:
      * `is_leader=True` → you own the slot, run the graph, and call
        `future.set_result(thread_id)` when done.
      * `is_leader=False` → another caller is already running it; `await
        future` to get their thread_id.
    """
    if threshold is None:
        threshold = settings.dedup_similarity_threshold

    embedding = await asyncio.to_thread(embed_text, topic)
    now = time.monotonic()
    async with _LOCK:
        _gc_expired(now)
        for e in _ENTRIES:
            if e.caller_hash != caller_hash:
                continue
            if _cosine(e.embedding, embedding) >= threshold:
                log.info(
                    "dedup_hit topic=%r matched=%r similarity>=%.2f",
                    topic,
                    e.topic,
                    threshold,
                )
                return e.future, False

        entry = _Entry(
            caller_hash=caller_hash,
            topic=topic,
            embedding=embedding,
            created_at=now,
        )
        _ENTRIES.append(entry)
        return entry.future, True


async def release(future: asyncio.Future[str], thread_id: str) -> None:
    """Broadcast the finished thread_id to any waiters."""
    if not future.done():
        future.set_result(thread_id)


async def fail(future: asyncio.Future[str], exc: BaseException) -> None:
    """Propagate failure to waiters so they fall back to their own graph run."""
    if not future.done():
        future.set_exception(exc)


def reset_for_tests() -> None:
    """Clear state between tests."""
    _ENTRIES.clear()
