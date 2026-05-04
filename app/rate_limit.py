"""Distributed rate limiter backed by Redis.

Implementation: atomic sliding-window counter using a single Lua script so
multiple API replicas share state. The script zrangebyscore-trims old
timestamps, counts what's left, and conditionally zadds — one round-trip,
one transaction per request.

Falls back to an in-process deque-based limiter when Redis is unconfigured
(tests / local dev without Redis container).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque

from app.config import settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lua script: returns 1 if the request was admitted, else seconds until retry.
#   KEYS[1] = rate-limit key (e.g. "rl:<api_key>")
#   ARGV[1] = window length (seconds)
#   ARGV[2] = max requests per window
#   ARGV[3] = current unix-ts (ms)
#   ARGV[4] = request token (unique, e.g. ms timestamp + random)
# ---------------------------------------------------------------------------

_LUA_SLIDING_WINDOW = """
local key          = KEYS[1]
local window_ms    = tonumber(ARGV[1]) * 1000
local max_reqs     = tonumber(ARGV[2])
local now_ms       = tonumber(ARGV[3])
local token        = ARGV[4]

-- Drop entries outside the window
redis.call('ZREMRANGEBYSCORE', key, 0, now_ms - window_ms)

local count = tonumber(redis.call('ZCARD', key))
if count >= max_reqs then
  -- Oldest entry determines retry-after
  local oldest_arr = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
  local oldest_ms  = tonumber(oldest_arr[2])
  local retry_s    = math.ceil((oldest_ms + window_ms - now_ms) / 1000)
  return {0, retry_s}
end

redis.call('ZADD', key, now_ms, token)
redis.call('PEXPIRE', key, window_ms)
return {1, 0}
"""


class _InProcessLimiter:
    """Fallback limiter for tests / Redis-less dev mode."""

    def __init__(self) -> None:
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: str, limit: int, window: float) -> tuple[bool, int]:
        now = time.monotonic()
        async with self._lock:
            bucket = self._buckets[key]
            while bucket and now - bucket[0] > window:
                bucket.popleft()
            if len(bucket) >= limit:
                retry = int(window - (now - bucket[0])) + 1
                return False, retry
            bucket.append(now)
            return True, 0

    async def cleanup(self) -> int:
        now = time.monotonic()
        async with self._lock:
            stale = [
                k for k, bucket in self._buckets.items() if not bucket or (now - bucket[-1]) > 120
            ]
            for k in stale:
                del self._buckets[k]
        return len(stale)


class RedisLimiter:
    """Distributed limiter. Constructed with an already-connected redis.asyncio client."""

    def __init__(self, client, prefix: str = "rl:") -> None:
        self._client = client
        self._prefix = prefix
        self._script = client.register_script(_LUA_SLIDING_WINDOW)

    async def check(self, key: str, limit: int, window: float) -> tuple[bool, int]:
        now_ms = int(time.time() * 1000)
        token = f"{now_ms}-{id(object())}"
        admitted, retry_s = await self._script(
            keys=[self._prefix + key],
            args=[int(window), int(limit), now_ms, token],
        )
        return bool(admitted), int(retry_s)

    async def cleanup(self) -> int:
        # Redis handles expiry natively via PEXPIRE
        return 0


# ---------------------------------------------------------------------------
# Singleton accessor used by routes.py
# ---------------------------------------------------------------------------

_limiter: RedisLimiter | _InProcessLimiter | None = None
_redis_client = None


async def get_limiter():
    """Return a lazily-initialized limiter.

    If REDIS_URL is set, connect once and use Redis-backed limiter.
    Otherwise fall back to the in-process deque limiter (single-instance mode).
    """
    global _limiter, _redis_client
    if _limiter is not None:
        return _limiter

    if settings.redis_url:
        try:
            import redis.asyncio as aioredis

            _redis_client = aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
                max_connections=20,
            )
            await _redis_client.ping()
            _limiter = RedisLimiter(_redis_client)
            log.info("rate_limiter_initialized backend=redis")
            return _limiter
        except Exception:  # noqa: BLE001
            log.exception("redis_unavailable_falling_back_to_in_process")

    _limiter = _InProcessLimiter()
    log.info("rate_limiter_initialized backend=in-process")
    return _limiter


async def shutdown_limiter() -> None:
    """Close the Redis connection pool on app shutdown."""
    global _limiter, _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
    _limiter = None
    _redis_client = None
