"""Minimal circuit breaker (G-5).

Three-state pattern (closed / open / half-open):

  * **closed**    — calls flow through. Every failure increments a counter;
                    on reaching `failure_threshold` the breaker opens.
  * **open**      — calls fail fast with `CircuitBreakerOpen`. After
                    `recovery_timeout_s` the breaker moves to half-open.
  * **half-open** — the next single call is allowed through as a probe. If
                    it succeeds the breaker closes; if it fails, the breaker
                    re-opens for another `recovery_timeout_s`.

Why not `pybreaker`? It pulls six dependencies for fifty lines of logic
we can write cleanly ourselves. This version is async-safe (asyncio.Lock
around state transitions) and emits Prometheus-ready state changes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")
State = Literal["closed", "open", "half_open"]


class CircuitBreakerOpen(Exception):
    """Raised when the breaker is open and a call arrives."""


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 5
    recovery_timeout_s: float = 120.0

    state: State = "closed"
    _consecutive_failures: int = 0
    _opened_at: float = 0.0
    _lock: asyncio.Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._lock = asyncio.Lock()

    async def _transition(self, new: State) -> None:
        if self.state != new:
            log.warning("circuit_breaker name=%s %s → %s", self.name, self.state, new)
            self.state = new
            from app.observability import CIRCUIT_BREAKER_STATE

            CIRCUIT_BREAKER_STATE.labels(name=self.name).set(
                {"closed": 0, "half_open": 1, "open": 2}[new]
            )

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        """Invoke `fn()` through the breaker. Raises CircuitBreakerOpen if open."""
        async with self._lock:
            if self.state == "open":
                if time.monotonic() - self._opened_at >= self.recovery_timeout_s:
                    await self._transition("half_open")
                else:
                    raise CircuitBreakerOpen(f"{self.name} open")

        try:
            result = await fn()
        except Exception as exc:  # noqa: BLE001
            async with self._lock:
                self._consecutive_failures += 1
                if (
                    self.state == "half_open"
                    or self._consecutive_failures >= self.failure_threshold
                ):
                    self._opened_at = time.monotonic()
                    await self._transition("open")
            raise exc

        async with self._lock:
            self._consecutive_failures = 0
            if self.state != "closed":
                await self._transition("closed")
        return result
