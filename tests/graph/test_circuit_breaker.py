"""Circuit breaker state machine tests (G-5)."""

from __future__ import annotations

import asyncio

import pytest

from app.tools.circuit_breaker import CircuitBreaker, CircuitBreakerOpen


def _mk(**kw) -> CircuitBreaker:
    defaults = {"name": "test", "failure_threshold": 3, "recovery_timeout_s": 60.0}
    defaults.update(kw)
    return CircuitBreaker(**defaults)


@pytest.mark.asyncio
async def test_closed_breaker_lets_calls_through():
    cb = _mk()

    async def ok():
        return "ok"

    assert await cb.call(ok) == "ok"
    assert cb.state == "closed"


@pytest.mark.asyncio
async def test_opens_after_threshold_failures():
    cb = _mk(failure_threshold=2)

    async def boom():
        raise RuntimeError("nope")

    # First failure → still closed
    with pytest.raises(RuntimeError):
        await cb.call(boom)
    assert cb.state == "closed"

    # Second failure → crosses threshold → open
    with pytest.raises(RuntimeError):
        await cb.call(boom)
    assert cb.state == "open"

    # Subsequent call fails fast with CircuitBreakerOpen (doesn't even call boom)
    called = {"count": 0}

    async def counting():
        called["count"] += 1
        return "never"

    with pytest.raises(CircuitBreakerOpen):
        await cb.call(counting)
    assert called["count"] == 0


@pytest.mark.asyncio
async def test_half_open_success_closes_breaker():
    cb = _mk(failure_threshold=1, recovery_timeout_s=0.01)

    async def boom():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await cb.call(boom)
    assert cb.state == "open"

    # Wait past the recovery window so the next call transitions to half_open
    await asyncio.sleep(0.02)

    async def ok():
        return "healed"

    # This call moves the state to half_open, then succeeds → closed.
    assert await cb.call(ok) == "healed"
    assert cb.state == "closed"


@pytest.mark.asyncio
async def test_half_open_failure_reopens_immediately():
    cb = _mk(failure_threshold=1, recovery_timeout_s=0.01)

    async def boom():
        raise RuntimeError("still broken")

    # Open.
    with pytest.raises(RuntimeError):
        await cb.call(boom)
    assert cb.state == "open"

    # Transition to half-open, then immediately fail → back to open.
    await asyncio.sleep(0.02)
    with pytest.raises(RuntimeError):
        await cb.call(boom)
    assert cb.state == "open"


@pytest.mark.asyncio
async def test_success_in_closed_resets_failure_counter():
    cb = _mk(failure_threshold=3)

    async def boom():
        raise RuntimeError("nope")

    async def ok():
        return "ok"

    # Two failures, then one success — counter resets, we can take two more before opening.
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(boom)
    await cb.call(ok)
    assert cb.state == "closed"

    # Two more failures should NOT open the breaker.
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(boom)
    assert cb.state == "closed"
