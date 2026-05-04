"""Request dedup tests (G-4).

Embedding is stubbed so similarity decisions are deterministic without OpenAI.
"""

from __future__ import annotations

import asyncio

import pytest

from app.cache import dedup as dedup_mod


@pytest.fixture(autouse=True)
def _reset_dedup():
    dedup_mod.reset_for_tests()
    yield
    dedup_mod.reset_for_tests()


def _make_embed(table: dict[str, list[float]]):
    def _embed(text: str) -> list[float]:
        return table[text]

    return _embed


@pytest.mark.asyncio
async def test_leader_runs_alone_when_no_match(monkeypatch):
    monkeypatch.setattr(dedup_mod, "embed_text", _make_embed({"topic-a": [1.0, 0.0]}))

    future, is_leader = await dedup_mod.claim_or_wait("topic-a", caller_hash="u1")
    assert is_leader is True
    assert not future.done()


@pytest.mark.asyncio
async def test_duplicate_within_window_waits_for_leader(monkeypatch):
    """Two concurrent claims with the same embedding — first is leader, second waits."""
    monkeypatch.setattr(
        dedup_mod,
        "embed_text",
        _make_embed({"topic-a": [1.0, 0.0], "topic-a-again": [1.0, 0.0]}),
    )

    leader_future, is_leader1 = await dedup_mod.claim_or_wait("topic-a", caller_hash="u1")
    follower_future, is_leader2 = await dedup_mod.claim_or_wait("topic-a-again", caller_hash="u1")

    assert is_leader1 is True
    assert is_leader2 is False
    # Both callers share the same future.
    assert leader_future is follower_future


@pytest.mark.asyncio
async def test_different_callers_do_not_share_entries(monkeypatch):
    """Dedup is scoped by api_key_hash (C-3 consistency)."""
    monkeypatch.setattr(
        dedup_mod,
        "embed_text",
        _make_embed({"topic-a": [1.0, 0.0], "topic-same": [1.0, 0.0]}),
    )

    _, leader_u1 = await dedup_mod.claim_or_wait("topic-a", caller_hash="u1")
    _, leader_u2 = await dedup_mod.claim_or_wait("topic-same", caller_hash="u2")

    assert leader_u1 is True
    assert leader_u2 is True  # different user → different entry


@pytest.mark.asyncio
async def test_below_threshold_does_not_match(monkeypatch):
    """Orthogonal embeddings → no dedup hit."""
    monkeypatch.setattr(
        dedup_mod,
        "embed_text",
        _make_embed({"topic-a": [1.0, 0.0], "topic-b": [0.0, 1.0]}),
    )

    _, leader_a = await dedup_mod.claim_or_wait("topic-a", caller_hash="u1")
    _, leader_b = await dedup_mod.claim_or_wait("topic-b", caller_hash="u1")

    assert leader_a is True
    assert leader_b is True


@pytest.mark.asyncio
async def test_release_broadcasts_to_waiters(monkeypatch):
    monkeypatch.setattr(
        dedup_mod,
        "embed_text",
        _make_embed({"topic-a": [1.0, 0.0], "topic-a-copy": [1.0, 0.0]}),
    )

    leader_future, _ = await dedup_mod.claim_or_wait("topic-a", caller_hash="u1")
    follower_future, _ = await dedup_mod.claim_or_wait("topic-a-copy", caller_hash="u1")

    await dedup_mod.release(leader_future, "resolved-thread")
    # Follower awaits and sees the same result.
    assert await asyncio.wait_for(follower_future, timeout=0.1) == "resolved-thread"


@pytest.mark.asyncio
async def test_fail_propagates_to_waiters(monkeypatch):
    monkeypatch.setattr(
        dedup_mod,
        "embed_text",
        _make_embed({"topic-a": [1.0, 0.0], "topic-a-copy": [1.0, 0.0]}),
    )

    leader_future, _ = await dedup_mod.claim_or_wait("topic-a", caller_hash="u1")
    follower_future, _ = await dedup_mod.claim_or_wait("topic-a-copy", caller_hash="u1")

    await dedup_mod.fail(leader_future, RuntimeError("leader died"))
    with pytest.raises(RuntimeError, match="leader died"):
        await asyncio.wait_for(follower_future, timeout=0.1)


@pytest.mark.asyncio
async def test_expired_entries_are_gced(monkeypatch):
    """An entry older than `dedup_window_seconds` is evicted on the next claim."""
    from app.config import settings

    monkeypatch.setattr(
        dedup_mod,
        "embed_text",
        _make_embed({"topic-a": [1.0, 0.0], "topic-a-2": [1.0, 0.0]}),
    )
    monkeypatch.setattr(settings, "dedup_window_seconds", 0.01)

    _, is_leader1 = await dedup_mod.claim_or_wait("topic-a", caller_hash="u1")
    assert is_leader1 is True

    await asyncio.sleep(0.05)  # let the entry age out

    _, is_leader2 = await dedup_mod.claim_or_wait("topic-a-2", caller_hash="u1")
    # After expiry, the previous entry is GC'd and this call becomes a new leader.
    assert is_leader2 is True
