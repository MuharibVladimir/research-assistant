"""Semantic cache integration tests — /start short-circuits on cache hit."""

import pytest

from app.cache.semantic import CachedReport


@pytest.mark.asyncio
async def test_start_short_circuits_on_cache_hit(async_client, monkeypatch):
    hit = CachedReport(
        topic="LangGraph production patterns",
        final_report="# Cached report\n\nAll precomputed.",
        citations=[{"claim": "cached claim", "source_section": "S1", "score": 0.9}],
        similarity=0.97,
        cache_id="00000000-0000-0000-0000-00000000cafe",
    )
    monkeypatch.setattr(
        "app.api.routes.semantic_cache.lookup",
        lambda topic, api_key_hash=None: hit,
    )

    resp = await async_client.post(
        "/research/start", json={"topic": "LangGraph production patterns"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["cached"] is True
    assert "semantic cache" in body["message"].lower()

    # /result should return the cached report immediately — no /approve needed.
    result = await async_client.get(f"/research/{body['thread_id']}/result")
    assert result.status_code == 200
    assert "Cached report" in result.json()["final_report"]


@pytest.mark.asyncio
async def test_start_falls_through_on_cache_miss(async_client, monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.semantic_cache.lookup",
        lambda topic, api_key_hash=None: None,
    )
    resp = await async_client.post("/research/start", json={"topic": "some fresh topic"})
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("cached", False) is False
