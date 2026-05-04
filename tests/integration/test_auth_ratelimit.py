"""Auth + rate-limit tests.

Requires setting RESEARCH_API_KEY so auth is active.
"""

import asyncio

import pytest

from app.config import settings


@pytest.mark.asyncio
async def test_auth_disabled_when_key_empty(async_client, monkeypatch):
    """When research_api_key is empty, no X-API-Key header is needed."""
    monkeypatch.setattr(settings, "research_api_key", "")
    resp = await async_client.post("/research/start", json={"topic": "open topic"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_missing_api_key_returns_401(async_client, monkeypatch):
    monkeypatch.setattr(settings, "research_api_key", "secret")
    resp = await async_client.post("/research/start", json={"topic": "auth test"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_wrong_api_key_returns_401(async_client, monkeypatch):
    monkeypatch.setattr(settings, "research_api_key", "secret")
    resp = await async_client.post(
        "/research/start",
        json={"topic": "auth test"},
        headers={"X-API-Key": "wrong"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_correct_api_key_passes(async_client, monkeypatch):
    monkeypatch.setattr(settings, "research_api_key", "secret")
    resp = await async_client.post(
        "/research/start",
        json={"topic": "auth test"},
        headers={"X-API-Key": "secret"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_rate_limit_returns_429(async_client, monkeypatch):
    """After rate_limit_per_minute requests, subsequent ones get 429."""
    monkeypatch.setattr(settings, "research_api_key", "secret")
    monkeypatch.setattr(settings, "rate_limit_per_minute", 3)
    headers = {"X-API-Key": "secret"}

    for i in range(3):
        resp = await async_client.post(
            "/research/start", json={"topic": f"ratelimit {i}"}, headers=headers
        )
        assert resp.status_code == 200

    # 4th call — over the limit
    resp = await async_client.post("/research/start", json={"topic": "over limit"}, headers=headers)
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers


@pytest.mark.asyncio
async def test_access_control_blocks_other_api_key(async_client, monkeypatch):
    """A session created by one key is invisible to another."""
    monkeypatch.setattr(settings, "research_api_key", "secret-a")

    # Key A creates
    await asyncio.sleep(0)  # ensure event loop schedules
    resp = await async_client.post(
        "/research/start",
        json={"topic": "owned by A"},
        headers={"X-API-Key": "secret-a"},
    )
    tid = resp.json()["thread_id"]

    # Key B tries to read
    monkeypatch.setattr(settings, "research_api_key", "secret-b")
    r = await async_client.get(f"/research/{tid}/plan", headers={"X-API-Key": "secret-b"})
    assert r.status_code == 403
