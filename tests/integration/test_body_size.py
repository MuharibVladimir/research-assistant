"""Body-size middleware tests."""

import pytest

from app.config import settings


@pytest.mark.asyncio
async def test_rejects_body_over_limit(async_client, monkeypatch):
    monkeypatch.setattr(settings, "max_request_body_bytes", 100)
    # Content-Length sent by httpx is the actual serialized length
    big = "x" * 500
    resp = await async_client.post("/research/start", json={"topic": big})
    assert resp.status_code == 413
    assert "too large" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_accepts_body_under_limit(async_client, monkeypatch):
    monkeypatch.setattr(settings, "max_request_body_bytes", 10_000)
    resp = await async_client.post("/research/start", json={"topic": "small topic"})
    assert resp.status_code == 200
