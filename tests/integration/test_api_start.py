"""Integration tests for POST /research/start."""

import pytest


@pytest.mark.asyncio
async def test_start_returns_thread_id(async_client):
    resp = await async_client.post("/research/start", json={"topic": "LangGraph vs CrewAI"})
    assert resp.status_code == 200
    body = resp.json()
    assert "thread_id" in body
    assert len(body["thread_id"]) == 36  # uuid


@pytest.mark.asyncio
async def test_start_rejects_short_topic(async_client):
    resp = await async_client.post("/research/start", json={"topic": "hi"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_start_rejects_too_long_topic(async_client):
    resp = await async_client.post("/research/start", json={"topic": "x" * 501})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_start_rejects_missing_topic(async_client):
    resp = await async_client.post("/research/start", json={})
    assert resp.status_code == 422
