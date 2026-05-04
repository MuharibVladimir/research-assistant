"""Integration tests for GET /result and GET /metrics."""

import pytest


@pytest.mark.asyncio
async def test_result_before_ready_returns_202(async_client):
    resp = await async_client.post("/research/start", json={"topic": "Ready test"})
    tid = resp.json()["thread_id"]
    # Haven't driven the graph yet
    result = await async_client.get(f"/research/{tid}/result")
    assert result.status_code == 202


@pytest.mark.asyncio
async def test_metrics_returns_zero_before_approval(async_client):
    resp = await async_client.post("/research/start", json={"topic": "Metrics test"})
    tid = resp.json()["thread_id"]
    m = await async_client.get(f"/research/{tid}/metrics")
    assert m.status_code == 200
    body = m.json()
    assert body["thread_id"] == tid
    assert body["total_tokens"] == 0
    assert body["cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_metrics_404_for_unknown(async_client):
    r = await async_client.get("/research/00000000-0000-0000-0000-000000000000/metrics")
    assert r.status_code == 404
