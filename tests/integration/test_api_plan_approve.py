"""Integration tests for GET /plan and POST /approve."""

import pytest


@pytest.mark.asyncio
async def test_plan_after_start(async_client):
    resp = await async_client.post("/research/start", json={"topic": "Test topic"})
    tid = resp.json()["thread_id"]
    plan = await async_client.get(f"/research/{tid}/plan")
    assert plan.status_code == 200
    body = plan.json()
    assert body["thread_id"] == tid
    assert body["plan"] == ["Section A", "Section B", "Section C"]


@pytest.mark.asyncio
async def test_plan_404_for_unknown_session(async_client):
    resp = await async_client.get("/research/00000000-0000-0000-0000-000000000000/plan")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_plan_400_malformed_id(async_client):
    resp = await async_client.get("/research/not-a-uuid/plan")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_approve_accepts_custom_plan(async_client):
    resp = await async_client.post("/research/start", json={"topic": "Topic X"})
    tid = resp.json()["thread_id"]
    approve = await async_client.post(
        f"/research/{tid}/approve",
        json={"plan": ["Custom section 1", "Custom section 2"]},
    )
    assert approve.status_code == 200


@pytest.mark.asyncio
async def test_approve_rejects_too_long_section(async_client):
    resp = await async_client.post("/research/start", json={"topic": "Topic Y"})
    tid = resp.json()["thread_id"]
    approve = await async_client.post(
        f"/research/{tid}/approve",
        json={"plan": ["x" * 201]},  # 201 chars > 200 cap
    )
    assert approve.status_code == 400


@pytest.mark.asyncio
async def test_approve_rejects_too_many_sections(async_client):
    resp = await async_client.post("/research/start", json={"topic": "Topic Z"})
    tid = resp.json()["thread_id"]
    approve = await async_client.post(
        f"/research/{tid}/approve",
        json={"plan": [f"s{i}" for i in range(11)]},  # 11 > 10 cap
    )
    assert approve.status_code == 422
