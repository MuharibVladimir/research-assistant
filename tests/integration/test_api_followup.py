"""Multi-turn follow-up endpoint."""

import pytest


async def _run_full_research(client) -> str:
    """Helper: start a session, approve, drive to final_report, return thread_id."""
    start = await client.post("/research/start", json={"topic": "multi-turn test topic"})
    assert start.status_code == 200
    tid = start.json()["thread_id"]
    approve = await client.post(f"/research/{tid}/approve", json={})
    assert approve.status_code == 200
    # Hit /stream to drive the pipeline to completion
    async with client.stream("GET", f"/research/{tid}/stream") as resp:
        async for _ in resp.aiter_lines():
            pass
    return tid


@pytest.mark.asyncio
async def test_followup_refines_report(async_client):
    tid = await _run_full_research(async_client)

    follow = await async_client.post(
        f"/research/{tid}/followup",
        json={"question": "Can you add a section on cost?"},
    )
    assert follow.status_code == 200
    body = follow.json()
    assert body["thread_id"] == tid
    assert "Refined" in body["final_report"]
    assert "Can you add a section on cost?" in body["final_report"]


@pytest.mark.asyncio
async def test_followup_409_when_no_report_yet(async_client):
    start = await async_client.post("/research/start", json={"topic": "unfinished topic"})
    tid = start.json()["thread_id"]
    # no approve/stream → final_report is empty
    resp = await async_client.post(
        f"/research/{tid}/followup",
        json={"question": "refine?"},
    )
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_followup_rejects_short_question(async_client):
    tid = await _run_full_research(async_client)
    resp = await async_client.post(f"/research/{tid}/followup", json={"question": "ok"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_followup_persists_message_history(async_client):
    """Two consecutive follow-ups should both see the prior messages in state."""
    tid = await _run_full_research(async_client)

    r1 = await async_client.post(f"/research/{tid}/followup", json={"question": "First follow-up"})
    assert r1.status_code == 200
    assert "First follow-up" in r1.json()["final_report"]

    r2 = await async_client.post(f"/research/{tid}/followup", json={"question": "Second follow-up"})
    assert r2.status_code == 200
    assert "Second follow-up" in r2.json()["final_report"]
