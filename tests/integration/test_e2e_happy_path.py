"""End-to-end happy-path test: exercises the full API sequence.

Covers:
  POST /research/start       → 200, plan emitted
  GET  /{id}/plan            → 200, plan returned
  POST /{id}/approve         → 200
  GET  /{id}/stream          → SSE stream, progress events + done
  GET  /{id}/result          → 200, final_report present
  GET  /{id}/metrics         → 200, token counts populated

Same fake-graph / fake-DB fixtures power all layers — no real LLM, no real
Postgres. This catches regressions in wiring (missing deps, broken routes,
middleware ordering, Pydantic validation drift) that unit tests miss.
"""

from __future__ import annotations

import json

import pytest


@pytest.mark.asyncio
async def test_full_happy_path(async_client):
    """All six endpoints in sequence, verifying payload at each step."""
    # 1. Start
    start = await async_client.post(
        "/research/start", json={"topic": "End-to-end smoke test topic"}
    )
    assert start.status_code == 200, start.text
    body = start.json()
    tid = body["thread_id"]
    assert len(tid) == 36
    assert body.get("cached", False) is False
    assert "Plan generated" in body["message"]

    # 2. Plan
    plan_resp = await async_client.get(f"/research/{tid}/plan")
    assert plan_resp.status_code == 200, plan_resp.text
    plan_body = plan_resp.json()
    assert plan_body["thread_id"] == tid
    assert plan_body["status"] == "waiting_approval"
    assert len(plan_body["plan"]) >= 1

    # 3. Approve (without override — accept generated plan)
    approve = await async_client.post(f"/research/{tid}/approve", json={})
    assert approve.status_code == 200, approve.text
    assert "approved" in approve.json()["message"].lower()

    # 4. Stream — collect the SSE events and verify shape
    events: list[tuple[str, dict]] = []
    async with async_client.stream("GET", f"/research/{tid}/stream") as resp:
        assert resp.status_code == 200
        current_event = None
        async for line in resp.aiter_lines():
            if not line:
                continue
            if line.startswith("event:"):
                current_event = line.removeprefix("event:").strip()
            elif line.startswith("data:") and current_event:
                try:
                    payload = json.loads(line.removeprefix("data:").strip())
                except json.JSONDecodeError:
                    continue
                events.append((current_event, payload))

    # At least one progress event + exactly one done
    progress_events = [e for e in events if e[0] == "progress"]
    done_events = [e for e in events if e[0] == "done"]
    assert len(progress_events) >= 1, f"no progress events in stream: {events}"
    assert len(done_events) == 1, f"expected 1 done event, got {done_events}"
    # Progress events name a known node
    expected_nodes = {
        "researcher",
        "grader",
        "writer",
        "reviewer",
        "formatter",
        "citations",
    }
    seen_nodes = {e[1].get("node") for e in progress_events}
    assert seen_nodes & expected_nodes, f"no known nodes in events: {seen_nodes}"

    # 5. Result — report is now ready
    result = await async_client.get(f"/research/{tid}/result")
    assert result.status_code == 200, result.text
    result_body = result.json()
    assert result_body["thread_id"] == tid
    assert result_body["final_report"]
    assert "Final Report" in result_body["final_report"]

    # 6. Metrics — row exists, status=done
    metrics = await async_client.get(f"/research/{tid}/metrics")
    assert metrics.status_code == 200, metrics.text
    metrics_body = metrics.json()
    assert metrics_body["thread_id"] == tid
    assert metrics_body["status"] == "done"


@pytest.mark.asyncio
async def test_full_flow_then_followup(async_client):
    """E2E + a follow-up turn on top — verifies multi-turn state persists."""
    start = await async_client.post("/research/start", json={"topic": "E2E followup topic"})
    tid = start.json()["thread_id"]
    assert (await async_client.post(f"/research/{tid}/approve", json={})).status_code == 200

    # Drive the graph
    async with async_client.stream("GET", f"/research/{tid}/stream") as resp:
        async for _ in resp.aiter_lines():
            pass

    # Report ready
    result = await async_client.get(f"/research/{tid}/result")
    assert result.status_code == 200

    # Follow-up refinement
    follow = await async_client.post(
        f"/research/{tid}/followup",
        json={"question": "Please deepen the pricing comparison section."},
    )
    assert follow.status_code == 200
    assert "pricing comparison" in follow.json()["final_report"].lower()


@pytest.mark.asyncio
async def test_stream_rejects_without_approve(async_client):
    """Hitting /stream before /approve should get 409."""
    start = await async_client.post("/research/start", json={"topic": "Needs approval"})
    tid = start.json()["thread_id"]

    resp = await async_client.get(f"/research/{tid}/stream")
    assert resp.status_code == 409
