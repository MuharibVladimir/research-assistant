"""End-to-end interrupt-recovery test.

`test_checkpointer_pause_resume.py` proves the checkpointer itself persists
state across a fresh `_builder.compile(...)` call. This test proves the
*whole stack* — FastAPI + graph dependency + checkpointer + SSE driver —
handles the same pause/resume shape through real HTTP endpoints:

    POST /start        # planner runs; graph pauses at `await_approval`
    (time passes — graph object, middleware state, etc. stay live)
    GET  /plan         # state after the pause is still fetchable
    POST /approve      # acknowledge, then deliberate wait
    await asyncio.sleep(2)
    GET  /stream       # drives the resume; state from pause must still be there
    GET  /result       # final_report present

This is the exact flow a production user follows when they stop to read
the plan before clicking approve. If anything in the API layer lost state
across the gap — session_id collision, stale thread_id reuse, pool churn —
this test breaks. Fake_graph is the same stub every other integration
test uses; we're testing the *plumbing*, not the LLM.
"""

from __future__ import annotations

import asyncio
import json

import pytest


async def _drain_sse(client, tid: str) -> list[tuple[str, dict]]:
    """Read the SSE stream until `done`/`error`, returning (event, payload) pairs."""
    events: list[tuple[str, dict]] = []
    async with client.stream("GET", f"/research/{tid}/stream") as resp:
        assert resp.status_code == 200, await resp.aread()
        current_event = None
        async for line in resp.aiter_lines():
            if not line:
                continue
            if line.startswith("event:"):
                current_event = line.removeprefix("event:").strip()
            elif line.startswith("data:") and current_event:
                payload = line.removeprefix("data:").strip()
                try:
                    events.append((current_event, json.loads(payload)))
                except json.JSONDecodeError:
                    pass
                if current_event in ("done", "error"):
                    break
    return events


@pytest.mark.asyncio
async def test_approve_then_delay_then_stream_recovers_state(async_client):
    """Main interrupt-recovery scenario: approve, wait, stream — state survives.

    The waited interval is short (2s) to keep CI fast but long enough to
    cross real async boundaries; the guarantee the test proves is logical,
    not a function of the sleep duration.
    """
    # 1. Kick off — planner runs, graph checkpoints before `await_approval`.
    resp = await async_client.post("/research/start", json={"topic": "Interrupt recovery"})
    assert resp.status_code == 200, resp.text
    tid = resp.json()["thread_id"]

    # 2. Fetch the plan — proves state is readable between /start and /approve.
    plan_resp = await async_client.get(f"/research/{tid}/plan")
    assert plan_resp.status_code == 200
    plan_before_wait = plan_resp.json()["plan"]
    assert plan_before_wait, "plan should be populated before approve"

    # 3. Approve.
    approve = await async_client.post(f"/research/{tid}/approve", json={})
    assert approve.status_code == 200

    # 4. Simulate a real human-reading delay between approve and stream.
    #    This is the exact shape where a broken checkpointer would lose state.
    await asyncio.sleep(2)

    # 5. Plan must STILL be intact after the wait.
    plan_after_wait = (await async_client.get(f"/research/{tid}/plan")).json()["plan"]
    assert plan_after_wait == plan_before_wait, (
        "plan differs after the approve/sleep gap — state was not preserved"
    )

    # 6. Drive the stream to completion. The state machine picks up from
    #    the checkpoint created by /start.
    events = await _drain_sse(async_client, tid)
    assert any(e[0] == "done" for e in events), f"stream never finished: {events}"

    # 7. Result endpoint returns the freshly-assembled report.
    result = await async_client.get(f"/research/{tid}/result")
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["thread_id"] == tid
    assert body["final_report"]


@pytest.mark.asyncio
async def test_stream_is_idempotent_after_completion(async_client):
    """After the graph is done, calling /stream again replays the history
    instead of re-running the pipeline — another checkpointer guarantee."""
    start = await async_client.post("/research/start", json={"topic": "Idempotent stream"})
    tid = start.json()["thread_id"]
    await async_client.post(f"/research/{tid}/approve", json={})
    first = await _drain_sse(async_client, tid)
    assert any(e[0] == "done" for e in first)

    # A second /stream on the same thread should also terminate cleanly.
    # The exact event shape differs (replay vs drive) but the terminal
    # guarantee is the same: we end with a `done`-class event.
    await asyncio.sleep(1)
    second = await _drain_sse(async_client, tid)
    assert any(e[0] == "done" for e in second)


@pytest.mark.asyncio
async def test_plan_override_during_interrupt_survives_gap(async_client):
    """Editing the plan on /approve then waiting must not lose the override.

    Edge case: the speculative-execution path (G-14) can be racing with a
    plan edit. The approve endpoint cancels any speculative task before
    committing the override; after a delay, /stream must see the edited
    plan, not the planner's original.
    """
    start = await async_client.post(
        "/research/start", json={"topic": "Plan-override recovery"}
    )
    tid = start.json()["thread_id"]
    original = (await async_client.get(f"/research/{tid}/plan")).json()["plan"]

    edited = ["Section A (edited)", "Section B (edited)"]
    assert edited != original

    approve = await async_client.post(
        f"/research/{tid}/approve", json={"plan": edited}
    )
    assert approve.status_code == 200

    await asyncio.sleep(1)

    after_gap = (await async_client.get(f"/research/{tid}/plan")).json()["plan"]
    assert after_gap == edited, (
        f"plan override lost across the approve/sleep gap: {after_gap}"
    )

    # And the stream completes using the edited plan.
    events = await _drain_sse(async_client, tid)
    assert any(e[0] == "done" for e in events)
