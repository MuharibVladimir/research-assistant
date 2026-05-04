"""Speculative graph execution (G-14).

After the planner pauses at `await_approval`, most users approve without
edits within ~5-30 seconds — during which the server sits idle. We exploit
that gap by kicking off the researcher+writer portion of the graph in a
background task. If the user approves on time, the `/stream` path finds
the work already done (state is advanced) and the user-facing latency
collapses to near-zero.

Current implementation — `human_approved = True` on the live thread
----------------------------------------------------------------------
The speculative task flips `human_approved=True` on the user's own
checkpointer thread and drives the graph to completion. Correctness
guarantees:

  * **Checkpoints are at superstep boundaries** so speculative progress is
    safe to abandon mid-flight — the next normal resume picks up from
    wherever we got to.
  * **If `/approve` supplies an edited plan**, `approve_plan` cancels the
    speculative task BEFORE it commits the override, then overwrites `plan`
    on the same thread. The checkpointer rolls forward from the pre-approval
    snapshot with the new plan. Verified by
    `test_plan_override_during_interrupt_survives_gap`.
  * **If the speculator errors**, the user path runs the graph as normal
    and the user sees no visible effect.

Known weakness: we assumed approval. A user who takes 10s to read the plan
and then edits it has paid for wasted researcher tokens; cancellation stops
future work but doesn't refund what already ran. The cost-aware planner
bounds the waste to at most one full researcher pass.

Future work — the "shadow thread" design
-----------------------------------------
The current approach couples speculation to the user's live thread. That's
cheap but asymmetric: a plan edit invalidates the speculation even if
most of the sections are unchanged. A stronger design moves speculation
to a *separate* checkpointer thread so the user's state is untouched
until they decide.

Sketch:

  1. **Fork**. On planner completion, generate `shadow_id = f"{thread_id}:spec"`.
     Copy the live state into `shadow_id` via `aupdate_state`. Set
     `human_approved=True` on the shadow only.
  2. **Drive**. Run the graph on `shadow_id` in a background task. Shadow
     results (search_results, sections, final_report) land in the shadow
     thread's checkpoints, never in the user's.
  3. **Reconcile on /approve**. Compute `plan_hash = sha256(sorted(plan))`
     for both the user's approved plan and the shadow's plan. Three cases:
       * Hashes match → promote: `aget_state(shadow_id)` → `aupdate_state(
         thread_id, shadow.values)`. User sees instant result.
       * Hashes differ but overlap → partial promote: copy matching
         sections from shadow, re-research only the diff. Needs an
         explicit "sections diff" step in state.
       * Hashes disjoint → discard shadow, fall through to normal flow.
  4. **GC**. Shadow threads expire in the checkpointer after N hours;
     a background sweep prunes them.

Why not implemented today:
  * LangGraph's checkpointer doesn't expose a "copy thread" primitive; we'd
    build it ourselves via `aget_state_history` → batch-write.
  * Partial-promote requires sections-level diffing the current state
    schema doesn't carry (writer sees per-section search_results, but
    the review/revision loop couples them).
  * The cancellation path covers the common case well enough that the
    engineering budget is better spent elsewhere (see growth plan).

If/when we turn this on, the interface here (`REGISTRY.register` / `cancel`
/ `wait_if_any`) is shaped so the shadow-promote logic bolts on without
touching callers.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

log = logging.getLogger(__name__)


class _SpeculativeRegistry:
    """Thread_id → running background Task."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def register(self, thread_id: str, task: asyncio.Task) -> None:
        async with self._lock:
            existing = self._tasks.pop(thread_id, None)
            if existing is not None and not existing.done():
                existing.cancel()
            self._tasks[thread_id] = task

    async def cancel(self, thread_id: str) -> None:
        async with self._lock:
            task = self._tasks.pop(thread_id, None)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def wait_if_any(self, thread_id: str, timeout: float = 0.0) -> None:
        """Await an existing speculative task (if any) up to `timeout` seconds."""
        async with self._lock:
            task = self._tasks.get(thread_id)
        if task is None or task.done():
            return
        if timeout <= 0:
            return
        with contextlib.suppress(TimeoutError, Exception):
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)

    def clear(self) -> None:
        """Cancel all outstanding tasks. Called on shutdown."""
        for t in self._tasks.values():
            t.cancel()
        self._tasks.clear()


REGISTRY = _SpeculativeRegistry()
