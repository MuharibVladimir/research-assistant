"""FastAPI routes for the Research Assistant.

Auth:
    All /research endpoints require `X-API-Key` header matching
    `settings.research_api_key`. If the setting is empty, auth is disabled
    (useful for local dev).

Access control:
    Every session is tagged with the SHA-256 hash of the API key that created
    it (column `research_sessions.api_key_hash`). Subsequent requests for the
    same thread_id must come from the same key — otherwise 403.

Rate-limit:
    In-memory per-API-key sliding window. Size: `settings.rate_limit_per_minute`
    requests per 60s. Applied to /start and /approve. The bucket dict is
    garbage-collected periodically by the background cleanup task (lifespan).

Streaming:
    POST /approve is lightweight — it marks `human_approved=True` and returns.
    The graph is driven by the subsequent GET /stream call, which uses
    `graph.astream(..., stream_mode="updates")` to emit real-time node updates.

Endpoints:
    POST /research/start            — Start a new research session
    GET  /research/{id}/plan        — Get the generated plan
    POST /research/{id}/approve     — Approve the plan (does NOT run graph)
    GET  /research/{id}/stream      — Drives the graph + SSE stream of node updates
    GET  /research/{id}/result      — Get the final report
    GET  /research/{id}/metrics     — Get token usage and cost
"""

import asyncio
import hashlib
import hmac
import json
import logging
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.cache import semantic as semantic_cache
from app.config import settings
from app.errors import (
    Forbidden,
    GraphTimeout,
    MalformedThreadId,
    PlanNotReady,
    ResearchNotFound,
    Unauthorized,
)
from app.graph.callbacks import UsageCallback
from app.graph.graph import (
    build_graph,
    build_refine_graph,
    close_pool,
    create_postgres_checkpointer,
)
from app.models.db import ResearchSession
from app.models.engine import SessionLocal
from app.rate_limit import get_limiter

log = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Graph bootstrap (single instance, lazily created)
# ---------------------------------------------------------------------------

_graph = None
_refine_graph = None
_checkpointer = None
_graph_lock = asyncio.Lock()


async def get_graph():
    """Lazy-initialize the compiled graph + pooled checkpointer.

    asyncio.Lock prevents a race when two concurrent requests trigger
    first-time initialization simultaneously.
    """
    global _graph, _checkpointer, _refine_graph
    if _graph is not None:
        return _graph
    async with _graph_lock:
        if _graph is None:
            _checkpointer = await create_postgres_checkpointer()
            _graph = build_graph(checkpointer=_checkpointer)
            _refine_graph = build_refine_graph(checkpointer=_checkpointer)
    return _graph


async def get_refine_graph():
    """Return the follow-up graph, initializing shared checkpointer if needed."""
    global _refine_graph
    if _refine_graph is None:
        await get_graph()
    return _refine_graph


async def shutdown_graph() -> None:
    """Close the checkpointer connection pool on app shutdown."""
    await close_pool()


# ---------------------------------------------------------------------------
# Auth + rate-limit + access control
# ---------------------------------------------------------------------------

_ANONYMOUS = "anonymous"  # used when research_api_key is empty (dev mode)


def _hash_key(api_key: str) -> str:
    """SHA-256 hex of the API key — stored in DB for access-control checks."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """Validate X-API-Key header. Returns the key (or `_ANONYMOUS` in dev mode).

    `hmac.compare_digest` is constant-time so an attacker can't brute-force
    the key one character at a time by measuring response latency (C-1).
    """
    expected = settings.research_api_key
    if not expected:
        return _ANONYMOUS
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise Unauthorized()
    return x_api_key


async def enforce_rate_limit(
    request: Request,
    api_key: str = Depends(verify_api_key),
) -> str:
    """Per-key sliding-window rate limit, with a secondary per-IP bucket (H-3).

    Per-key alone is bypassable by rotating API keys. The secondary per-IP
    bucket is 5× the per-key ceiling so legitimate users with shared IPs
    (office, mobile carrier NAT) aren't blocked.
    """
    limiter = await get_limiter()

    admitted, retry_after = await limiter.check(
        key=api_key,
        limit=settings.rate_limit_per_minute,
        window=60.0,
    )
    if not admitted:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit: {settings.rate_limit_per_minute}/min. Retry in {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )

    client_ip = request.client.host if request.client else "unknown"
    admitted_ip, retry_after_ip = await limiter.check(
        key=f"ip:{client_ip}",
        limit=settings.rate_limit_per_minute * 5,
        window=60.0,
    )
    if not admitted_ip:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Per-IP rate limit ({settings.rate_limit_per_minute * 5}/min) "
                f"hit. Retry in {retry_after_ip}s."
            ),
            headers={"Retry-After": str(retry_after_ip)},
        )
    return api_key


async def rate_bucket_cleanup_task() -> None:
    """Periodically evict stale buckets (in-process limiter only).

    Runs every 5 minutes. Redis-backed limiter handles TTL natively via PEXPIRE.
    """
    try:
        while True:
            await asyncio.sleep(300)
            limiter = await get_limiter()
            evicted = await limiter.cleanup()
            log.debug("rate_bucket_cleanup evicted=%d", evicted)
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# Schemas — with validation constraints against abuse
# ---------------------------------------------------------------------------


class StartRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500)
    # G-7: optional USD ceiling. When set, the planner sees it and plans
    # for a number of sections that fits; the reviewer loop also respects it.
    budget_usd: float | None = Field(default=None, gt=0.0, le=100.0)


class StartResponse(BaseModel):
    thread_id: str
    message: str
    cached: bool = False  # True when the semantic cache short-circuited the graph


class PlanResponse(BaseModel):
    thread_id: str
    plan: list[str]
    status: str


class ApproveRequest(BaseModel):
    # Optional override: if present, each item max 200 chars, max 10 sections.
    plan: list[str] | None = Field(default=None, max_length=10)


class FollowupRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class FollowupResponse(BaseModel):
    thread_id: str
    final_report: str


class ResultResponse(BaseModel):
    thread_id: str
    final_report: str


class MetricsResponse(BaseModel):
    thread_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    status: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(thread_id: str, callbacks=None) -> dict:
    cfg: dict = {"configurable": {"thread_id": thread_id}}
    if callbacks:
        cfg["callbacks"] = callbacks
    return cfg


async def _get_state(thread_id: str) -> dict:
    graph = await get_graph()
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    return snapshot.values if snapshot else {}


def _parse_thread_id(thread_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(thread_id)
    except ValueError as e:
        raise MalformedThreadId() from e


def _load_session(thread_id: str) -> ResearchSession:
    """Load a session row or 404."""
    tid = _parse_thread_id(thread_id)
    with SessionLocal() as db:
        session = db.get(ResearchSession, tid)
    if session is None:
        raise ResearchNotFound()
    return session


def _check_ownership(session: ResearchSession, api_key: str) -> None:
    """Verify the session was created by this API key. 403 on mismatch.

    In dev mode (no research_api_key), ownership is skipped.
    """
    if api_key == _ANONYMOUS:
        return
    expected_hash = _hash_key(api_key)
    if session.api_key_hash and session.api_key_hash != expected_hash:
        raise Forbidden()


def _save_session(thread_id: str, topic: str, api_key: str, status_: str = "pending") -> None:
    """Upsert a ResearchSession row."""
    tid = _parse_thread_id(thread_id)
    api_key_hash = None if api_key == _ANONYMOUS else _hash_key(api_key)
    with SessionLocal() as db:
        session = db.get(ResearchSession, tid)
        if session is None:
            session = ResearchSession(
                id=tid, topic=topic, status=status_, api_key_hash=api_key_hash
            )
            db.add(session)
        else:
            session.status = status_
        db.commit()


def _update_metrics(thread_id: str, usage: dict, status_: str = "done") -> None:
    """Persist token usage to research_sessions."""
    tid = _parse_thread_id(thread_id)
    with SessionLocal() as db:
        session = db.get(ResearchSession, tid)
        if session:
            session.prompt_tokens = usage["prompt_tokens"]
            session.completion_tokens = usage["completion_tokens"]
            session.total_tokens = usage["total_tokens"]
            session.cost_usd = usage["cost_usd"]
            session.status = status_
            db.commit()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/start", response_model=StartResponse)
async def start_research(
    body: StartRequest,
    api_key: str = Depends(enforce_rate_limit),
) -> StartResponse:
    """Kick off a new research session.

    Runs the graph until the first interrupt (before `await_approval`) —
    i.e., right after the planner generates the outline. Returns the
    thread_id so the client can fetch the plan and approve it.
    """
    graph = await get_graph()
    thread_id = str(uuid.uuid4())
    cb = UsageCallback(thread_id=thread_id)
    config = _config(thread_id, callbacks=[cb])

    caller_hash_for_planner = None if api_key == _ANONYMOUS else _hash_key(api_key)

    # G-15: redact PII in the topic before anything gets embedded or logged.
    from app.tools.safety import redact_pii

    redacted_topic, pii_counts = redact_pii(body.topic)
    if pii_counts:
        log.info("pii_redacted_on_start counts=%s", pii_counts)

    initial_state = {
        "topic": redacted_topic,
        "thread_id": thread_id,
        "plan": [],
        "human_approved": False,
        "search_results": {},
        "sections": {},
        "review_feedback": "",
        "revision_count": 0,
        "final_report": "",
        # Seed `_caller_hash` into retrieval_grades so planner_node can pull
        # this user's episodic memory. Reducer is a dict-merge so the field
        # co-exists with actual per-section grades later.
        "retrieval_grades": (
            {"_caller_hash": caller_hash_for_planner} if caller_hash_for_planner else {}
        ),
        "budget_usd": body.budget_usd or 0.0,  # 0.0 → no budget enforcement
        "messages": [],
    }

    _save_session(thread_id, body.topic, api_key, status_="planning")

    # Short-circuit if the semantic cache has a near-identical topic.
    # We seed the graph's state with the cached report and return immediately,
    # treating it as auto-approved (no plan review needed — the report exists).
    # Lookup is scoped by api_key_hash (C-3) — anonymous callers only see
    # anonymous rows; authenticated callers only see their own rows.
    caller_hash = None if api_key == _ANONYMOUS else _hash_key(api_key)
    cached = await asyncio.to_thread(semantic_cache.lookup, body.topic, caller_hash)
    if cached is not None:
        log.info(
            "semantic_cache_hit topic=%s similarity=%.3f",
            body.topic,
            cached.similarity,
        )
        await graph.aupdate_state(
            {"configurable": {"thread_id": thread_id}},
            {
                "plan": ["(served from semantic cache)"],
                "human_approved": True,
                "final_report": cached.final_report,
                "citations": cached.citations,
            },
        )
        _update_metrics(thread_id, cb.usage, status_="done")
        return StartResponse(
            thread_id=thread_id,
            message=(
                f"Served from semantic cache (similarity {cached.similarity:.2f}). "
                f"GET /research/{{thread_id}}/result for the report."
            ),
            cached=True,
        )

    try:
        async with asyncio.timeout(settings.graph_timeout_seconds):
            # Stops at interrupt_before=["await_approval"] after planner runs
            await graph.ainvoke(initial_state, config)
    except TimeoutError as e:
        _update_metrics(thread_id, cb.usage, status_="error")
        raise GraphTimeout("Planner timed out.") from e

    _update_metrics(thread_id, cb.usage, status_="waiting_approval")

    # G-14: kick off the researcher+writer chain speculatively so that if the
    # user approves without editing, /stream finds the work already done.
    if settings.speculative_execution_enabled:
        from app.graph.speculative import REGISTRY

        async def _speculate() -> None:
            spec_cb = UsageCallback(thread_id=thread_id)
            spec_cfg = _config(thread_id, callbacks=[spec_cb])
            try:
                # Forcing approval on a *copy* of state is not trivial with
                # the shared checkpointer, so we simply set `human_approved=True`
                # and drive the graph to completion. If the user edits the plan,
                # /approve cancels this task before it finishes and applies the
                # edit on top of the pre-approval checkpoint.
                await graph.aupdate_state(
                    {"configurable": {"thread_id": thread_id}},
                    {"human_approved": True},
                )
                await graph.ainvoke(None, spec_cfg)
            except asyncio.CancelledError:
                log.info("speculative_cancelled thread_id=%s", thread_id)
                raise
            except Exception:  # noqa: BLE001
                log.exception("speculative_failed thread_id=%s", thread_id)

        await REGISTRY.register(thread_id, asyncio.create_task(_speculate()))

    return StartResponse(
        thread_id=thread_id,
        message="Plan generated. GET /research/{thread_id}/plan to review.",
    )


@router.get("/{thread_id}/plan", response_model=PlanResponse)
async def get_plan(
    thread_id: str,
    api_key: str = Depends(verify_api_key),
) -> PlanResponse:
    session = _load_session(thread_id)
    _check_ownership(session, api_key)

    state = await _get_state(thread_id)
    plan = state.get("plan", [])
    if not plan:
        raise PlanNotReady()
    return PlanResponse(thread_id=thread_id, plan=plan, status="waiting_approval")


@router.post("/{thread_id}/approve")
async def approve_plan(
    thread_id: str,
    body: ApproveRequest,
    api_key: str = Depends(enforce_rate_limit),
) -> dict:
    """Mark the plan approved. Does NOT run the graph.

    The client must GET /stream next to drive the graph to completion.
    """
    session = _load_session(thread_id)
    _check_ownership(session, api_key)

    state = await _get_state(thread_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session state missing")

    update: dict = {"human_approved": True}
    plan_edited = False
    if body.plan is not None:
        # Enforce per-item length bound
        if any(len(s) > 200 for s in body.plan):
            raise HTTPException(status_code=400, detail="Section title too long (max 200)")
        update["plan"] = body.plan
        plan_edited = True

    # G-14: if the user edited the plan, the speculative task was running with
    # the old plan — cancel it so /stream doesn't return stale results.
    if plan_edited and settings.speculative_execution_enabled:
        from app.graph.speculative import REGISTRY

        await REGISTRY.cancel(thread_id)

    _save_session(thread_id, state.get("topic", ""), api_key, status_="researching")

    graph = await get_graph()
    await graph.aupdate_state(_config(thread_id), update)
    return {
        "thread_id": thread_id,
        "message": "Plan approved. GET /research/{thread_id}/stream to run it.",
    }


@router.get("/{thread_id}/stream")
async def stream_progress(
    thread_id: str,
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    """Drive the graph and stream real-time progress via SSE.

    If the graph is already complete, replays checkpoint history.
    Otherwise drives the graph with `astream(stream_mode="updates")`, wrapped
    in `asyncio.timeout(GRAPH_TIMEOUT_SECONDS)`.
    """
    session = _load_session(thread_id)
    _check_ownership(session, api_key)

    graph = await get_graph()
    config = _config(thread_id)
    state = await _get_state(thread_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session state missing")

    if state.get("final_report"):
        return EventSourceResponse(_replay_history(graph, config, request))

    if not state.get("human_approved"):
        raise HTTPException(
            status_code=409,
            detail="Approve the plan first via POST /{thread_id}/approve.",
        )

    cb = UsageCallback(thread_id=thread_id)
    drive_config = _config(thread_id, callbacks=[cb])
    caller_hash = None if api_key == _ANONYMOUS else _hash_key(api_key)

    async def _driver() -> AsyncGenerator[dict]:
        """Drive the graph via astream_events and surface three event kinds:

        - `progress` — fired at each node boundary (start/end)
        - `token`    — fired for every LLM delta chunk inside writer/formatter
                       (lets the UI print the report as it's being written)
        - `done`/`error` — terminal
        """
        try:
            async with asyncio.timeout(settings.graph_timeout_seconds):
                # H-6 idle timeout: if no event fires for `sse_idle_timeout_seconds`,
                # close the stream — prevents slow-loris SSE holding pool connections.
                loop = asyncio.get_event_loop()
                last_event_at = loop.time()
                idle_limit = settings.sse_idle_timeout_seconds
                async for event in graph.astream_events(None, drive_config, version="v2"):
                    now = loop.time()
                    if now - last_event_at > idle_limit:
                        log.warning(
                            "sse_idle_timeout thread_id=%s idle_s=%.1f",
                            thread_id,
                            now - last_event_at,
                        )
                        break
                    last_event_at = now

                    if await request.is_disconnected():
                        break

                    kind = event.get("event")
                    name = event.get("name", "")
                    tags = event.get("tags") or []

                    # 1) Node boundaries
                    if kind == "on_chain_start" and name in _PROGRESS_NODES:
                        yield {
                            "event": "progress",
                            "data": json.dumps({"node": name, "phase": "start"}),
                        }
                    elif kind == "on_chain_end" and name in _PROGRESS_NODES:
                        output = event.get("data", {}).get("output") or {}
                        yield {
                            "event": "progress",
                            "data": json.dumps(
                                {
                                    "node": name,
                                    "phase": "end",
                                    "sections_written": (
                                        list(output.get("sections", {}).keys())
                                        if isinstance(output, dict) and output.get("sections")
                                        else []
                                    ),
                                }
                            ),
                        }

                    # 2) LLM token stream — only emit for writer & formatter so
                    # clients can render the report incrementally. Grader/reviewer
                    # tokens are noise.
                    elif kind == "on_chat_model_stream" and any(
                        t in _TOKEN_STREAM_TAGS for t in tags
                    ):
                        chunk = event["data"].get("chunk")
                        text = getattr(chunk, "content", None) if chunk else None
                        if text:
                            yield {
                                "event": "token",
                                "data": json.dumps({"delta": text}),
                            }

            _update_metrics(thread_id, cb.usage, status_="done")

            # Persist the finished report to the semantic cache so next time
            # a near-identical topic short-circuits the whole graph. Tag the
            # row with the caller's api_key_hash so C-3 cross-user leakage
            # is impossible on read-back.
            final_state = await _get_state(thread_id)
            final_report = final_state.get("final_report") if final_state else None
            if final_report:
                try:
                    await asyncio.to_thread(
                        semantic_cache.store,
                        final_state.get("topic", ""),
                        final_report,
                        final_state.get("citations", []),
                        caller_hash,
                    )
                except Exception:  # noqa: BLE001
                    log.exception("semantic_cache_store_failed")
                # G-12: also record an episodic-memory entry per user so
                # future topics from the same caller can reference prior work.
                if caller_hash:
                    from app.cache import history as _history

                    try:
                        await asyncio.to_thread(
                            _history.record,
                            caller_hash,
                            thread_id,
                            final_state.get("topic", ""),
                            final_report[:2000],
                        )
                    except Exception:  # noqa: BLE001
                        log.exception("episodic_memory_record_failed")

            yield {"event": "done", "data": json.dumps({"final_report_ready": True})}
        except TimeoutError:
            _update_metrics(thread_id, cb.usage, status_="error")
            yield {"event": "error", "data": json.dumps({"message": "Graph execution timed out"})}
        except Exception as e:  # noqa: BLE001
            log.exception("graph_stream_failed")
            _update_metrics(thread_id, cb.usage, status_="error")
            yield {"event": "error", "data": json.dumps({"message": str(e)})}

    return EventSourceResponse(_driver())


_PROGRESS_NODES = {
    "planner",
    "researcher",
    "grader",
    "web_search",
    "writer",
    "reviewer",
    "formatter",
}
# Tags we set on writer/formatter chains so we can filter streamed tokens
# to just the ones users want to see.
_TOKEN_STREAM_TAGS = {"writer", "formatter"}


async def _replay_history(graph, config, request) -> AsyncGenerator[dict]:
    """Replay finalized checkpoint history for late subscribers."""
    state_history = [s async for s in graph.aget_state_history(config)]
    for snapshot in reversed(state_history):
        if await request.is_disconnected():
            return
        event_data = {
            "step": snapshot.metadata.get("step", 0),
            "node": snapshot.metadata.get("source", "unknown"),
            "plan": snapshot.values.get("plan", []),
            "sections_written": list(snapshot.values.get("sections", {}).keys()),
            "revision_count": snapshot.values.get("revision_count", 0),
            "review_feedback": snapshot.values.get("review_feedback", ""),
            "done": bool(snapshot.values.get("final_report")),
        }
        yield {"event": "progress", "data": json.dumps(event_data)}
        await asyncio.sleep(0.05)
    yield {"event": "done", "data": json.dumps({"final_report_ready": True})}


@router.get("/{thread_id}/result", response_model=ResultResponse)
async def get_result(
    thread_id: str,
    api_key: str = Depends(verify_api_key),
) -> ResultResponse:
    session = _load_session(thread_id)
    _check_ownership(session, api_key)

    state = await _get_state(thread_id)
    report = state.get("final_report", "")
    if not report:
        raise HTTPException(
            status_code=202,
            detail="Report not ready yet. Check /stream for progress.",
        )
    return ResultResponse(thread_id=thread_id, final_report=report)


@router.post("/{thread_id}/followup", response_model=FollowupResponse)
async def followup(
    thread_id: str,
    body: FollowupRequest,
    api_key: str = Depends(enforce_rate_limit),
) -> FollowupResponse:
    """Refine an already-completed research session with a follow-up question.

    Appends the user's question to the conversation history (via add_messages
    reducer), drives the `refine_graph`, and returns the revised report.
    The original thread_id is preserved so subsequent follow-ups see the
    full multi-turn history.
    """
    from langchain_core.messages import HumanMessage

    session = _load_session(thread_id)
    _check_ownership(session, api_key)

    state = await _get_state(thread_id)
    if not state:
        raise ResearchNotFound("Session state missing")
    if not state.get("final_report"):
        raise HTTPException(
            status_code=409,
            detail="Original research hasn't finished yet — follow-up requires a completed report.",
        )

    refine = await get_refine_graph()
    cb = UsageCallback(thread_id=thread_id)
    config = _config(thread_id, callbacks=[cb])

    # Append the human question so the add_messages reducer gets it into state.
    await refine.aupdate_state(
        {"configurable": {"thread_id": thread_id}},
        {"messages": [HumanMessage(content=body.question)]},
    )

    try:
        async with asyncio.timeout(settings.graph_timeout_seconds):
            await refine.ainvoke(None, config)
    except TimeoutError as e:
        _update_metrics(thread_id, cb.usage, status_="error")
        raise GraphTimeout("Follow-up refinement timed out.") from e

    _update_metrics(thread_id, cb.usage, status_="done")

    new_state = await _get_state(thread_id)
    return FollowupResponse(
        thread_id=thread_id,
        final_report=new_state.get("final_report", ""),
    )


@router.get("/{thread_id}/metrics", response_model=MetricsResponse)
async def get_metrics(
    thread_id: str,
    api_key: str = Depends(verify_api_key),
) -> MetricsResponse:
    session = _load_session(thread_id)
    _check_ownership(session, api_key)
    return MetricsResponse(
        thread_id=thread_id,
        prompt_tokens=session.prompt_tokens,
        completion_tokens=session.completion_tokens,
        total_tokens=session.total_tokens,
        cost_usd=session.cost_usd,
        status=session.status,
    )
