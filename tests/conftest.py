"""Shared fixtures for API + graph tests.

Design choices:
- The whole FastAPI app depends on a Postgres connection pool at startup
  (AsyncPostgresSaver). We don't spin up a real DB in unit tests — we
  stub `create_postgres_checkpointer` with an in-memory `MemorySaver` so
  tests run fast and don't need Docker.
- DB writes (`SessionLocal().commit()`) are replaced by a no-op context
  manager so route handlers can still call `_save_session`/`_update_metrics`.
- External APIs (OpenAI, Tavily) are mocked at the chain/tool level via
  monkeypatching node chains — simpler and faster than respx.
"""

import os
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

# Ensure we never accidentally talk to real services during tests.
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("TAVILY_API_KEY", "tvly-test")
os.environ.setdefault("RESEARCH_API_KEY", "")
os.environ.setdefault("SENTRY_DSN", "")
os.environ.setdefault("LOG_FORMAT", "human")
os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://research:research@localhost:5432/research_db"
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _reset_rate_limiter(monkeypatch):
    """Ensure rate-limit state doesn't bleed across tests.

    Forces a fresh in-process limiter for every test (no shared Redis).
    """
    import app.rate_limit as rl

    monkeypatch.setattr(rl, "_limiter", None)
    monkeypatch.setattr(rl, "_redis_client", None)
    yield


# ---------------------------------------------------------------------------
# Fake in-memory DB session
# ---------------------------------------------------------------------------


class _FakeRow:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _InMemoryDB:
    """Subset of SQLAlchemy Session just large enough for routes.py."""

    def __init__(self) -> None:
        self._sessions: dict[uuid.UUID, _FakeRow] = {}

    def __enter__(self) -> "_InMemoryDB":
        return self

    def __exit__(self, *exc) -> None:
        return None

    # --- ORM subset
    def get(self, model, pk):  # noqa: ARG002
        return self._sessions.get(pk)

    def add(self, obj) -> None:
        self._sessions[obj.id] = obj

    def commit(self) -> None:
        pass

    def execute(self, *a, **kw):  # noqa: ARG002
        class _R:
            rowcount = 0  # used by DELETE in cache_routes

            def fetchall(self):
                return []

            def fetchone(self):
                # embedding validation sees an unset atttypmod (-1) = no-op
                return None

        return _R()


@pytest.fixture
def fake_db(monkeypatch) -> _InMemoryDB:
    """Replace SessionLocal so no real DB is needed."""
    store = _InMemoryDB()
    monkeypatch.setattr("app.api.routes.SessionLocal", lambda: store)
    monkeypatch.setattr("app.api.cache_routes.SessionLocal", lambda: store)
    return store


# ---------------------------------------------------------------------------
# Fake compiled graph (planner returns a fixed plan; approve/stream no-op)
# ---------------------------------------------------------------------------


class _StateSnapshot:
    def __init__(self, values: dict) -> None:
        self.values = values
        self.metadata = {"step": 0, "source": "planner"}


class _FakeGraph:
    """Minimal stand-in for the compiled StateGraph used in routes.py."""

    def __init__(self) -> None:
        self.states: dict[str, dict] = {}

    async def ainvoke(self, initial_state, config):
        thread_id = config["configurable"]["thread_id"]
        if initial_state is None:
            # Resume — pretend pipeline completed
            state = self.states.get(thread_id, {})
            state["final_report"] = "# Final Report\n\nAll sections assembled."
            self.states[thread_id] = state
            return state

        state = dict(initial_state)
        state["plan"] = ["Section A", "Section B", "Section C"]
        self.states[thread_id] = state
        return state

    async def aget_state(self, config):
        thread_id = config["configurable"]["thread_id"]
        values = self.states.get(thread_id)
        return _StateSnapshot(values) if values else None

    async def aupdate_state(self, config, update):
        thread_id = config["configurable"]["thread_id"]
        state = self.states.setdefault(thread_id, {})
        state.update(update)
        return config

    async def astream(self, inp, config, stream_mode):  # noqa: ARG002
        """Pretend the pipeline drives through nodes one by one."""
        thread_id = config["configurable"]["thread_id"]
        state = self.states.setdefault(thread_id, {})
        for node in (
            "researcher",
            "grader",
            "writer",
            "reviewer",
            "formatter",
            "citations",
            "knowledge_graph",
        ):
            yield {node: {"node": node}}
        state["final_report"] = "# Final Report\n\nAll sections assembled."
        state["citations"] = []

    async def astream_events(self, inp, config, version="v2"):  # noqa: ARG002
        """Emit pseudo-events shaped like LangGraph's on_chain_start/end."""
        thread_id = config["configurable"]["thread_id"]
        state = self.states.setdefault(thread_id, {})
        for node in (
            "researcher",
            "grader",
            "writer",
            "reviewer",
            "formatter",
            "citations",
            "knowledge_graph",
        ):
            yield {"event": "on_chain_start", "name": node, "tags": []}
            yield {
                "event": "on_chain_end",
                "name": node,
                "tags": [],
                "data": {"output": {"node": node}},
            }
        state["final_report"] = "# Final Report\n\nAll sections assembled."
        state["citations"] = []

    async def aget_state_history(self, config):  # noqa: ARG002
        # minimal replay — one snapshot
        if False:
            yield None


class _FakeRefineGraph:
    """Refine graph stub — appends messages + rewrites final_report."""

    def __init__(self, shared_states: dict[str, dict]) -> None:
        self.states = shared_states

    async def ainvoke(self, inp, config):  # noqa: ARG002
        thread_id = config["configurable"]["thread_id"]
        state = self.states.setdefault(thread_id, {})
        # Build a new report reflecting the latest follow-up message
        msgs = state.get("messages", []) or []
        last_q = getattr(msgs[-1], "content", "") if msgs else ""
        state["final_report"] = f"# Refined\n\nAnswering: {last_q}"
        return state

    async def aupdate_state(self, config, update):
        thread_id = config["configurable"]["thread_id"]
        state = self.states.setdefault(thread_id, {})
        # Emulate the add_messages reducer for the messages key
        if "messages" in update:
            existing = state.get("messages") or []
            state["messages"] = [*existing, *update["messages"]]
            state.update({k: v for k, v in update.items() if k != "messages"})
        else:
            state.update(update)
        return config


@pytest_asyncio.fixture
async def fake_graph(monkeypatch) -> _FakeGraph:
    g = _FakeGraph()
    refine = _FakeRefineGraph(g.states)

    async def _get_graph():
        return g

    async def _get_refine():
        return refine

    async def _shutdown():
        return None

    monkeypatch.setattr("app.api.routes.get_graph", _get_graph)
    monkeypatch.setattr("app.api.routes.get_refine_graph", _get_refine)
    monkeypatch.setattr("app.api.routes.shutdown_graph", _shutdown)
    return g


# ---------------------------------------------------------------------------
# httpx AsyncClient + ASGI lifespan
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def async_client(fake_graph, fake_db, monkeypatch) -> AsyncIterator:  # noqa: ARG001
    from asgi_lifespan import LifespanManager
    from httpx import ASGITransport, AsyncClient

    # Stub the startup embedding-dim check (no real DB in these tests).
    monkeypatch.setattr("app.tools.retriever.validate_embedding_dimensions", lambda: None)

    # G-14 speculative execution off by default in tests — existing tests
    # assert on pre-approval state which speculation would clobber.
    # Dedicated speculative-specific tests re-enable it locally.
    from app.config import settings as _s

    monkeypatch.setattr(_s, "speculative_execution_enabled", False)

    # Stub semantic cache — default to "miss" so tests drive the graph.
    # Individual tests can re-patch with a fake CachedReport to test the hit path.
    monkeypatch.setattr(
        "app.api.routes.semantic_cache.lookup",
        lambda topic, api_key_hash=None: None,
    )
    monkeypatch.setattr("app.api.routes.semantic_cache.store", lambda *a, **kw: None)

    from app.main import app

    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
