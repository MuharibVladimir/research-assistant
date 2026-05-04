"""Tests for speculative execution registry (G-14), per-user history (G-12),
and the adaptive_retrieval graph node (G-8)."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from app.graph import nodes as nodes_mod
from app.graph.speculative import REGISTRY

# ---------------------------------------------------------------------------
# G-14 Speculative-task REGISTRY
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_registry():
    REGISTRY.clear()
    yield
    REGISTRY.clear()


@pytest.mark.asyncio
async def test_register_and_cancel_task():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def never() -> None:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(never())
    await REGISTRY.register("tid-1", task)
    # Let the task start before we cancel — otherwise CancelledError fires
    # before the try/except and we can't observe the cancellation.
    await started.wait()
    assert "tid-1" in REGISTRY._tasks
    await REGISTRY.cancel("tid-1")
    assert cancelled.is_set()
    assert "tid-1" not in REGISTRY._tasks


@pytest.mark.asyncio
async def test_double_register_cancels_previous():
    first_started = asyncio.Event()
    first_cancelled = asyncio.Event()

    async def first() -> None:
        first_started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            first_cancelled.set()
            raise

    async def second() -> None:
        await asyncio.sleep(0.01)

    t1 = asyncio.create_task(first())
    await REGISTRY.register("tid-1", t1)
    await first_started.wait()
    t2 = asyncio.create_task(second())
    await REGISTRY.register("tid-1", t2)

    # Registering a new task for the same thread_id cancels the previous.
    await asyncio.sleep(0.01)  # yield so the cancellation lands
    assert first_cancelled.is_set()
    await t2


@pytest.mark.asyncio
async def test_cancel_missing_thread_is_noop():
    # Doesn't raise, doesn't fail.
    await REGISTRY.cancel("never-registered")


# ---------------------------------------------------------------------------
# G-12 History — pure record+fetch using a monkeypatched SessionLocal
# ---------------------------------------------------------------------------


class _FakeRow:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeHistoryDB:
    """Minimal in-memory stand-in for history.SessionLocal."""

    def __init__(self) -> None:
        self._rows: list = []

    def __enter__(self) -> _FakeHistoryDB:
        return self

    def __exit__(self, *exc) -> None:
        return None

    def add(self, row) -> None:
        self._rows.append(row)

    def commit(self) -> None:
        pass

    def execute(self, stmt, params):  # noqa: ARG002
        # Return rows in _rows order by dot-product proximity, with a
        # synthetic `similarity` column that always says 0.99 so the test
        # asserts on the ordering/filter, not the cosine arithmetic.
        key_hash = params.get("key_hash")
        k = params.get("k", 3)
        matches = [r for r in self._rows if r.api_key_hash == key_hash]
        results = []
        for r in matches[:k]:
            results.append(
                _FakeRow(
                    id=r.id,
                    topic=r.topic,
                    summary=r.summary,
                    similarity=0.99,
                )
            )

        class _R:
            def fetchall(self_inner):  # noqa: ARG002
                return results

        return _R()


@pytest.mark.asyncio
async def test_history_record_then_fetch(monkeypatch):
    from app.cache import history as history_mod

    db = _FakeHistoryDB()
    monkeypatch.setattr(history_mod, "SessionLocal", lambda: db)
    monkeypatch.setattr(history_mod, "embed_text", lambda _t: [1.0, 0.0])

    session_a = uuid.uuid4()
    session_b = uuid.uuid4()
    history_mod.record("hashA", session_a, "LangGraph intro", "summary A")
    history_mod.record("hashA", session_b, "LangGraph advanced", "summary B")
    # Another user's entry should not leak into hashA's view.
    history_mod.record("hashB", uuid.uuid4(), "CrewAI overview", "summary for B")

    hits = history_mod.fetch_relevant("hashA", "LangGraph revisited", top_k=5)
    summaries = {h["summary"] for h in hits}
    assert "summary A" in summaries
    assert "summary B" in summaries
    assert "summary for B" not in summaries


def test_history_no_op_on_empty_key(monkeypatch):
    from app.cache import history as history_mod

    sentinel = {"called": False}

    def fake_session():
        sentinel["called"] = True
        return _FakeHistoryDB()

    monkeypatch.setattr(history_mod, "SessionLocal", fake_session)
    history_mod.record("", uuid.uuid4(), "topic", "summary")
    assert sentinel["called"] is False
    assert history_mod.fetch_relevant("", "topic") == []


# ---------------------------------------------------------------------------
# G-8 adaptive_retrieval_node — pure state transformation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adaptive_retrieval_exits_when_depth_exhausted(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "max_retrieval_depth", 2)

    async def boom(*_a, **_kw):
        raise AssertionError("should not hit retrieval when depth is exhausted")

    monkeypatch.setattr(nodes_mod, "retrieve_relevant", boom)

    out = await nodes_mod.adaptive_retrieval_node(
        {
            "topic": "T",
            "retrieval_grades": {"s1": "irrelevant"},
            "retrieval_depth_count": 2,
            "thread_id": "tid",
        }
    )
    assert out == {}


@pytest.mark.asyncio
async def test_adaptive_retrieval_exits_when_no_irrelevant(monkeypatch):
    async def boom(*_a, **_kw):
        raise AssertionError("no work needed if nothing is irrelevant")

    monkeypatch.setattr(nodes_mod, "retrieve_relevant", boom)
    out = await nodes_mod.adaptive_retrieval_node(
        {
            "topic": "T",
            "retrieval_grades": {"s1": "relevant", "s2": "relevant"},
            "retrieval_depth_count": 0,
            "thread_id": "tid",
        }
    )
    assert out == {}


@pytest.mark.asyncio
async def test_adaptive_retrieval_flips_irrelevant_when_cache_rich(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "max_retrieval_depth", 2)
    monkeypatch.setattr(settings, "retriever_top_k", 3)
    monkeypatch.setattr(settings, "similarity_threshold", 0.75)

    def fake_retrieve(query, threshold, max_age_days):  # noqa: ARG001
        # Deeper retrieval returns content → section flips to relevant.
        return [{"content": "deeper find"}]

    monkeypatch.setattr(nodes_mod, "retrieve_relevant", fake_retrieve)

    out = await nodes_mod.adaptive_retrieval_node(
        {
            "topic": "T",
            "retrieval_grades": {"s1": "irrelevant", "s2": "relevant"},
            "retrieval_depth_count": 0,
            "thread_id": "tid",
            "search_results": {"s2": "kept"},
        }
    )
    assert out["retrieval_grades"]["s1"] == "relevant"
    assert out["retrieval_grades"]["s2"] == "relevant"
    assert out["search_results"]["s1"] == "deeper find"
    assert out["retrieval_depth_count"] == 1


@pytest.mark.asyncio
async def test_adaptive_retrieval_keeps_irrelevant_on_empty_result(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "max_retrieval_depth", 2)

    def fake_retrieve(*_a, **_kw):
        return []  # nothing found even at higher top_k

    monkeypatch.setattr(nodes_mod, "retrieve_relevant", fake_retrieve)

    out = await nodes_mod.adaptive_retrieval_node(
        {
            "topic": "T",
            "retrieval_grades": {"s1": "irrelevant"},
            "retrieval_depth_count": 0,
            "thread_id": "tid",
            "search_results": {},
        }
    )
    # Depth counter still increments (we did a pass), but grade stays irrelevant.
    assert out["retrieval_depth_count"] == 1
    assert out["retrieval_grades"]["s1"] == "irrelevant"
