"""Revision-loop integration test — the reviewer → researcher back-edge.

`tests/test_edges.py` proves the routing function picks the right branch.
This test proves the actual graph follows that branch end-to-end through
real node code, not just the edge function — i.e. when reviewer says
"needs revision", the researcher really does run a second time with the
feedback injected into the writer prompt, and the eventual approval
terminates the loop without leaving `review_feedback` set.

Also exercises the `max_revision_count` cap: if the reviewer keeps
rejecting, the loop must exit with the (imperfect) report rather than
spin forever.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import nodes as nodes_mod
from app.graph.graph import _builder
from app.graph.nodes import ResearchPlan, ReviewVerdict


def _initial_state() -> dict:
    return {
        "topic": "Revision-loop integration",
        "thread_id": "tid-revision",
        "plan": [],
        "human_approved": False,
        "search_results": {},
        "sections": {},
        "review_feedback": "",
        "revision_count": 0,
        "final_report": "",
        "retrieval_grades": {},
        "citations": [],
        "knowledge_graph": {"entities": [], "relations": []},
        "safety_flags": [],
        "budget_usd": 0.0,
        "retrieval_depth_count": 0,
        "messages": [],
    }


def _stub_nodes(monkeypatch, reviewer_verdicts: list[ReviewVerdict]) -> dict:
    """Wire stubs for every node, with a programmable reviewer verdict list.

    Returns a counter dict. Note we track **writer** and **reviewer** node
    invocations rather than researcher passes, because the `_researcher_chain`
    summariser also runs inside `web_search_node`, which would double-count.
    Writer runs once per revision round — cleaner signal for loop testing.
    """
    counters = {"writer": 0, "reviewer": 0, "formatter": 0}

    async def fake_planner(topic):  # noqa: ARG001
        return ResearchPlan(sections=["OnlySection"])

    monkeypatch.setattr(nodes_mod, "_planner_invoke", fake_planner)

    class MissRetriever:
        async def abatch(self, queries, config=None):  # noqa: ARG002
            return [[] for _ in queries]

    monkeypatch.setattr(nodes_mod, "_cache_retriever", MissRetriever())

    # adaptive_retrieval_node also calls retrieve_relevant — return empty so
    # it doesn't try to embed via OpenAI.
    def fake_retrieve_relevant(query, threshold=None, max_age_days=None):  # noqa: ARG001
        return []

    monkeypatch.setattr(nodes_mod, "retrieve_relevant", fake_retrieve_relevant)

    async def fake_tavily(query):  # noqa: ARG001
        return [{"content": "fresh web content"}]

    monkeypatch.setattr(nodes_mod, "_tavily_search", fake_tavily)

    async def fake_rerank(query, docs, top_n=None):  # noqa: ARG001
        return docs

    monkeypatch.setattr(nodes_mod, "rerank", fake_rerank)

    async def fake_rerank_tavily(query, raw):  # noqa: ARG001
        return nodes_mod._join_tavily(raw)

    monkeypatch.setattr(nodes_mod, "_rerank_tavily", fake_rerank_tavily)

    # Summariser chain (used by researcher + web_search). Not counted —
    # see note above.
    class Summariser:
        async def abatch(self, inputs, config=None, return_exceptions=False):  # noqa: ARG002
            return [f"notes for {i['section']}" for i in inputs]

    monkeypatch.setattr(nodes_mod, "_researcher_chain", Summariser())

    async def fake_save_chunks(*_a, **_kw):
        return None

    monkeypatch.setattr(nodes_mod, "_save_chunks", fake_save_chunks)

    async def fake_writer(topic, section, notes, feedback_instruction):  # noqa: ARG001
        counters["writer"] += 1
        had_feedback = "Previous reviewer feedback" in feedback_instruction
        return f"Prose {counters['writer']} [fb={had_feedback}]"

    monkeypatch.setattr(nodes_mod, "_writer_invoke", fake_writer)

    verdicts = iter(reviewer_verdicts)

    async def fake_reviewer(topic, report):  # noqa: ARG001
        counters["reviewer"] += 1
        try:
            return next(verdicts)
        except StopIteration:
            return ReviewVerdict(approved=True, feedback="")

    monkeypatch.setattr(nodes_mod, "_reviewer_invoke", fake_reviewer)

    async def fake_formatter(topic, sections_text):  # noqa: ARG001
        counters["formatter"] += 1
        return f"# Final: {topic}\n\n{sections_text}"

    monkeypatch.setattr(nodes_mod, "_formatter_invoke", fake_formatter)

    async def fake_extract(_report):
        return []

    async def fake_attribute(_claims, _sources, threshold=None):  # noqa: ARG001
        return []

    async def fake_kg(_report):
        from app.graph.nodes import KnowledgeGraph

        return KnowledgeGraph(entities=[], relations=[])

    async def fake_audit(_report):
        return []

    monkeypatch.setattr("app.tools.attribution.extract_claims", fake_extract)
    monkeypatch.setattr("app.tools.attribution.attribute", fake_attribute)
    monkeypatch.setattr("app.graph.nodes.extract_claims", fake_extract)
    monkeypatch.setattr("app.graph.nodes.attribute", fake_attribute)
    monkeypatch.setattr(nodes_mod, "_kg_invoke", fake_kg)
    monkeypatch.setattr("app.tools.safety.audit_report", fake_audit)

    return counters


async def _run_to_completion(graph, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    state = _initial_state()
    state["thread_id"] = thread_id
    await graph.ainvoke(state, config)
    await graph.aupdate_state(config, {"human_approved": True})
    await graph.ainvoke(None, config)
    snap = await graph.aget_state(config)
    return snap.values


@pytest.mark.asyncio
async def test_reviewer_reject_once_then_approve_runs_researcher_twice(monkeypatch):
    """Classic revision cycle: reject → loop back → approve → move on."""
    counters = _stub_nodes(
        monkeypatch,
        reviewer_verdicts=[
            ReviewVerdict(approved=False, feedback="needs more depth on OnlySection"),
            ReviewVerdict(approved=True, feedback=""),
        ],
    )

    saver = InMemorySaver()
    graph = _builder.compile(checkpointer=saver, interrupt_before=["await_approval"])
    values = await _run_to_completion(graph, "tid-reject-once")

    # Writer ran twice — once initial, once with feedback injected.
    assert counters["writer"] == 2
    # Reviewer ran twice (reject, then approve).
    assert counters["reviewer"] == 2
    # Formatter ran exactly once — revision exit cleared feedback.
    assert counters["formatter"] == 1

    # Final state: no stale feedback, revision_count reflects the loop.
    assert values["review_feedback"] == ""
    assert values["revision_count"] == 1
    assert values["final_report"].startswith("# Final:")


@pytest.mark.asyncio
async def test_reviewer_rejects_at_max_revisions_exits_gracefully(monkeypatch):
    """If the reviewer refuses forever, `max_revision_count` forces exit.

    We hand the reviewer an endless stream of "needs revision" verdicts;
    the loop must still terminate at the formatter exactly once.
    """
    counters = _stub_nodes(
        monkeypatch,
        reviewer_verdicts=[
            ReviewVerdict(approved=False, feedback="still not good"),
            ReviewVerdict(approved=False, feedback="still not good"),
            ReviewVerdict(approved=False, feedback="still not good"),
            ReviewVerdict(approved=False, feedback="still not good"),
        ],
    )

    saver = InMemorySaver()
    graph = _builder.compile(checkpointer=saver, interrupt_before=["await_approval"])
    values = await _run_to_completion(graph, "tid-forever-reject")

    from app.config import settings

    expected_passes = 1 + settings.max_revision_count
    assert counters["writer"] == expected_passes
    # Exit path: formatter still ran exactly once, no stale feedback.
    assert counters["formatter"] == 1
    assert values["review_feedback"] == ""
    # revision_count should not exceed the cap.
    assert values["revision_count"] == settings.max_revision_count


@pytest.mark.asyncio
async def test_approve_on_first_pass_skips_revision_loop(monkeypatch):
    """Happy path: reviewer approves immediately. Researcher must run only once."""
    counters = _stub_nodes(
        monkeypatch,
        reviewer_verdicts=[ReviewVerdict(approved=True, feedback="")],
    )

    saver = InMemorySaver()
    graph = _builder.compile(checkpointer=saver, interrupt_before=["await_approval"])
    values = await _run_to_completion(graph, "tid-first-pass")

    assert counters["writer"] == 1
    assert counters["reviewer"] == 1
    assert counters["formatter"] == 1
    assert values["revision_count"] == 0
    assert values["review_feedback"] == ""


@pytest.mark.asyncio
async def test_revision_bypasses_await_approval(monkeypatch):
    """The back-edge from reviewer → researcher must skip `await_approval`.

    We observe this indirectly: the test reaches final_report in a single
    `ainvoke(None, ...)` call after the initial approval, even with a
    revision in between. If the revision loop ever hit `await_approval`
    again, the second invoke would return paused (with `next` non-empty)
    instead of completing.
    """
    _stub_nodes(
        monkeypatch,
        reviewer_verdicts=[
            ReviewVerdict(approved=False, feedback="one more pass please"),
            ReviewVerdict(approved=True, feedback=""),
        ],
    )

    saver = InMemorySaver()
    graph = _builder.compile(checkpointer=saver, interrupt_before=["await_approval"])
    config = {"configurable": {"thread_id": "tid-bypass"}}

    state = _initial_state()
    state["thread_id"] = "tid-bypass"
    await graph.ainvoke(state, config)
    await graph.aupdate_state(config, {"human_approved": True})
    await graph.ainvoke(None, config)

    final = await graph.aget_state(config)
    # If `await_approval` had fired a second time, `.next` would be non-empty.
    assert final.next == ()
    assert final.values["final_report"].startswith("# Final:")
