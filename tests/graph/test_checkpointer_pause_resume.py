"""Pause → (simulated restart) → resume test for the research graph.

Exercises the concrete guarantee every interviewer asks about: the graph
persists state at every superstep, so interrupting mid-run, tearing down
the app, and resuming later with the same thread_id picks up exactly where
we left off — including across the `interrupt_before=await_approval` point.

Uses `InMemorySaver` instead of Postgres so the test is fast and hermetic.
It's the same checkpointer interface `AsyncPostgresSaver` implements; if
this test passes, Postgres-backed runs have the same guarantee.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import nodes as nodes_mod
from app.graph.graph import _builder
from app.graph.nodes import ResearchPlan, ReviewVerdict


def _initial_state(topic: str = "Checkpointer pause/resume demo") -> dict:
    return {
        "topic": topic,
        "thread_id": "tid-pause-resume",
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


def _stub_all_llms(monkeypatch):
    """Replace every node's external dependency with a deterministic stub."""

    # planner
    async def fake_planner(topic):  # noqa: ARG001
        return ResearchPlan(sections=["Alpha", "Beta"])

    monkeypatch.setattr(nodes_mod, "_planner_invoke", fake_planner)

    # cache retriever → always miss so we take the web path
    class MissRetriever:
        async def abatch(self, queries, config=None):  # noqa: ARG002
            return [[] for _ in queries]

    monkeypatch.setattr(nodes_mod, "_cache_retriever", MissRetriever())

    # Tavily
    async def fake_tavily(query):  # noqa: ARG001
        return [{"content": "search snippet for " + query}]

    monkeypatch.setattr(nodes_mod, "_tavily_search", fake_tavily)

    # rerank is a no-op on single-hit inputs; stub anyway to skip LLM.
    async def fake_rerank(query, docs, top_n=None):  # noqa: ARG001
        return docs

    monkeypatch.setattr(nodes_mod, "rerank", fake_rerank)

    async def fake_rerank_tavily(query, raw):  # noqa: ARG001
        return nodes_mod._join_tavily(raw)

    monkeypatch.setattr(nodes_mod, "_rerank_tavily", fake_rerank_tavily)

    # researcher_chain summariser
    class FakeSummariser:
        async def abatch(self, inputs, config=None, return_exceptions=False):  # noqa: ARG002
            return [f"summary of {i['section']}" for i in inputs]

    monkeypatch.setattr(nodes_mod, "_researcher_chain", FakeSummariser())

    # save_chunks: never persist
    async def fake_save_chunks(*_a, **_kw):
        return None

    monkeypatch.setattr(nodes_mod, "_save_chunks", fake_save_chunks)

    # writer
    async def fake_writer(topic, section, notes, feedback_instruction):  # noqa: ARG001
        suffix = " (v2)" if "Previous reviewer feedback" in feedback_instruction else ""
        return f"Prose for {section}{suffix}"

    monkeypatch.setattr(nodes_mod, "_writer_invoke", fake_writer)

    # reviewer — approve on first pass
    async def fake_reviewer(topic, report):  # noqa: ARG001
        return ReviewVerdict(approved=True, feedback="")

    monkeypatch.setattr(nodes_mod, "_reviewer_invoke", fake_reviewer)

    # formatter
    async def fake_formatter(topic, sections_text):  # noqa: ARG001
        return f"# Final: {topic}\n\n{sections_text}"

    monkeypatch.setattr(nodes_mod, "_formatter_invoke", fake_formatter)

    # citations (extract + attribute) — stub out the LLM and cosine work.
    async def fake_extract_claims(report):  # noqa: ARG001
        return []

    async def fake_attribute(claims, sources, threshold=None):  # noqa: ARG001
        return []

    monkeypatch.setattr("app.tools.attribution.extract_claims", fake_extract_claims)
    monkeypatch.setattr("app.tools.attribution.attribute", fake_attribute)
    monkeypatch.setattr("app.graph.nodes.extract_claims", fake_extract_claims)
    monkeypatch.setattr("app.graph.nodes.attribute", fake_attribute)

    # KG + safety
    async def fake_kg(report):  # noqa: ARG001
        from app.graph.nodes import KnowledgeGraph

        return KnowledgeGraph(entities=[], relations=[])

    async def fake_audit(report):  # noqa: ARG001
        return []

    monkeypatch.setattr(nodes_mod, "_kg_invoke", fake_kg)
    monkeypatch.setattr("app.tools.safety.audit_report", fake_audit)


@pytest.mark.asyncio
async def test_graph_pauses_at_interrupt_and_resumes_with_same_thread_id(monkeypatch):
    """Phase 1: run until the interrupt. Phase 2: simulate a long gap and a
    fresh app lifetime (new graph object, *same* checkpointer). Phase 3:
    resume — the planner's plan must be visible in the new state and the
    pipeline must complete to final_report."""
    _stub_all_llms(monkeypatch)

    saver = InMemorySaver()
    graph_v1 = _builder.compile(
        checkpointer=saver,
        interrupt_before=["await_approval"],
    )

    config = {"configurable": {"thread_id": "tid-pause-resume"}}

    # --- Phase 1 — drive until the interrupt_before fires
    await graph_v1.ainvoke(_initial_state(), config)

    snap_after_pause = await graph_v1.aget_state(config)
    paused_plan = snap_after_pause.values.get("plan", [])
    assert paused_plan == ["Alpha", "Beta"], "planner output should be checkpointed"
    # No report yet — we're paused before researcher.
    assert not snap_after_pause.values.get("final_report")
    # Next step is `await_approval` (the interrupt point).
    assert snap_after_pause.next == ("await_approval",)

    # --- Phase 2 — simulate 60s downtime and a fresh process. We discard the
    # graph object entirely and build a new one against the same saver. If
    # state really persists to the checkpointer, the new graph can pick up.
    await asyncio.sleep(0.01)  # token "time passed" — kept short for CI
    del graph_v1
    graph_v2 = _builder.compile(
        checkpointer=saver,
        interrupt_before=["await_approval"],
    )

    # Fresh graph, same thread_id → sees the paused state.
    snap_after_restart = await graph_v2.aget_state(config)
    assert snap_after_restart.values.get("plan") == paused_plan
    assert snap_after_restart.next == ("await_approval",)

    # --- Phase 3 — approve the plan and resume.
    await graph_v2.aupdate_state(config, {"human_approved": True})
    await graph_v2.ainvoke(None, config)

    final = await graph_v2.aget_state(config)
    assert final.values["final_report"].startswith("# Final:")
    # Both planned sections made it into the final report.
    assert "Alpha" in final.values["final_report"]
    assert "Beta" in final.values["final_report"]
    # Graph is fully drained — no more pending nodes.
    assert final.next == ()


@pytest.mark.asyncio
async def test_plan_edit_between_pause_and_resume_takes_effect(monkeypatch):
    """Between pause and resume, a plan override via aupdate_state must be
    the plan the researcher actually uses."""
    _stub_all_llms(monkeypatch)

    saver = InMemorySaver()
    graph = _builder.compile(
        checkpointer=saver,
        interrupt_before=["await_approval"],
    )
    config = {"configurable": {"thread_id": "tid-edit"}}
    await graph.ainvoke({**_initial_state(), "thread_id": "tid-edit"}, config)

    # User edits the plan — drop "Beta", add a new section.
    await graph.aupdate_state(
        config,
        {"plan": ["Alpha", "Gamma"], "human_approved": True},
    )

    await graph.ainvoke(None, config)
    final = await graph.aget_state(config)
    report = final.values["final_report"]
    assert "Alpha" in report
    assert "Gamma" in report
    # The original second section should not appear — the override replaced it.
    assert "Beta" not in report


@pytest.mark.asyncio
async def test_state_history_is_accessible_after_resume(monkeypatch):
    """aget_state_history should include every superstep across a pause/resume
    boundary — essential for the replay path in /stream."""
    _stub_all_llms(monkeypatch)

    saver = InMemorySaver()
    graph = _builder.compile(
        checkpointer=saver,
        interrupt_before=["await_approval"],
    )
    config = {"configurable": {"thread_id": "tid-history"}}
    await graph.ainvoke({**_initial_state(), "thread_id": "tid-history"}, config)
    await graph.aupdate_state(config, {"human_approved": True})
    await graph.ainvoke(None, config)

    history = [s async for s in graph.aget_state_history(config)]
    assert len(history) >= 5, "expected multiple superstep snapshots"
    # The oldest snapshot should have no plan yet; the newest should have
    # the final report.
    assert history[-1].values.get("plan", []) in ([], None)
    assert history[0].values.get("final_report", "").startswith("# Final:")


@pytest.mark.asyncio
async def test_messages_reducer_accumulates_across_resumes(monkeypatch):
    """The `add_messages` reducer must preserve prior turns when state is
    updated across a checkpoint boundary — underpins the /followup flow."""
    _stub_all_llms(monkeypatch)

    saver = InMemorySaver()
    graph = _builder.compile(
        checkpointer=saver,
        interrupt_before=["await_approval"],
    )
    config = {"configurable": {"thread_id": "tid-msgs"}}
    await graph.ainvoke({**_initial_state(), "thread_id": "tid-msgs"}, config)

    # Append two messages across two separate updates (simulating followup turns).
    await graph.aupdate_state(config, {"messages": [HumanMessage(content="first")]})
    await graph.aupdate_state(config, {"messages": [HumanMessage(content="second")]})

    snap = await graph.aget_state(config)
    contents = [getattr(m, "content", "") for m in snap.values["messages"]]
    assert "first" in contents
    assert "second" in contents
