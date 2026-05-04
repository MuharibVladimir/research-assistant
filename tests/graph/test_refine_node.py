"""Unit tests for refine_node — the multi-turn refinement step.

Validates that refine_node:
  * no-ops when there are no messages (nothing to refine)
  * feeds the conversation history and latest human request to the LLM
  * appends the new AI message to `messages` so the reducer grows history
  * preserves prior report as context when refining
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.graph.nodes as nodes_mod


@pytest.mark.asyncio
async def test_refine_noop_when_no_messages(monkeypatch):
    """With no messages queued, refine_node should return empty dict (nothing to do)."""

    async def boom(*a, **kw):  # noqa: ARG001
        raise AssertionError("LLM should not be called when state.messages is empty")

    monkeypatch.setattr(nodes_mod, "_refine_invoke", boom)
    state = {
        "topic": "t",
        "final_report": "# Original",
        "messages": [],
        "thread_id": "tid",
    }
    result = await nodes_mod.refine_node(state)
    assert result == {}


@pytest.mark.asyncio
async def test_refine_feeds_latest_human_message(monkeypatch):
    """The last human message is passed to the LLM as `request`."""
    captured: dict = {}

    async def fake_refine(report, conversation, request):
        captured.update(report=report, conversation=conversation, request=request)
        return "# Refined output"

    monkeypatch.setattr(nodes_mod, "_refine_invoke", fake_refine)

    state = {
        "topic": "LangGraph",
        "final_report": "# Prior report body",
        "messages": [HumanMessage(content="Expand the pricing section.")],
        "thread_id": "tid",
    }
    result = await nodes_mod.refine_node(state)
    assert result["final_report"] == "# Refined output"
    # AIMessage was appended
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "# Refined output"
    # LLM saw the right pieces
    assert captured["request"] == "Expand the pricing section."
    assert "Prior report body" in captured["report"]


@pytest.mark.asyncio
async def test_refine_preserves_prior_conversation(monkeypatch):
    """When multiple turns exist, conversation context is formatted from all but the last."""
    captured: dict = {}

    async def fake_refine(report, conversation, request):
        captured.update(report=report, conversation=conversation, request=request)
        return "# Revised"

    monkeypatch.setattr(nodes_mod, "_refine_invoke", fake_refine)

    state = {
        "topic": "Topic",
        "final_report": "# Prior",
        "messages": [
            HumanMessage(content="First question"),
            AIMessage(content="First answer"),
            HumanMessage(content="Second question"),
        ],
        "thread_id": "tid",
    }
    result = await nodes_mod.refine_node(state)
    assert result["final_report"] == "# Revised"

    # Latest turn is the `request`; prior turns end up in `conversation`.
    assert captured["request"] == "Second question"
    assert "First question" in captured["conversation"]
    assert "First answer" in captured["conversation"]
    # Ensure the latest question isn't duplicated in conversation (off-by-one)
    assert "Second question" not in captured["conversation"]


@pytest.mark.asyncio
async def test_refine_handles_missing_prior_report(monkeypatch):
    """If `final_report` is absent (shouldn't happen in prod but be defensive)."""
    captured: dict = {}

    async def fake_refine(report, conversation, request):  # noqa: ARG001
        captured["report"] = report
        return "# First"

    monkeypatch.setattr(nodes_mod, "_refine_invoke", fake_refine)

    state = {
        "topic": "T",
        "messages": [HumanMessage(content="Please write something.")],
        "thread_id": "tid",
    }
    result = await nodes_mod.refine_node(state)
    assert result["final_report"] == "# First"
    assert "(no prior report)" in captured["report"]


def test_format_conversation_shapes_roles():
    """_format_conversation renders `type: content` per message."""
    msgs = [HumanMessage(content="Q1"), AIMessage(content="A1")]
    out = nodes_mod._format_conversation(msgs).lower()
    assert "human" in out and "q1" in out
    assert "ai" in out and "a1" in out


def test_format_conversation_empty():
    assert nodes_mod._format_conversation([]) == "(no prior turns)"
