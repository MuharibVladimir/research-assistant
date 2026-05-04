"""Tests for HyDE (G-10), cost-aware planner (G-7), knowledge-graph node (G-13)."""

from __future__ import annotations

import pytest

from app.config import settings
from app.graph import nodes as nodes_mod
from app.graph.nodes import KGEntity, KGRelation, KnowledgeGraph, ResearchPlan
from app.tools import hyde as hyde_mod

# ---------------------------------------------------------------------------
# G-10 HyDE — blend arithmetic is deterministic
# ---------------------------------------------------------------------------


def test_blend_pure_query_when_alpha_zero(monkeypatch):
    monkeypatch.setattr(settings, "hyde_blend", 0.0)
    blended = hyde_mod.blend_embedding([1.0, 0.0], [0.0, 1.0])
    # alpha=0 → pure query vector (normalised).
    assert blended[0] == pytest.approx(1.0)
    assert blended[1] == pytest.approx(0.0, abs=1e-6)


def test_blend_pure_hypothetical_when_alpha_one(monkeypatch):
    monkeypatch.setattr(settings, "hyde_blend", 1.0)
    blended = hyde_mod.blend_embedding([1.0, 0.0], [0.0, 1.0])
    assert blended[0] == pytest.approx(0.0, abs=1e-6)
    assert blended[1] == pytest.approx(1.0)


def test_blend_is_renormalised_to_unit_length(monkeypatch):
    monkeypatch.setattr(settings, "hyde_blend", 0.5)
    blended = hyde_mod.blend_embedding([3.0, 4.0], [0.0, 0.0])
    norm = (blended[0] ** 2 + blended[1] ** 2) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)


@pytest.mark.asyncio
async def test_expanded_embedding_returns_none_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "hyde_enabled", False)
    result = await hyde_mod.expanded_embedding("t", "s", "query")
    assert result is None


# ---------------------------------------------------------------------------
# G-7 Cost-aware planner
# ---------------------------------------------------------------------------


def test_budget_aware_topic_no_budget_returns_original():
    out = nodes_mod._budget_aware_topic("LangGraph", 0.0)
    assert out == "LangGraph"


def test_budget_aware_topic_emits_section_cap():
    out = nodes_mod._budget_aware_topic("LangGraph", 0.15)
    # 0.15 / 0.015 = 10 → clamped to 6.
    assert "6 sections" in out or "plan at most" in out


def test_budget_aware_topic_respects_small_budget():
    out = nodes_mod._budget_aware_topic("LangGraph", 0.03)
    # 0.03 / 0.015 = 2
    assert "2 sections" in out


@pytest.mark.asyncio
async def test_planner_node_truncates_by_budget(monkeypatch):
    async def fake_invoke(topic):  # noqa: ARG001
        # LLM returns 6 sections even though budget only fits 2.
        return ResearchPlan(sections=[f"Section {i}" for i in range(6)])

    monkeypatch.setattr(nodes_mod, "_planner_invoke", fake_invoke)

    result = await nodes_mod.planner_node({"topic": "X", "thread_id": "t", "budget_usd": 0.03})
    # Budget hard-cap keeps at most `budget / est_per_section` = 2 sections.
    assert len(result["plan"]) == 2


@pytest.mark.asyncio
async def test_planner_node_no_budget_keeps_all_sections(monkeypatch):
    async def fake_invoke(topic):  # noqa: ARG001
        return ResearchPlan(sections=[f"S{i}" for i in range(5)])

    monkeypatch.setattr(nodes_mod, "_planner_invoke", fake_invoke)
    result = await nodes_mod.planner_node({"topic": "X", "thread_id": "t", "budget_usd": 0.0})
    assert len(result["plan"]) == 5


# ---------------------------------------------------------------------------
# G-13 kg_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kg_node_empty_report_returns_empty_graph(monkeypatch):
    # Even short reports skip the LLM. Verify the safety audit path, too.
    monkeypatch.setattr("app.tools.safety.audit_report", _make_async(return_value=[]))
    out = await nodes_mod.kg_node({"final_report": "", "thread_id": "t"})
    assert out["knowledge_graph"] == {"entities": [], "relations": []}
    assert out["safety_flags"] == []


@pytest.mark.asyncio
async def test_kg_node_extracts_entities_and_relations(monkeypatch):
    fake_kg = KnowledgeGraph(
        entities=[
            KGEntity(name="LangGraph", type="tool"),
            KGEntity(name="Anthropic", type="company"),
        ],
        relations=[
            KGRelation.model_validate(
                {"from": "LangGraph", "relation": "maintained_by", "to": "Anthropic"}
            )
        ],
    )

    async def fake_invoke(report):  # noqa: ARG001
        return fake_kg

    async def fake_audit(report):  # noqa: ARG001
        return [{"kind": "speculative_claim", "quote": "q", "explanation": "e"}]

    monkeypatch.setattr(nodes_mod, "_kg_invoke", fake_invoke)
    monkeypatch.setattr("app.tools.safety.audit_report", fake_audit)

    out = await nodes_mod.kg_node({"final_report": "Long enough " * 50, "thread_id": "t"})
    kg = out["knowledge_graph"]
    assert len(kg["entities"]) == 2
    assert kg["entities"][0]["name"] == "LangGraph"
    assert kg["relations"][0]["from"] == "LangGraph"
    assert kg["relations"][0]["to"] == "Anthropic"
    assert out["safety_flags"][0]["kind"] == "speculative_claim"


@pytest.mark.asyncio
async def test_kg_node_survives_llm_failure(monkeypatch):
    async def boom(report):  # noqa: ARG001
        raise RuntimeError("LLM down")

    async def fake_audit(report):  # noqa: ARG001
        return []

    monkeypatch.setattr(nodes_mod, "_kg_invoke", boom)
    monkeypatch.setattr("app.tools.safety.audit_report", fake_audit)
    out = await nodes_mod.kg_node({"final_report": "Long enough " * 50, "thread_id": "t"})
    assert out["knowledge_graph"] == {"entities": [], "relations": []}
    assert out["safety_flags"] == []


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------


def _make_async(return_value):
    async def _f(*_a, **_kw):
        return return_value

    return _f
