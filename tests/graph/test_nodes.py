"""Unit tests for each graph node.

LLM chains and the Tavily tool are stubbed by monkeypatching module-level
helpers — we verify the node's state-transformation logic, not the LLM itself.
"""

import pytest

import app.graph.nodes as nodes
from app.graph.nodes import (
    GradeVerdict,
    ResearchPlan,
    ReviewVerdict,
)


@pytest.mark.asyncio
async def test_planner_node_extracts_sections(monkeypatch):
    async def fake_invoke(topic):  # noqa: ARG001
        return ResearchPlan(sections=["  A  ", "B", "", "C"])

    monkeypatch.setattr(nodes, "_planner_invoke", fake_invoke)
    result = await nodes.planner_node({"topic": "X", "thread_id": "t"})
    assert result["plan"] == ["A", "B", "C"]
    assert result["human_approved"] is False


@pytest.mark.asyncio
async def test_researcher_uses_cache_when_relevant(monkeypatch):
    """Cache hit path: LangChain retriever returns Documents, no Tavily call."""
    from langchain_core.documents import Document as LCDocument

    class FakeRetriever:
        async def abatch(self, queries, config=None):  # noqa: ARG002
            return [
                [LCDocument(page_content="cached chunk", metadata={"similarity": 0.9})]
                for _ in queries
            ]

    async def fake_tavily(query):  # noqa: ARG001
        raise AssertionError("Tavily should not be called on a cache hit")

    monkeypatch.setattr(nodes, "_cache_retriever", FakeRetriever())
    monkeypatch.setattr(nodes, "_tavily_search", fake_tavily)

    state = {"plan": ["s1"], "topic": "t", "thread_id": "tid"}
    result = await nodes.researcher_node(state)
    assert result["search_results"]["s1"] == "cached chunk"
    assert result["retrieval_grades"]["s1"] == "relevant"


@pytest.mark.asyncio
async def test_researcher_rerank_skipped_on_single_hit(monkeypatch):
    """When cache returns only one doc, we should not call the LLM reranker."""
    from langchain_core.documents import Document as LCDocument

    class FakeRetriever:
        async def abatch(self, queries, config=None):  # noqa: ARG002
            return [
                [LCDocument(page_content="only one", metadata={"similarity": 0.9})] for _ in queries
            ]

    rerank_calls = {"n": 0}

    async def fake_rerank(*_a, **_kw):
        rerank_calls["n"] += 1
        raise AssertionError("rerank should not be called for a single-doc hit")

    monkeypatch.setattr(nodes, "_cache_retriever", FakeRetriever())
    monkeypatch.setattr(nodes, "rerank", fake_rerank)

    state = {"plan": ["s1"], "topic": "t", "thread_id": "tid"}
    result = await nodes.researcher_node(state)
    assert result["search_results"]["s1"] == "only one"
    assert rerank_calls["n"] == 0


@pytest.mark.asyncio
async def test_researcher_rerank_used_on_multi_hit(monkeypatch):
    """When cache returns multiple docs, rerank IS called."""
    from langchain_core.documents import Document as LCDocument

    class FakeRetriever:
        async def abatch(self, queries, config=None):  # noqa: ARG002
            return [
                [
                    LCDocument(page_content="doc1", metadata={"id": "1", "similarity": 0.9}),
                    LCDocument(page_content="doc2", metadata={"id": "2", "similarity": 0.85}),
                    LCDocument(page_content="doc3", metadata={"id": "3", "similarity": 0.8}),
                ]
                for _ in queries
            ]

    rerank_calls = {"n": 0}

    async def fake_rerank(query, docs, top_n=None):  # noqa: ARG001
        rerank_calls["n"] += 1
        # Reverse to prove reranker output is used (vs. input order).
        return list(reversed(docs))

    monkeypatch.setattr(nodes, "_cache_retriever", FakeRetriever())
    monkeypatch.setattr(nodes, "rerank", fake_rerank)

    state = {"plan": ["s1"], "topic": "t", "thread_id": "tid"}
    result = await nodes.researcher_node(state)
    # Reranker flipped the order → last doc first.
    assert result["search_results"]["s1"].startswith("doc3")
    assert rerank_calls["n"] == 1


@pytest.mark.asyncio
async def test_researcher_falls_back_to_tavily_on_cache_miss(monkeypatch):
    """Cache miss path: Tavily + summariser .abatch + chunked save."""

    class FakeRetriever:
        async def abatch(self, queries, config=None):  # noqa: ARG002
            return [[] for _ in queries]  # all misses

    class FakeSummariser:
        async def abatch(self, inputs, config=None, return_exceptions=False):  # noqa: ARG002
            return [f"summary of {i['section']}" for i in inputs]

    async def fake_tavily(query):  # noqa: ARG001
        return [{"content": "fresh from web"}]

    async def fake_save(topic, section, content):  # noqa: ARG001
        pass

    monkeypatch.setattr(nodes, "_cache_retriever", FakeRetriever())
    monkeypatch.setattr(nodes, "_researcher_chain", FakeSummariser())
    monkeypatch.setattr(nodes, "_tavily_search", fake_tavily)
    monkeypatch.setattr(nodes, "_save_chunks", fake_save)

    state = {"plan": ["s1"], "topic": "t", "thread_id": "tid"}
    result = await nodes.researcher_node(state)
    assert result["search_results"]["s1"] == "summary of s1"
    assert result["retrieval_grades"]["s1"] == "irrelevant"


@pytest.mark.asyncio
async def test_grader_skips_llm_for_irrelevant(monkeypatch):
    """Sections already tagged 'irrelevant' (fresh web data) skip the grader."""
    calls = []

    async def fake_grade(section, content):  # noqa: ARG001
        calls.append(section)
        return GradeVerdict(grade="relevant")

    monkeypatch.setattr(nodes, "_grade_invoke", fake_grade)

    state = {
        "search_results": {"s1": "X", "s2": "Y"},
        "retrieval_grades": {"s1": "relevant", "s2": "irrelevant"},
        "thread_id": "tid",
    }
    result = await nodes.grader_node(state)
    assert result["retrieval_grades"]["s2"] == "irrelevant"
    assert calls == ["s1"]  # only cache hit was re-graded


@pytest.mark.asyncio
async def test_reviewer_approves(monkeypatch):
    async def fake_reviewer(topic, report):  # noqa: ARG001
        return ReviewVerdict(approved=True, feedback="")

    monkeypatch.setattr(nodes, "_reviewer_invoke", fake_reviewer)
    state = {"topic": "t", "sections": {"a": "b"}, "revision_count": 0, "thread_id": "tid"}
    result = await nodes.reviewer_node(state)
    assert result["review_feedback"] == ""


@pytest.mark.asyncio
async def test_reviewer_requests_revision(monkeypatch):
    async def fake_reviewer(topic, report):  # noqa: ARG001
        return ReviewVerdict(approved=False, feedback="add more depth")

    monkeypatch.setattr(nodes, "_reviewer_invoke", fake_reviewer)
    state = {"topic": "t", "sections": {"a": "b"}, "revision_count": 0, "thread_id": "tid"}
    result = await nodes.reviewer_node(state)
    assert result["review_feedback"] == "add more depth"
    assert result["revision_count"] == 1


@pytest.mark.asyncio
async def test_reviewer_caps_at_max_revisions(monkeypatch):
    """Even with 'needs revision' feedback, if we've hit the cap, exit the loop."""
    from app.config import settings

    async def fake_reviewer(topic, report):  # noqa: ARG001
        return ReviewVerdict(approved=False, feedback="still bad")

    monkeypatch.setattr(nodes, "_reviewer_invoke", fake_reviewer)
    state = {
        "topic": "t",
        "sections": {"a": "b"},
        "revision_count": settings.max_revision_count,
        "thread_id": "tid",
    }
    result = await nodes.reviewer_node(state)
    assert result["review_feedback"] == ""


@pytest.mark.asyncio
async def test_formatter_assembles_report(monkeypatch):
    async def fake_formatter(topic, sections_text):  # noqa: ARG001
        return f"# Report\n\n{sections_text}"

    monkeypatch.setattr(nodes, "_formatter_invoke", fake_formatter)
    state = {"topic": "t", "sections": {"Intro": "hello"}, "thread_id": "tid"}
    result = await nodes.formatter_node(state)
    assert "# Report" in result["final_report"]
    assert "hello" in result["final_report"]
