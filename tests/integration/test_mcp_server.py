"""MCP server tests — validate tool registration and payload shapes.

We don't boot a real MCP transport (that's a subprocess test); instead we
introspect the FastMCP instance directly and call the tool functions in-
process with stubbed backends. This catches schema drift (renamed fields,
missing return types) without any network or LLM cost.
"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_mcp_tools_registered():
    """Every tool we expose should be visible in the MCP tool list."""
    from app.mcp_server.server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}

    expected = {
        "search_cache",
        "search_hybrid",
        "search_bm25",
        "save_to_cache",
        "search_web",
        "rerank",
        "extract_claims",
        "attribute_claims",
    }
    missing = expected - names
    assert not missing, f"MCP tools missing from registration: {missing}"


@pytest.mark.asyncio
async def test_mcp_tools_have_descriptions():
    """Every MCP tool must carry a docstring description so clients know what it does."""
    from app.mcp_server.server import mcp

    tools = await mcp.list_tools()
    empty = [t.name for t in tools if not (t.description or "").strip()]
    assert not empty, f"tools missing description: {empty}"


@pytest.mark.asyncio
async def test_search_cache_filters_by_threshold(monkeypatch):
    """search_cache should drop hits below threshold."""
    from app.mcp_server import server as mcp_mod

    def fake_similar(query, top_k=None, max_age_days=None):  # noqa: ARG001
        return [
            {"id": "1", "topic": "t", "section": "s", "content": "high", "similarity": 0.9},
            {"id": "2", "topic": "t", "section": "s", "content": "low", "similarity": 0.5},
        ]

    monkeypatch.setattr(mcp_mod, "retrieve_similar", fake_similar)

    # Call the underlying function — FastMCP wraps but keeps .fn accessible via
    # the tool registration. Simpler: call module-level function directly.
    from app.mcp_server.server import search_cache

    # FastMCP @mcp.tool() registers the function and returns it unchanged,
    # so the module attribute is the underlying callable.
    result = search_cache(query="hi", top_k=5, threshold=0.75)
    assert len(result) == 1
    assert result[0]["content"] == "high"


@pytest.mark.asyncio
async def test_search_web_normalises_tavily_output(monkeypatch):
    """Tavily returns list of dicts; search_web must flatten to {content, url}."""
    from app.mcp_server import server as mcp_mod

    class FakeTool:
        def invoke(self, q):  # noqa: ARG002
            return [
                {"content": "A", "url": "http://a"},
                {"content": "B", "url": "http://b"},
                {"bogus": "keep going"},  # non-dict-ish payload
            ]

    monkeypatch.setattr(mcp_mod, "get_search_tool", lambda max_results=3: FakeTool())

    from app.mcp_server.server import search_web

    result = search_web(query="x", max_results=2)
    assert len(result) == 3
    assert result[0]["url"] == "http://a"
    assert result[1]["content"] == "B"
