"""Model Context Protocol server — exposes this project's tools to other agents.

Why an MCP server?
    Every tool in `app/tools/` is useful beyond this one graph:
      * the pgvector retriever / hybrid search would help any RAG agent;
      * the Tavily wrapper is a generic web-search tool;
      * the claim extractor + attribution can audit any LLM-generated text.

    Wrapping them in an MCP server means Claude Desktop, Cursor, any LangGraph
    agent, or any other MCP-compatible runtime can call them without linking
    against our code. The tools stay in one place; agents compose freely.

Usage (stdio — for Claude Desktop and most LangGraph agent integrations):

    uv run python -m app.mcp_server

Usage (SSE — for networked clients):

    uv run python -m app.mcp_server --transport sse --port 8765

See `loadtests/README.md` and the top-level README "MCP Integration" section
for client configuration examples.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import sys

from mcp.server.fastmcp import FastMCP

from app.config import settings
from app.tools.attribution import attribute as _attribute
from app.tools.attribution import extract_claims as _extract_claims
from app.tools.reranker import rerank as _rerank
from app.tools.retriever import (
    VectorCacheRetriever,
    retrieve_bm25,
    retrieve_similar,
    save_document,
)
from app.tools.search import get_search_tool

log = logging.getLogger(__name__)


# Write-tool auth (C-4): the MCP `save_to_cache` tool is a cache-poisoning
# primitive if exposed without authentication — any client could inject
# prompt-injection payloads into the pgvector cache that later get served
# to all other users. We require a token for write tools that matches
# either `MCP_WRITE_TOKEN` env var (preferred — distinct from the API key)
# or, as a fallback, `settings.research_api_key`.
_MCP_WRITE_TOKEN = os.environ.get("MCP_WRITE_TOKEN") or ""


def _check_write_token(token: str | None) -> None:
    expected = _MCP_WRITE_TOKEN or settings.research_api_key
    if not expected:
        raise PermissionError(
            "save_to_cache is disabled: set MCP_WRITE_TOKEN (or RESEARCH_API_KEY) "
            "on the MCP server process to enable writes."
        )
    if not token or not hmac.compare_digest(token, expected):
        raise PermissionError("save_to_cache: invalid or missing token")


mcp = FastMCP(
    "research-assistant-tools",
    instructions=(
        "Tools for researching a topic: pgvector-backed semantic search, "
        "hybrid BM25+vector retrieval, Tavily web search, LLM-based "
        "claim extraction + source attribution, and an LLM reranker. "
        "Use search_cache first; fall back to search_web on miss."
    ),
)


# ---------------------------------------------------------------------------
# Retrieval tools
# ---------------------------------------------------------------------------


@mcp.tool()
def search_cache(
    query: str,
    top_k: int = 5,
    threshold: float = 0.75,
    max_age_days: int | None = None,
) -> list[dict]:
    """Semantic search over the pgvector document cache (cosine similarity).

    Returns only documents whose similarity is >= threshold. Empty list
    means cache miss — caller should fall back to search_web.

    Each returned item has: topic, section, content, similarity.
    """
    docs = retrieve_similar(query, top_k=top_k, max_age_days=max_age_days)
    return [
        {
            "topic": d["topic"],
            "section": d["section"],
            "content": d["content"],
            "similarity": d["similarity"],
        }
        for d in docs
        if d["similarity"] >= threshold
    ]


@mcp.tool()
def search_hybrid(
    query: str,
    top_k: int = 5,
    max_age_days: int | None = None,
) -> list[dict]:
    """Hybrid BM25 + pgvector retrieval fused via Reciprocal Rank Fusion.

    Use this over search_cache when the query mixes semantic and keyword
    intent (e.g. "Q3 2024 earnings for Nvidia") — RRF gives keyword
    matches a fair shot against embeddings.
    """
    retriever = VectorCacheRetriever(
        threshold=0.0,  # we want RRF order, not cache-hit gating
        max_age_days=max_age_days,
        hybrid=True,
    )
    docs = retriever.invoke(query)
    return [
        {
            "content": d.page_content,
            **d.metadata,
        }
        for d in docs[:top_k]
    ]


@mcp.tool()
def search_bm25(query: str, top_k: int = 5, max_age_days: int | None = None) -> list[dict]:
    """Lexical-only search via Postgres full-text (ts_rank_cd)."""
    return retrieve_bm25(query, top_k=top_k, max_age_days=max_age_days)


@mcp.tool()
def save_to_cache(topic: str, section: str, content: str, token: str = "") -> dict:
    """Embed + persist a document chunk into the pgvector cache.

    Requires `token` matching MCP_WRITE_TOKEN (or RESEARCH_API_KEY). Without
    this gate, any MCP client could poison the cache with prompt-injection
    payloads that later get served to all users (C-4).
    """
    _check_write_token(token)
    # Content floor (L-3) — don't let junk poison the vector index.
    if not content or len(content.strip()) < 50:
        raise ValueError("content too short (min 50 chars after strip)")
    save_document(topic, section, content)
    return {"ok": True, "topic": topic, "section": section, "chars": len(content)}


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------


@mcp.tool()
def search_web(query: str, max_results: int = 3) -> list[dict]:
    """Live web search via Tavily. Returns raw snippets — no LLM summarisation."""
    tool = get_search_tool(max_results=max_results)
    raw = tool.invoke(query)
    if isinstance(raw, list):
        return [
            {"content": r.get("content", ""), "url": r.get("url", "")}
            for r in raw
            if isinstance(r, dict)
        ]
    return [{"content": str(raw), "url": ""}]


# ---------------------------------------------------------------------------
# Quality tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def rerank(query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
    """Rerank `documents` by LLM-judged relevance to `query` (cross-encoder style).

    Input is a list of plain strings; output annotates each with its
    `rerank_score` (0 = most relevant).
    """
    docs = [{"id": str(i), "content": c} for i, c in enumerate(documents)]
    reranked = await _rerank(query, docs, top_n=top_n)
    return [{"content": d["content"], "rerank_score": d.get("rerank_score", 0)} for d in reranked]


@mcp.tool()
async def extract_claims(report: str) -> list[dict]:
    """Extract factual / definitional / statistical claims from a piece of prose.

    Useful for downstream faithfulness / hallucination checks.
    """
    claims = await _extract_claims(report)
    return [{"claim": c.claim, "claim_type": c.claim_type} for c in claims]


@mcp.tool()
async def attribute_claims(
    claims: list[str],
    sources: dict[str, str],
    threshold: float = 0.75,
) -> list[dict]:
    """For each claim string, find the best-matching source section via embedding similarity.

    Returns { claim, source_section | None, score }.
    Claims below `threshold` come back with source_section=None (unsourced).
    """
    from app.tools.attribution import ExtractedClaim

    parsed = [ExtractedClaim(claim=c, claim_type="factual") for c in claims]
    result = await _attribute(parsed, sources, threshold=threshold)
    return [
        {
            "claim": r["claim"],
            "source_section": r["source_section"],
            "score": r["score"],
        }
        for r in result
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> tuple[str, int]:
    """Tiny argparse: --transport {stdio,sse} [--port N]."""
    import argparse

    p = argparse.ArgumentParser(description="Research Assistant MCP server")
    p.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="stdio for Claude Desktop / LangGraph, sse|streamable-http for HTTP clients",
    )
    p.add_argument("--port", type=int, default=8765, help="only used with sse / streamable-http")
    args = p.parse_args()
    return args.transport, args.port


def main() -> None:
    transport, port = _parse_args()
    if transport == "stdio":
        log.info("starting MCP server transport=stdio")
        mcp.run()
    else:
        log.info("starting MCP server transport=%s port=%d", transport, port)
        # FastMCP supports `sse_port` / `streamable_http_port` via settings
        mcp.settings.port = port
        asyncio.run(mcp.run_async(transport=transport))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    main()
