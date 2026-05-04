"""Tests for LangChain-native pieces wired into the research graph.

* VectorCacheRetriever — standard BaseRetriever interface (sync + async)
* RecursiveCharacterTextSplitter — factory + chunking behaviour
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document as LCDocument
from langchain_core.retrievers import BaseRetriever


def _patch_hybrid_sources(monkeypatch, *, make_vector, make_bm25=None):
    """Patch the three pure functions the retriever calls under the hood.

    retrieve_similar → cosine path (used for cache-hit gating too)
    retrieve_bm25    → lexical path (fused into hybrid)
    """
    from app.tools import retriever as retriever_mod

    monkeypatch.setattr(retriever_mod, "retrieve_similar", make_vector)
    if make_bm25 is None:
        make_bm25 = lambda *a, **kw: []  # noqa: E731
    monkeypatch.setattr(retriever_mod, "retrieve_bm25", make_bm25)


@pytest.mark.asyncio
async def test_cache_retriever_is_a_langchain_runnable(monkeypatch):
    from app.tools import retriever as retriever_mod

    def fake_vector(query, top_k=None, max_age_days=None):  # noqa: ARG001
        return [
            {
                "id": "00000000-0000-0000-0000-000000000001",
                "topic": "t",
                "section": "s",
                "content": "hello world",
                "similarity": 0.9,
            }
        ]

    _patch_hybrid_sources(monkeypatch, make_vector=fake_vector)
    r = retriever_mod.VectorCacheRetriever(hybrid=False)
    assert isinstance(r, BaseRetriever)

    # Async path
    docs_async = await r.ainvoke("what")
    assert len(docs_async) == 1
    assert isinstance(docs_async[0], LCDocument)
    assert docs_async[0].page_content == "hello world"
    assert docs_async[0].metadata["similarity"] == 0.9

    # Sync path
    docs_sync = r.invoke("what")
    assert docs_sync[0].page_content == "hello world"


@pytest.mark.asyncio
async def test_cache_retriever_abatch_parallel(monkeypatch):
    """abatch() must return one result list per query, preserving order."""
    from app.tools import retriever as retriever_mod

    def fake_vector(query, top_k=None, max_age_days=None):  # noqa: ARG001
        return [
            {
                "id": "00000000-0000-0000-0000-000000000002",
                "topic": "t",
                "section": query,
                "content": f"matched:{query}",
                "similarity": 0.88,
            }
        ]

    _patch_hybrid_sources(monkeypatch, make_vector=fake_vector)

    r = retriever_mod.VectorCacheRetriever(hybrid=False)
    results = await r.abatch(["q1", "q2", "q3"], config={"max_concurrency": 2})
    assert [d[0].page_content for d in results] == [
        "matched:q1",
        "matched:q2",
        "matched:q3",
    ]


@pytest.mark.asyncio
async def test_cache_retriever_rrf_merges_vector_and_bm25(monkeypatch):
    """Hybrid retriever should fuse vector + BM25 results via RRF, not use only one."""
    from app.tools import retriever as retriever_mod

    def fake_vector(query, top_k=None, max_age_days=None):  # noqa: ARG001
        return [
            {"id": "v1", "topic": "t", "section": "s", "content": "V1", "similarity": 0.9},
            {"id": "shared", "topic": "t", "section": "s", "content": "SHARED", "similarity": 0.85},
        ]

    def fake_bm25(query, top_k=None, max_age_days=None):  # noqa: ARG001
        return [
            {"id": "b1", "topic": "t", "section": "s", "content": "B1", "bm25_score": 0.3},
            {"id": "shared", "topic": "t", "section": "s", "content": "SHARED", "bm25_score": 0.25},
        ]

    _patch_hybrid_sources(monkeypatch, make_vector=fake_vector, make_bm25=fake_bm25)

    r = retriever_mod.VectorCacheRetriever(hybrid=True, threshold=0.8)
    docs = await r.ainvoke("query")
    ids = [d.metadata["id"] for d in docs]
    # 'shared' appears in both → RRF rewards it most
    assert ids[0] == "shared"
    assert set(ids) == {"shared", "v1", "b1"}


def test_rrf_function_direct():
    """Reciprocal Rank Fusion arithmetic — standalone unit test."""
    from app.tools.retriever import reciprocal_rank_fusion

    list_a = [{"id": "1"}, {"id": "2"}]  # ranks 1, 2 in list A
    list_b = [{"id": "2"}, {"id": "3"}]  # ranks 1, 2 in list B

    fused = reciprocal_rank_fusion(list_a, list_b, k=10)
    ids_in_order = [d["id"] for d in fused]
    # "2" appears in both at top ranks → wins
    assert ids_in_order[0] == "2"
    # unique to A (rank 1) wins over unique to B (rank 2)
    assert ids_in_order[1] == "1"
    assert ids_in_order[2] == "3"


def test_text_splitter_chunks_on_paragraphs():
    from app.tools.splitter import get_text_splitter

    splitter = get_text_splitter()
    # Force a tiny chunk size by reconfiguring via chunk_size.
    # Default config is 800; a 2k text should split.
    long_text = ("A paragraph about LangGraph. " * 40 + "\n\n") + (
        "A paragraph about pgvector. " * 40
    )
    chunks = splitter.split_text(long_text)
    assert len(chunks) >= 2
    # Each chunk should stay within the configured max (+ overlap slack)
    for c in chunks:
        assert len(c) <= 800 + 200


def test_text_splitter_short_text_yields_single_chunk():
    from app.tools.splitter import get_text_splitter

    splitter = get_text_splitter()
    chunks = splitter.split_text("Tiny text.")
    assert chunks == ["Tiny text."]
