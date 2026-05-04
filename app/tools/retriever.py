"""pgvector-based retriever for CRAG.

Stores research summaries as embeddings after each Tavily search.
On the next research run, first checks the vector store for relevant content
before falling back to a live web search.

Cache TTL:
    `retrieve_similar` / `retrieve_relevant` only return documents whose
    `created_at` is within the last `max_age_days` (default: from settings).
    This prevents stale cache hits on time-sensitive topics.

Security:
    Uses SQLAlchemy bindparam with pgvector.sqlalchemy.Vector type so the
    embedding is bound as a typed parameter, not string-interpolated.
"""

import asyncio
import logging
import uuid

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document as LCDocument
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, text

from app.config import settings
from app.models.db import Document
from app.models.engine import SessionLocal

log = logging.getLogger(__name__)

# text-embedding-3-small: 1536 dims, cheapest; matches the migration's Vector(1536)
EMBEDDING_DIMS = 1536
EMBEDDING_MODEL = "text-embedding-3-small"

_embeddings = OpenAIEmbeddings(
    api_key=settings.openai_api_key,
    model=EMBEDDING_MODEL,
)


def embed_text(text_: str) -> list[float]:
    return _embeddings.embed_query(text_)


def validate_embedding_dimensions() -> None:
    """Verify `documents.embedding_vec` actually uses EMBEDDING_DIMS dimensions.

    Called at app startup. If the pgvector column was created with a
    different size (e.g. a migration was forgotten), we fail loudly instead
    of silently breaking similarity search later.
    """
    sql = text(
        """
        SELECT atttypmod
        FROM pg_attribute
        WHERE attrelid = 'documents'::regclass
          AND attname  = 'embedding_vec'
        """
    )
    with SessionLocal() as db:
        row = db.execute(sql).fetchone()
    if row is None:
        log.warning("embedding_column_missing — skipping dim check (table not migrated?)")
        return
    # pgvector stores dim in atttypmod (-1 if unset)
    actual = int(row[0])
    if actual not in (-1, EMBEDDING_DIMS):
        raise RuntimeError(
            f"Embedding dim mismatch: column={actual} code={EMBEDDING_DIMS}. "
            f"Run the correct migration or change EMBEDDING_MODEL in retriever.py."
        )


_MIN_DOCUMENT_CHARS = 50  # L-3: junk-content floor to prevent index poisoning
_MAX_DOCUMENT_CHARS = 20_000


def save_document(
    topic: str,
    section: str,
    content: str,
    *,
    parent_doc_id: uuid.UUID | None = None,
    parent_content: str | None = None,
    chunk_offset_start: int | None = None,
    chunk_offset_end: int | None = None,
) -> None:
    """Embed `content` and persist it as a Document row.

    Optional `parent_*` args enable parent-document retrieval (G-9):
      * `parent_doc_id` groups chunks from the same source.
      * `parent_content` stores the full source for context-windowed reads.
      * `chunk_offset_{start,end}` locate the chunk within the parent.

    Rejects junk-short content (L-3).
    """
    if not content or len(content.strip()) < _MIN_DOCUMENT_CHARS:
        log.debug("save_document skipped: content below %d chars", _MIN_DOCUMENT_CHARS)
        return
    content = content[:_MAX_DOCUMENT_CHARS]
    vector = embed_text(content)
    with SessionLocal() as db:
        doc = Document(
            id=uuid.uuid4(),
            topic=topic,
            section=section,
            content=content,
            embedding_vec=vector,
            parent_doc_id=parent_doc_id,
            parent_content=(parent_content or "")[: _MAX_DOCUMENT_CHARS * 4] or None,
            chunk_offset_start=chunk_offset_start,
            chunk_offset_end=chunk_offset_end,
        )
        db.add(doc)
        db.commit()


def get_parent_context(
    chunk_id: str,
    window_chars: int = 400,
) -> str | None:
    """Return a window of `parent_content` around a chunk's offsets.

    Caller: retriever/researcher picks a chunk via similarity, then calls
    this to enrich it with surrounding paragraphs before feeding the LLM.
    """
    sql = text(
        "SELECT parent_content, chunk_offset_start, chunk_offset_end FROM documents WHERE id = :id"
    )
    with SessionLocal() as db:
        row = db.execute(sql, {"id": chunk_id}).fetchone()
    if row is None or row.parent_content is None:
        return None
    start = max(0, (row.chunk_offset_start or 0) - window_chars)
    end = (row.chunk_offset_end or len(row.parent_content)) + window_chars
    return row.parent_content[start:end]


# Cosine distance (<=>) — we convert to similarity (1 - distance) in SELECT.
_SIMILAR_SQL = text(
    """
    SELECT id, topic, section, content,
           1 - (embedding_vec <=> :vec) AS similarity
    FROM documents
    WHERE embedding_vec IS NOT NULL
      AND created_at > now() - make_interval(days => :max_age_days)
    ORDER BY embedding_vec <=> :vec
    LIMIT :k
    """
).bindparams(
    bindparam("vec", type_=Vector(EMBEDDING_DIMS)),
    bindparam("k"),
    bindparam("max_age_days"),
)


def retrieve_similar(
    query: str,
    top_k: int | None = None,
    max_age_days: int | None = None,
) -> list[dict]:
    """Return top-k most similar documents with cosine similarity scores.

    Args:
        query: Natural-language query to embed and match against.
        top_k: Maximum number of documents to return (default: settings.retriever_top_k).
        max_age_days: Exclude documents older than this (default: settings.cache_ttl_days).
    """
    if top_k is None:
        top_k = settings.retriever_top_k
    if max_age_days is None:
        max_age_days = settings.cache_ttl_days

    vector = embed_text(query)
    with SessionLocal() as db:
        rows = db.execute(
            _SIMILAR_SQL,
            {"vec": vector, "k": top_k, "max_age_days": max_age_days},
        ).fetchall()

    return [
        {
            "id": str(row.id),
            "topic": row.topic,
            "section": row.section,
            "content": row.content,
            "similarity": float(row.similarity),
        }
        for row in rows
    ]


def retrieve_relevant(
    query: str,
    threshold: float | None = None,
    max_age_days: int | None = None,
) -> list[dict]:
    """Return documents above the similarity threshold — empty list = cache miss."""
    if threshold is None:
        threshold = settings.similarity_threshold
    docs = retrieve_similar(query, max_age_days=max_age_days)
    return [d for d in docs if d["similarity"] >= threshold]


# ---------------------------------------------------------------------------
# BM25 / lexical retrieval via Postgres full-text search.
#
# Vector similarity fails on lexical queries ("Q3 2024 earnings", numeric
# references, rare named entities). Pairing vector with BM25 and merging
# via Reciprocal Rank Fusion (RRF) gives robust results on both semantic
# and keyword-heavy queries — this is what production RAG systems ship.
# ---------------------------------------------------------------------------

_BM25_SQL = text(
    """
    SELECT id, topic, section, content,
           ts_rank_cd(content_tsv, websearch_to_tsquery('english', :q)) AS score
    FROM documents
    WHERE content_tsv @@ websearch_to_tsquery('english', :q)
      AND created_at > now() - make_interval(days => :max_age_days)
    ORDER BY score DESC
    LIMIT :k
    """
).bindparams(
    bindparam("q"),
    bindparam("k"),
    bindparam("max_age_days"),
)


def retrieve_bm25(
    query: str,
    top_k: int | None = None,
    max_age_days: int | None = None,
) -> list[dict]:
    """Lexical retrieval via Postgres `ts_rank_cd`."""
    if top_k is None:
        top_k = settings.retriever_top_k
    if max_age_days is None:
        max_age_days = settings.cache_ttl_days

    with SessionLocal() as db:
        rows = db.execute(
            _BM25_SQL,
            {"q": query, "k": top_k, "max_age_days": max_age_days},
        ).fetchall()

    return [
        {
            "id": str(row.id),
            "topic": row.topic,
            "section": row.section,
            "content": row.content,
            "bm25_score": float(row.score),
        }
        for row in rows
    ]


def reciprocal_rank_fusion(
    *ranked_lists: list[dict],
    k: int = 60,
    id_field: str = "id",
) -> list[dict]:
    """Merge multiple ranked lists via Reciprocal Rank Fusion.

    RRF score for a doc = sum(1 / (k + rank_in_list)) across all lists it appears in.
    The standard constant k=60 dampens high-rank dominance. Lists can have
    different scales (cosine ∈ [0,1] vs ts_rank_cd arbitrary) — RRF normalises.

    Reference: Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion outperforms
    Condorcet and individual Rank Learning Methods", SIGIR 2009.
    """
    merged: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            doc_id = doc[id_field]
            score = 1.0 / (k + rank)
            if doc_id not in merged:
                merged[doc_id] = {**doc, "rrf_score": 0.0}
            merged[doc_id]["rrf_score"] += score
    return sorted(merged.values(), key=lambda d: d["rrf_score"], reverse=True)


def retrieve_hybrid(
    query: str,
    top_k: int | None = None,
    max_age_days: int | None = None,
) -> list[dict]:
    """Hybrid retrieval: vector + BM25, fused via RRF.

    Fetches 2×top_k from each index so RRF has enough overlap to work with,
    then truncates to top_k.
    """
    if top_k is None:
        top_k = settings.retriever_top_k

    # Over-fetch so RRF sees a richer candidate pool.
    k_each = max(top_k * 2, 6)
    vector_hits = retrieve_similar(query, top_k=k_each, max_age_days=max_age_days)
    bm25_hits = retrieve_bm25(query, top_k=k_each, max_age_days=max_age_days)
    fused = reciprocal_rank_fusion(vector_hits, bm25_hits)
    return fused[:top_k]


# ---------------------------------------------------------------------------
# LangChain Runnable retriever
#
# Wraps `retrieve_relevant` so the cache can be slotted into any LCEL chain.
# Exposes both sync/async interfaces, returns `langchain_core.documents.Document`
# (page_content + metadata) — the standard shape every LangChain component
# expects. This makes the cache interoperable with LangChain prebuilt chains
# (MultiQueryRetriever, ContextualCompressionRetriever, etc.) without any
# custom adapters.
# ---------------------------------------------------------------------------


class VectorCacheRetriever(BaseRetriever):
    """Hybrid BM25+vector retriever fused via RRF, exposed as BaseRetriever.

    Pipeline:
        query
          ├─ pgvector cosine  (top 2k)
          └─ Postgres BM25    (top 2k)
                   │
                   └──> Reciprocal Rank Fusion → top_k

    Cache-hit gate: `threshold` is applied to the vector path only —
    BM25 rank_cd scores are not comparable to cosine similarity, so the
    miss/hit decision is driven by vector similarity, but **what** we
    return is the RRF-fused set. This gives keyword queries a real shot
    without diluting semantic similarity.

    Attributes:
        threshold: minimum cosine similarity to consider the cache populated.
        max_age_days: discard documents older than this (TTL).
        hybrid: set False to bypass BM25 (vector-only) — useful for A/B.
    """

    threshold: float | None = None
    max_age_days: int | None = None
    hybrid: bool = True

    model_config = {"arbitrary_types_allowed": True}

    def _to_lc_docs(self, raw: list[dict]) -> list[LCDocument]:
        return [
            LCDocument(
                page_content=d["content"],
                metadata={
                    "id": d["id"],
                    "topic": d["topic"],
                    "section": d["section"],
                    "similarity": d.get("similarity"),
                    "bm25_score": d.get("bm25_score"),
                    "rrf_score": d.get("rrf_score"),
                },
            )
            for d in raw
        ]

    def _fetch(self, query: str) -> list[dict]:
        threshold = self.threshold if self.threshold is not None else settings.similarity_threshold

        # 1) Cache-hit decision: any vector result above threshold?
        vector_hits = retrieve_similar(query, max_age_days=self.max_age_days)
        if not any(h["similarity"] >= threshold for h in vector_hits):
            return []  # cache miss — let caller fall back to Tavily

        if not self.hybrid:
            return [h for h in vector_hits if h["similarity"] >= threshold]

        # 2) Fuse vector with BM25 via RRF.
        return retrieve_hybrid(query, max_age_days=self.max_age_days)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,  # noqa: ARG002
    ) -> list[LCDocument]:
        return self._to_lc_docs(self._fetch(query))

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,  # noqa: ARG002
    ) -> list[LCDocument]:
        raw = await asyncio.to_thread(self._fetch, query)
        return self._to_lc_docs(raw)
