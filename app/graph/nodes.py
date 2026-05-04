"""Graph nodes for the Research Assistant.

Each node is a pure async function: (ResearchState) -> dict[str, Any]
The returned dict is merged into the state by LangGraph reducers.

Node chain (CRAG):
  planner -> [human interrupt] -> researcher -> grader
                                                    |
                                        relevant: writer -> reviewer
                                        irrelevant: web_search -> writer -> reviewer
                                                              |
                                              ok: formatter -> END
                                              needs_revision: researcher (max 2x)

Key design choices:
  * Structured output (planner / grader / reviewer) via `with_structured_output`
    — no fragile string parsing, no hallucinated formats.
  * User input (topic, section) is wrapped in XML tags inside prompts so
    injected instructions are treated as data. See PROMPT_INJECTION_NOTE.
  * Retry with exponential backoff on transient OpenAI/Tavily failures.
  * Bounded concurrency for embeddings via asyncio.Semaphore.
"""

import asyncio
import functools
import logging
import re as _re
import time
from collections.abc import Awaitable, Callable
from typing import Literal

import httpx
import openai
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings
from app.graph.prompts import (
    FORMATTER_PROMPT,
    GRADER_PROMPT,
    KG_PROMPT,
    PLANNER_PROMPT,
    REFINE_PROMPT,
    RESEARCHER_PROMPT,
    REVIEWER_PROMPT,
    WRITER_PROMPT,
)
from app.graph.state import ResearchState
from app.llm.router import NodeRole, get_llm
from app.observability import (
    CACHE_HIT_TOTAL,
    CACHE_MISS_TOTAL,
    GRAPH_NODE_DURATION,
    GRAPH_NODES_TOTAL,
)
from app.tools.attribution import attribute, extract_claims, inject_footnotes
from app.tools.reranker import rerank
from app.tools.retriever import VectorCacheRetriever, retrieve_relevant, save_document
from app.tools.search import get_search_tool
from app.tools.splitter import get_text_splitter

log = logging.getLogger(__name__)


def _node_timeout(node_name: str) -> float:
    """Read the per-node timeout budget from settings (seconds)."""
    attr = f"{node_name}_timeout_seconds"
    return float(getattr(settings, attr, settings.graph_timeout_seconds))


def _instrument(node_name: str) -> Callable:
    """Decorator: Prometheus histogram + counter + per-node timeout + log.

    Each node gets its own `asyncio.timeout` budget from settings so a
    stuck LLM call on one phase can't consume the whole graph's time.
    """

    def wrap(fn: Callable[..., Awaitable[dict]]) -> Callable[..., Awaitable[dict]]:
        @functools.wraps(fn)
        async def inner(state: ResearchState) -> dict:
            thread_id = state.get("thread_id", "unknown")
            budget = _node_timeout(node_name)
            start = time.perf_counter()
            log.info(
                "node_start node=%s thread_id=%s timeout_s=%.0f",
                node_name,
                thread_id,
                budget,
            )
            try:
                async with asyncio.timeout(budget):
                    result = await fn(state)
            except TimeoutError:
                GRAPH_NODES_TOTAL.labels(node=node_name, status="error").inc()
                log.error(
                    "node_timeout node=%s thread_id=%s budget_s=%.0f",
                    node_name,
                    thread_id,
                    budget,
                )
                raise
            except Exception:
                GRAPH_NODES_TOTAL.labels(node=node_name, status="error").inc()
                log.exception("node_failed node=%s thread_id=%s", node_name, thread_id)
                raise
            duration = time.perf_counter() - start
            GRAPH_NODE_DURATION.labels(node=node_name).observe(duration)
            GRAPH_NODES_TOTAL.labels(node=node_name, status="success").inc()
            log.info(
                "node_end node=%s thread_id=%s duration_ms=%.1f",
                node_name,
                thread_id,
                duration * 1000,
            )
            return result

        return inner

    return wrap


# Retry policy applied to every external-API call below.
# Retries on OpenAI transient errors and any httpx network error.
_retryable = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
    httpx.HTTPError,
)

_retry_llm = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(_retryable),
    reraise=True,
)

# ---------------------------------------------------------------------------
# LLM instances — per-role with provider fallback (OpenAI → Anthropic).
# `get_llm(role)` returns a LangChain Runnable with:
#   * model picked from settings.model_<role> (falls back to openai_model)
#   * fallback to Anthropic if ANTHROPIC_API_KEY is set
#   * temperature from settings.llm_temperature (or deterministic flavour)
# ---------------------------------------------------------------------------

_planner_llm = get_llm(NodeRole.PLANNER)
_researcher_llm = get_llm(NodeRole.RESEARCHER)
_grader_llm = get_llm(NodeRole.GRADER, deterministic=True)
_writer_llm = get_llm(NodeRole.WRITER)
_reviewer_llm = get_llm(NodeRole.REVIEWER, deterministic=True)
_formatter_llm = get_llm(NodeRole.FORMATTER)

# Concurrency guard for embedding calls — keeps OpenAI Embeddings API safe
# under parallel researcher/web_search flows.
_embedding_sem = asyncio.Semaphore(settings.embedding_max_concurrency)


# ---------------------------------------------------------------------------
# Structured-output schemas
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    sections: list[str] = Field(
        ...,
        description="4-6 concise research section titles, no numbering.",
        min_length=1,
    )


class GradeVerdict(BaseModel):
    grade: Literal["relevant", "irrelevant"] = Field(
        ..., description="Whether retrieved content is relevant and sufficient."
    )


class ReviewVerdict(BaseModel):
    approved: bool = Field(..., description="True if the report is publishable.")
    feedback: str = Field(
        default="",
        description="Brief actionable feedback if not approved, else empty.",
    )


class KGEntity(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    type: Literal["tool", "company", "concept", "person", "product", "place", "metric", "other"] = (
        Field(default="other")
    )


class KGRelation(BaseModel):
    from_: str = Field(..., alias="from", min_length=1, max_length=200)
    relation: str = Field(..., min_length=1, max_length=100)
    to: str = Field(..., min_length=1, max_length=200)

    model_config = {"populate_by_name": True}


class KnowledgeGraph(BaseModel):
    entities: list[KGEntity] = Field(default_factory=list, max_length=50)
    relations: list[KGRelation] = Field(default_factory=list, max_length=80)


# ---------------------------------------------------------------------------
# Prompt-injection note:
#
# User-controlled strings (topic, section, search results) are always wrapped
# in XML tags inside prompts. System prompts instruct the model to treat tag
# contents as data, not instructions. This is not bulletproof, but stops the
# trivial "Ignore previous instructions, ..." attacks.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Node: planner
# ---------------------------------------------------------------------------

_planner_chain = PLANNER_PROMPT | _planner_llm.with_structured_output(ResearchPlan)


# Empirically, a typical section costs ~$0.01-0.02 in total LLM work on
# gpt-4o-mini (researcher summary + writer + grader share). At budget X,
# plan for ~X / cost_per_section sections clamped to the 4..6 band.
_EST_USD_PER_SECTION = 0.015


def _budget_aware_topic(topic: str, budget_usd: float) -> str:
    if budget_usd <= 0:
        return topic
    max_sections = max(2, min(6, int(budget_usd / _EST_USD_PER_SECTION)))
    return (
        f"{topic}\n\n"
        f"[BUDGET HINT, data only — plan at most {max_sections} sections to stay "
        f"within a ${budget_usd:.2f} ceiling. Shorter is fine.]"
    )


@_retry_llm
async def _planner_invoke(topic: str) -> ResearchPlan:
    return await _planner_chain.ainvoke({"topic": topic})


@_instrument("planner")
async def planner_node(state: ResearchState) -> dict:
    topic = state["topic"]
    budget = float(state.get("budget_usd") or 0.0)
    topic_for_llm = _budget_aware_topic(topic, budget)

    # G-12: surface the user's prior related research (if any) as planner context.
    # The api_key_hash lives in `thread_id` state only indirectly — routes.py
    # passes it via the `retrieval_grades["_caller_hash"]` slot to avoid
    # changing state schema; we read it back here opportunistically.
    prior_hash = _caller_hash_from_state(state)
    if prior_hash:
        from app.cache import history as _history

        prior = await asyncio.to_thread(_history.fetch_relevant, prior_hash, topic)
        if prior:
            prior_block = "\n".join(f"- {p['topic']}: {p['summary'][:200]}" for p in prior)
            topic_for_llm = (
                f"{topic_for_llm}\n\n"
                f"[PRIOR RESEARCH BY THIS USER, data only — avoid duplicating these "
                f"findings; build on them instead:]\n{prior_block}"
            )

    plan_obj = await _planner_invoke(topic_for_llm)
    plan = [s.strip() for s in plan_obj.sections if s and s.strip()]
    if budget > 0:
        max_sections = max(2, min(6, int(budget / _EST_USD_PER_SECTION)))
        plan = plan[:max_sections]
    return {"plan": plan, "human_approved": False}


def _caller_hash_from_state(state: ResearchState) -> str | None:
    """Retrieve the caller's api_key_hash from state, if the route seeded it."""
    meta = state.get("retrieval_grades") or {}
    if isinstance(meta, dict):
        return meta.get("_caller_hash") if isinstance(meta.get("_caller_hash"), str) else None
    return None


# ---------------------------------------------------------------------------
# Node: researcher — CRAG retrieval with pgvector cache
#
# LangChain glue used here:
#   * VectorCacheRetriever (BaseRetriever) — the pgvector cache exposed as
#     a standard LangChain Runnable; slots into any LCEL chain.
#   * RecursiveCharacterTextSplitter — splits fresh Tavily results into
#     overlapping chunks before embedding, so retrieval matches specific
#     facts instead of a blurred average of the whole response.
#   * LCEL `prompt | llm | parser` summariser chain, invoked via `.abatch()`
#     on the list of irrelevant sections — a single call fans out in parallel
#     with built-in concurrency control (max_concurrency=settings.embedding_max_concurrency).
# ---------------------------------------------------------------------------

search_tool = get_search_tool(max_results=settings.tavily_max_results)
_cache_retriever = VectorCacheRetriever()
_text_splitter = get_text_splitter()

# G-5: circuit breaker on Tavily. After 5 consecutive failures in 60s,
# open the circuit — callers get CircuitBreakerOpen immediately and can
# fall back to cache-only paths instead of waiting tenacity's full retry
# budget × every section.
from app.tools.circuit_breaker import CircuitBreaker, CircuitBreakerOpen  # noqa: E402

_tavily_breaker = CircuitBreaker(
    name="tavily",
    failure_threshold=5,
    recovery_timeout_s=120.0,
)

# LCEL summariser chain — dict in, str out. Callable via .ainvoke() or .abatch().
_researcher_chain = (RESEARCHER_PROMPT | _researcher_llm | StrOutputParser()).with_config(
    tags=["researcher-summariser"]
)


@_retry_llm
async def _tavily_search_raw(query: str):
    # Tavily tool is sync; run in a thread so we don't block the loop.
    return await asyncio.to_thread(search_tool.invoke, query)


async def _tavily_search(query: str):
    """Tavily call wrapped in the shared circuit breaker (G-5).

    On `CircuitBreakerOpen`, we return an empty result instead of raising —
    the caller treats it like "no web hits", letting the cache-only path
    still produce a degraded-but-valid report.
    """
    try:
        return await _tavily_breaker.call(lambda: _tavily_search_raw(query))
    except CircuitBreakerOpen:
        log.warning("tavily_breaker_open — returning empty result for query=%r", query)
        return []


def _join_tavily(raw) -> str:
    if isinstance(raw, list):
        return "\n\n".join(r.get("content", "") for r in raw if isinstance(r, dict))
    return str(raw)


async def _rerank_tavily(query: str, raw) -> str:
    """Rerank Tavily snippets by relevance to query, then join.

    Symmetry with cache-hit path: both retrieval modes pass through the
    LLM cross-encoder before the summariser sees them. Without this, cached
    answers would enjoy reranker-picked top-k while fresh web searches got
    Tavily's raw ordering.

    If Tavily returned ≤1 snippet the reranker is a no-op, so we skip the
    extra LLM call in the common case.
    """
    if not isinstance(raw, list) or len(raw) <= 1:
        return _join_tavily(raw)

    snippets = [
        {"id": str(i), "content": r.get("content", "")}
        for i, r in enumerate(raw)
        if isinstance(r, dict)
    ]
    if len(snippets) <= 1:
        return _join_tavily(raw)

    reranked = await rerank(query, snippets, top_n=settings.retriever_top_k)
    return "\n\n".join(r["content"] for r in reranked)


async def _save_chunks(topic: str, section: str, content: str) -> None:
    """Split long content into chunks and save each with parent-doc linkage (G-9).

    Every chunk carries the shared `parent_doc_id` and the full parent text
    plus its offsets within the parent. At retrieval time the caller can
    pull `get_parent_context(chunk_id)` to enrich the chunk with surrounding
    paragraphs — much better signal for the downstream summariser.

    Bounded by the embedding semaphore so chunk count doesn't blow up
    OpenAI rate limits.
    """
    import uuid as _uuid

    chunks = _text_splitter.split_text(content) or [content]
    parent_doc_id = _uuid.uuid4()
    # Compute offsets for each chunk within the parent (best-effort: the
    # splitter doesn't report offsets so we locate the first occurrence).
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for chunk in chunks:
        start = content.find(chunk, cursor)
        if start == -1:
            start = cursor
        end = start + len(chunk)
        offsets.append((start, end))
        cursor = end

    async def _save_one(chunk: str, start: int, end: int) -> None:
        async with _embedding_sem:
            await asyncio.to_thread(
                save_document,
                topic,
                section,
                chunk,
                parent_doc_id=parent_doc_id,
                parent_content=content,
                chunk_offset_start=start,
                chunk_offset_end=end,
            )

    await asyncio.gather(
        *[_save_one(c, s, e) for c, (s, e) in zip(chunks, offsets, strict=True)],
        return_exceptions=True,
    )


@_instrument("researcher")
async def researcher_node(state: ResearchState) -> dict:
    """For each section: try pgvector cache first, fall back to Tavily on miss.

    Cache hits return concatenated chunk text (from LangChain retriever).
    Cache misses hit Tavily, summarise via LCEL `.abatch()`, and persist
    split chunks to the cache for future retrieval.
    """
    plan: list[str] = state["plan"]
    topic: str = state["topic"]

    # Phase 1: concurrent cache lookups — LangChain retriever exposed as Runnable.
    queries = [f"{topic}: {s}" for s in plan]
    # .abatch() runs queries in parallel with built-in concurrency throttling.
    cached_results = await _cache_retriever.abatch(
        queries,
        config={"max_concurrency": settings.embedding_max_concurrency},
    )

    async def _rerank_hit(query: str, docs) -> list:
        """Rerank a single cache hit's docs; cheap because top-k is small.

        Short-circuit when there's at most one document — reranking a list
        of one is a no-op with the LLM backend, but still pays a round-trip.
        Same shortcut the reranker itself has internally; we apply it here
        too so the task list skips creating the coroutine entirely.
        """
        raw = [{"id": d.metadata.get("id"), "content": d.page_content, **d.metadata} for d in docs]
        if len(raw) <= 1:
            return raw
        return await rerank(query, raw, top_n=settings.retriever_top_k)

    cache_hits: dict[str, str] = {}
    misses: list[str] = []
    rerank_tasks: list[tuple[str, asyncio.Task]] = []
    for section, docs, query in zip(plan, cached_results, queries, strict=True):
        if docs:
            CACHE_HIT_TOTAL.inc()
            rerank_tasks.append((section, asyncio.create_task(_rerank_hit(query, docs))))
        else:
            CACHE_MISS_TOTAL.inc()
            misses.append(section)

    for section, task in rerank_tasks:
        try:
            ranked = await task
            cache_hits[section] = "\n\n".join(d["content"] for d in ranked)
        except Exception:
            log.exception("rerank_failed section=%s", section)
            # fall through — section will be reprocessed as a miss? keep as hit without rerank
            cache_hits[section] = "\n\n".join(
                d.page_content for d in cached_results[plan.index(section)]
            )

    # Phase 2: Tavily fetch + LLM summarise + save chunks, fan out via .abatch().
    search_results: dict[str, str] = dict(cache_hits)
    retrieval_grades: dict[str, str] = {s: "relevant" for s in cache_hits}

    if misses:
        raw_batch = await asyncio.gather(
            *[_tavily_search(f"{topic}: {s}") for s in misses],
            return_exceptions=True,
        )
        # Symmetric reranking: Tavily snippets → LLM cross-encoder → top-k,
        # same treatment the cache-hit path gets after retrieve_hybrid().
        rerank_tasks = [
            _rerank_tavily(f"{topic}: {section}", raw)
            for section, raw in zip(misses, raw_batch, strict=True)
            if not isinstance(raw, BaseException)
        ]
        reranked_snippets = await asyncio.gather(*rerank_tasks, return_exceptions=True)

        summariser_inputs = []
        active_misses = []
        rerank_iter = iter(reranked_snippets)
        for section, raw in zip(misses, raw_batch, strict=True):
            if isinstance(raw, BaseException):
                log.exception("tavily_failed section=%s", section, exc_info=raw)
                continue
            joined = next(rerank_iter)
            if isinstance(joined, BaseException):
                log.exception("rerank_tavily_failed section=%s", section, exc_info=joined)
                joined = _join_tavily(raw)
            summariser_inputs.append({"section": section, "results": joined})
            active_misses.append(section)

        summaries = await _researcher_chain.abatch(
            summariser_inputs,
            config={"max_concurrency": settings.embedding_max_concurrency},
            return_exceptions=True,
        )

        save_tasks = []
        for section, summary in zip(active_misses, summaries, strict=True):
            if isinstance(summary, BaseException):
                log.exception("summariser_failed section=%s", section, exc_info=summary)
                continue
            search_results[section] = summary
            retrieval_grades[section] = "irrelevant"
            save_tasks.append(_save_chunks(topic, section, summary))

        await asyncio.gather(*save_tasks, return_exceptions=True)

    return {"search_results": search_results, "retrieval_grades": retrieval_grades}


# ---------------------------------------------------------------------------
# Node: grader — verify cache hit quality (temperature=0.0)
# ---------------------------------------------------------------------------

_grader_chain = GRADER_PROMPT | _grader_llm.with_structured_output(GradeVerdict)


@_retry_llm
async def _grade_invoke(section: str, content: str) -> GradeVerdict:
    return await _grader_chain.ainvoke({"section": section, "content": content})


@_instrument("grader")
async def grader_node(state: ResearchState) -> dict:
    """Re-grade cache hits with the deterministic LLM.

    Sections already marked 'irrelevant' (fresh web results) skip the LLM call.
    """
    search_results = state.get("search_results", {})
    retrieval_grades = state.get("retrieval_grades", {})

    async def _grade_section(section: str, content: str) -> tuple[str, str]:
        if retrieval_grades.get(section) == "irrelevant":
            return section, "irrelevant"
        verdict = await _grade_invoke(section, content)
        return section, verdict.grade

    results = await asyncio.gather(
        *[_grade_section(s, c) for s, c in search_results.items()],
        return_exceptions=True,
    )

    grades: dict[str, str] = {}
    for item in results:
        if isinstance(item, BaseException):
            log.exception("grader_section_failed", exc_info=item)
            continue
        section, grade = item
        grades[section] = grade

    return {"retrieval_grades": grades}


# ---------------------------------------------------------------------------
# Node: web_search — CRAG corrective fetch for irrelevant sections
# ---------------------------------------------------------------------------


@_instrument("adaptive_retrieval")
async def adaptive_retrieval_node(state: ResearchState) -> dict:
    """Retry cache retrieval with doubled top_k on sections graded irrelevant (G-8).

    Runs before `web_search`: if an adaptive retrieval lift flips the
    section to 'relevant', we save a Tavily call. Otherwise proceed to
    web_search normally.
    """
    search_results = dict(state.get("search_results", {}))
    grades = dict(state.get("retrieval_grades", {}))
    topic = state["topic"]

    depth = int(state.get("retrieval_depth_count") or 0)
    if depth >= settings.max_retrieval_depth:
        return {}

    irrelevant = [s for s, g in grades.items() if g == "irrelevant"]
    if not irrelevant:
        return {}

    top_k = settings.retriever_top_k * (2 ** (depth + 1))

    async def _deeper(section: str) -> tuple[str, str, str]:
        query = f"{topic}: {section}"
        raw = await asyncio.to_thread(
            retrieve_relevant,
            query,
            # Use a slightly relaxed threshold when deepening — the top-k
            # widens the candidate pool; we pay for it by requiring a real
            # cache hit to avoid fabrication.
            settings.similarity_threshold - 0.05,
            None,
        )
        if not raw:
            return section, "", "irrelevant"
        combined = "\n\n".join(d["content"] for d in raw[:top_k])
        return section, combined, "relevant"

    results = await asyncio.gather(*[_deeper(s) for s in irrelevant], return_exceptions=True)
    for item in results:
        if isinstance(item, BaseException):
            log.exception("adaptive_retrieval_section_failed", exc_info=item)
            continue
        section, content, grade = item
        if grade == "relevant":
            CACHE_HIT_TOTAL.inc()
            search_results[section] = content
            grades[section] = "relevant"

    return {
        "search_results": search_results,
        "retrieval_grades": grades,
        "retrieval_depth_count": depth + 1,
    }


@_instrument("web_search")
async def web_search_node(state: ResearchState) -> dict:
    """Re-fetch sections graded 'irrelevant' from the web.

    Uses the LCEL summariser chain's `.abatch()` to fan out over sections
    with built-in concurrency control — same pattern as researcher_node,
    reusing `_researcher_chain` rather than re-creating the chain inline.
    """
    topic = state["topic"]
    search_results = dict(state.get("search_results", {}))
    grades = state.get("retrieval_grades", {})
    irrelevant = [s for s, g in grades.items() if g == "irrelevant"]
    if not irrelevant:
        return {"search_results": search_results}

    raw_batch = await asyncio.gather(
        *[_tavily_search(f"{topic}: {s}") for s in irrelevant],
        return_exceptions=True,
    )
    rerank_tasks = [
        _rerank_tavily(f"{topic}: {section}", raw)
        for section, raw in zip(irrelevant, raw_batch, strict=True)
        if not isinstance(raw, BaseException)
    ]
    reranked_snippets = await asyncio.gather(*rerank_tasks, return_exceptions=True)

    active_sections, summariser_inputs = [], []
    rerank_iter = iter(reranked_snippets)
    for section, raw in zip(irrelevant, raw_batch, strict=True):
        if isinstance(raw, BaseException):
            log.exception("web_search_tavily_failed section=%s", section, exc_info=raw)
            continue
        joined = next(rerank_iter)
        if isinstance(joined, BaseException):
            log.exception("web_search_rerank_failed section=%s", section, exc_info=joined)
            joined = _join_tavily(raw)
        active_sections.append(section)
        summariser_inputs.append({"section": section, "results": joined})

    summaries = await _researcher_chain.abatch(
        summariser_inputs,
        config={"max_concurrency": settings.embedding_max_concurrency},
        return_exceptions=True,
    )

    save_tasks = []
    for section, summary in zip(active_sections, summaries, strict=True):
        if isinstance(summary, BaseException):
            log.exception("web_search_summariser_failed section=%s", section, exc_info=summary)
            continue
        search_results[section] = summary
        save_tasks.append(_save_chunks(topic, section, summary))

    await asyncio.gather(*save_tasks, return_exceptions=True)
    return {"search_results": search_results}


# ---------------------------------------------------------------------------
# Node: writer
# ---------------------------------------------------------------------------

_writer_chain = (WRITER_PROMPT | _writer_llm | StrOutputParser()).with_config(tags=["writer"])


@_retry_llm
async def _writer_invoke(topic: str, section: str, notes: str, feedback_instruction: str) -> str:
    return await _writer_chain.ainvoke(
        {
            "topic": topic,
            "section": section,
            "notes": notes,
            "feedback_instruction": feedback_instruction,
        }
    )


@_instrument("writer")
async def writer_node(state: ResearchState) -> dict:
    topic = state["topic"]
    search_results = state.get("search_results", {})
    review_feedback = state.get("review_feedback", "")
    feedback_instruction = (
        f"Previous reviewer feedback to address:\n{review_feedback}"
        if review_feedback
        else "Write the best version you can."
    )

    async def _write_section(section: str, notes: str) -> tuple[str, str]:
        text = await _writer_invoke(topic, section, notes, feedback_instruction)
        return section, text

    results = await asyncio.gather(
        *[_write_section(s, n) for s, n in search_results.items()],
        return_exceptions=True,
    )

    sections: dict[str, str] = {}
    for item in results:
        if isinstance(item, BaseException):
            log.exception("writer_section_failed", exc_info=item)
            continue
        section, text = item
        sections[section] = text

    return {"sections": sections, "review_feedback": ""}


# ---------------------------------------------------------------------------
# Node: reviewer (structured output)
# ---------------------------------------------------------------------------

_reviewer_chain = REVIEWER_PROMPT | _reviewer_llm.with_structured_output(ReviewVerdict)


@_retry_llm
async def _reviewer_invoke(topic: str, report: str) -> ReviewVerdict:
    return await _reviewer_chain.ainvoke({"topic": topic, "report": report})


@_instrument("reviewer")
async def reviewer_node(state: ResearchState) -> dict:
    topic = state["topic"]
    sections = state.get("sections", {})
    report_text = "\n\n".join(f"## {t}\n{b}" for t, b in sections.items())

    # G-11: if the report is too long to fit reviewer context comfortably,
    # recursively summarise before sending. Keeps nuance intact on long runs.
    if len(report_text) > 12_000:
        from app.tools.summariser import recursive_summarise

        report_text = await recursive_summarise(report_text, target_chars=8_000)

    verdict = await _reviewer_invoke(topic, report_text)
    revision_count = state.get("revision_count", 0)

    # H-4: exit the revision loop if we've spent too many tokens already.
    from app.graph.callbacks import get_session_tokens

    thread_id = state.get("thread_id", "")
    if thread_id and get_session_tokens(thread_id) >= settings.max_tokens_per_session:
        log.warning(
            "reviewer_force_exit thread_id=%s tokens=%d cap=%d",
            thread_id,
            get_session_tokens(thread_id),
            settings.max_tokens_per_session,
        )
        return {"review_feedback": "", "revision_count": revision_count}

    # G-7: if a per-session USD budget was set and we've spent it, exit too.
    budget = float(state.get("budget_usd") or 0.0)
    if budget > 0 and thread_id:
        from app.pricing import chat_cost

        # Rough cost estimate: treat session tokens as 75% prompt / 25% completion.
        toks = get_session_tokens(thread_id)
        cost_so_far = chat_cost(settings.openai_model, int(toks * 0.75), int(toks * 0.25))
        if cost_so_far >= budget:
            log.warning(
                "reviewer_force_exit_budget thread_id=%s cost=$%.4f budget=$%.4f",
                thread_id,
                cost_so_far,
                budget,
            )
            return {"review_feedback": "", "revision_count": revision_count}

    if not verdict.approved and revision_count < settings.max_revision_count:
        return {
            "review_feedback": verdict.feedback.strip() or "Improve quality and coverage.",
            "revision_count": revision_count + 1,
        }
    return {"review_feedback": "", "revision_count": revision_count}


# ---------------------------------------------------------------------------
# Node: formatter
# ---------------------------------------------------------------------------

_formatter_chain = (FORMATTER_PROMPT | _formatter_llm | StrOutputParser()).with_config(
    tags=["formatter"]
)


@_retry_llm
async def _formatter_invoke(topic: str, sections_text: str) -> str:
    return await _formatter_chain.ainvoke({"topic": topic, "sections": sections_text})


@_instrument("formatter")
async def formatter_node(state: ResearchState) -> dict:
    topic = state["topic"]
    sections = state.get("sections", {})
    sections_text = "\n\n".join(f"## {t}\n{b}" for t, b in sections.items())
    final_report = await _formatter_invoke(topic, sections_text)
    # L-2: bound the report length so an LLM that runs away can't OOM downstream.
    cap = settings.max_report_length
    if len(final_report) > cap:
        final_report = final_report[:cap] + "\n\n_[report truncated to fit budget]_"
    return {"final_report": final_report}


# ---------------------------------------------------------------------------
# Node: citations — extract claims from final_report, match to source sections
# ---------------------------------------------------------------------------


_kg_chain = KG_PROMPT | _reviewer_llm.with_structured_output(KnowledgeGraph)


@_retry_llm
async def _kg_invoke(report: str) -> KnowledgeGraph:
    return await _kg_chain.ainvoke({"report": report})


@_instrument("knowledge_graph")
async def kg_node(state: ResearchState) -> dict:
    """Extract entities + relations from the final report (G-13), and run
    the output-safety classifier (G-15) in parallel.

    Output is structured via Pydantic schemas so downstream consumers
    (API, follow-up queries) can reason over it without re-parsing prose.
    """
    report = state.get("final_report", "")
    if not report or len(report) < 200:
        return {
            "knowledge_graph": {"entities": [], "relations": []},
            "safety_flags": [],
        }

    from app.tools.safety import audit_report

    kg_task = asyncio.create_task(_kg_invoke(report[:8000]))
    safety_task = asyncio.create_task(audit_report(report))
    kg_result, safety_result = await asyncio.gather(kg_task, safety_task, return_exceptions=True)

    if isinstance(kg_result, BaseException):
        log.exception("kg_extraction_failed", exc_info=kg_result)
        kg_payload = {"entities": [], "relations": []}
    else:
        kg_payload = {
            "entities": [e.model_dump() for e in kg_result.entities],
            "relations": [r.model_dump(by_alias=True) for r in kg_result.relations],
        }

    if isinstance(safety_result, BaseException):
        log.exception("safety_audit_failed", exc_info=safety_result)
        flags: list[dict] = []
    else:
        flags = safety_result

    return {"knowledge_graph": kg_payload, "safety_flags": flags}


@_instrument("citations")
async def citations_node(state: ResearchState) -> dict:
    """Extract factual claims from the report and attribute each to its source.

    Appends a '## Sources' table + an 'Unsourced claims' section to the report,
    and stores the structured citations in state for downstream consumers
    (eval, UI highlighting).
    """
    report = state.get("final_report", "")
    sources = state.get("search_results", {})
    if not report:
        return {"citations": []}

    claims = await extract_claims(report)
    citations = await attribute(claims, sources)
    enriched_report = inject_footnotes(report, citations)
    return {"final_report": enriched_report, "citations": citations}


# ---------------------------------------------------------------------------
# Node: refine — multi-turn refinement based on user follow-up
#
# Reads the conversation history from state.messages (built up across
# follow-up turns), the prior final_report, and user's latest instruction
# to produce a revised report. This is what turns single-request → assistant.
# ---------------------------------------------------------------------------

_refine_chain = (REFINE_PROMPT | _writer_llm | StrOutputParser()).with_config(
    tags=["refine", "writer"]
)


@_retry_llm
async def _refine_invoke(report: str, conversation: str, request: str) -> str:
    return await _refine_chain.ainvoke(
        {"report": report, "conversation": conversation, "request": request}
    )


# H-2: sanitise any line of a message that *looks* like a role header (system:,
# assistant:, ###, ---, <|...|>, etc.). An attacker-crafted earlier AIMessage
# (e.g. via a manipulated upstream turn) could otherwise inject a fake system
# directive into <conversation>. We prefix every offending line with "  " so
# the model still sees the text but not the role cue.

_ROLE_HEADER_RE = _re.compile(
    r"""^(?:
        \s*(?:system|assistant|user|human|ai|tool)\s*:  # role: prefix
      | \s*\#{2,}                                       # markdown ### heading
      | \s*-{3,}\s*$                                    # --- separator
      | \s*<\|.*?\|>                                    # ChatML-style markers
    )""",
    _re.VERBOSE | _re.IGNORECASE,
)


_DEFANG_PREFIX = "(quoted) "


def _sanitise_message_content(text: str) -> str:
    cleaned: list[str] = []
    for line in (text or "").splitlines():
        if _ROLE_HEADER_RE.match(line):
            # Prefix with a non-whitespace token so the regex no longer matches
            # and the LLM reads the content as quoted data, not a role header.
            cleaned.append(_DEFANG_PREFIX + line)
        else:
            cleaned.append(line)
    return "\n".join(cleaned)


def _format_conversation(messages: list) -> str:
    """Flatten prior messages into a `role: content` transcript for the prompt.

    Content is sanitised against role-header injection (H-2) so a prior
    assistant reply can't smuggle a fake `system:` directive past the model.
    """
    lines: list[str] = []
    for m in messages or []:
        role = getattr(m, "type", None) or getattr(m, "role", "user")
        content = _sanitise_message_content(getattr(m, "content", None) or str(m))
        lines.append(f"{role}: {content}")
    return "\n".join(lines) or "(no prior turns)"


@_instrument("refine")
async def refine_node(state: ResearchState) -> dict:
    """Produce a revised report that incorporates the user's follow-up request.

    Expects `state.messages[-1]` to be the latest human request. Emits a new
    `final_report` AND appends an AI message to `messages` so subsequent
    follow-ups see the full history.
    """
    from langchain_core.messages import AIMessage

    prior_report = state.get("final_report", "")
    messages = state.get("messages", []) or []
    if not messages:
        # No follow-up yet — nothing to refine
        return {}

    last = messages[-1]
    latest_request = getattr(last, "content", None) or str(last)

    revised = await _refine_invoke(
        report=prior_report or "(no prior report)",
        conversation=_format_conversation(messages[:-1]),
        request=latest_request,
    )

    return {
        "final_report": revised,
        "messages": [AIMessage(content=revised)],
    }
