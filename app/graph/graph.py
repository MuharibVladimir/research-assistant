"""Assemble and compile the Research Assistant LangGraph StateGraph.

Graph topology (CRAG):
    START
      └─> planner
            └─> [interrupt_before await_approval — human approves plan]
                  └─> researcher  (checks pgvector cache first)
                        └─> grader  (LLM validates cache-hit quality)
                              ├─(all relevant — cache hit)────> writer
                              └─(any irrelevant — cache miss)─> web_search ─> writer
                                                                      └─> reviewer
                                                                            ├─(needs revision)─> researcher
                                                                            └─(approved)───────> formatter
                                                                                                    └─> END

Checkpointing:
    AsyncPostgresSaver with a connection pool (min=2, max=20). Pool is owned
    by `create_postgres_checkpointer` and must be closed via `close_pool()`
    on app shutdown (wired into FastAPI lifespan in main.py).
"""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg_pool import AsyncConnectionPool

from app.config import settings
from app.graph.edges import (
    route_after_adaptive_retrieval,
    route_after_grader,
    route_after_reviewer,
)
from app.graph.nodes import (
    adaptive_retrieval_node,
    citations_node,
    formatter_node,
    grader_node,
    kg_node,
    planner_node,
    refine_node,
    researcher_node,
    reviewer_node,
    web_search_node,
    writer_node,
)
from app.graph.state import ResearchState

# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------


async def _await_approval_node(state: ResearchState) -> dict:
    """No-op node that exists solely as the interrupt point.

    Interrupting before THIS node (not before researcher) means the graph
    pauses exactly once — after planner, before the first research pass.
    Revision cycles loop directly back to researcher and skip this node.
    """
    return {}


_builder = StateGraph(ResearchState)

_builder.add_node("planner", planner_node)
_builder.add_node("await_approval", _await_approval_node)
_builder.add_node("researcher", researcher_node)
_builder.add_node("grader", grader_node)
_builder.add_node("adaptive_retrieval", adaptive_retrieval_node)
_builder.add_node("web_search", web_search_node)
_builder.add_node("writer", writer_node)
_builder.add_node("reviewer", reviewer_node)
_builder.add_node("formatter", formatter_node)
_builder.add_node("citations", citations_node)
_builder.add_node("knowledge_graph", kg_node)

_builder.add_edge(START, "planner")
_builder.add_edge("planner", "await_approval")
_builder.add_edge("await_approval", "researcher")
_builder.add_edge("researcher", "grader")
_builder.add_conditional_edges(
    "grader",
    route_after_grader,
    {
        "writer": "writer",
        "adaptive_retrieval": "adaptive_retrieval",
        "web_search": "web_search",
    },
)
_builder.add_conditional_edges(
    "adaptive_retrieval",
    route_after_adaptive_retrieval,
    {"writer": "writer", "web_search": "web_search"},
)
_builder.add_edge("web_search", "writer")
_builder.add_edge("writer", "reviewer")
_builder.add_conditional_edges(
    "reviewer",
    route_after_reviewer,
    {"researcher": "researcher", "formatter": "formatter"},
)
_builder.add_edge("formatter", "citations")
_builder.add_edge("citations", "knowledge_graph")
_builder.add_edge("knowledge_graph", END)


def build_graph(checkpointer=None):
    """Compile the graph, optionally attaching a checkpointer.

    interrupt_before=["await_approval"] fires only once — after planner.
    Revision cycles bypass await_approval entirely.
    """
    return _builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["await_approval"],
    )


# ---------------------------------------------------------------------------
# Follow-up graph — a tiny sub-graph that drives a single refine_node pass.
# Shares the checkpointer (and thus the thread_id + persisted messages) with
# the main research graph, so the refined report is a continuation of the
# original session, not a new conversation.
# ---------------------------------------------------------------------------

_refine_builder = StateGraph(ResearchState)
_refine_builder.add_node("refine", refine_node)
_refine_builder.add_edge(START, "refine")
_refine_builder.add_edge("refine", END)


def build_refine_graph(checkpointer=None):
    """Compile the refine-only follow-up graph."""
    return _refine_builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# AsyncPostgresSaver with connection pool
# ---------------------------------------------------------------------------

_pool: AsyncConnectionPool | None = None


async def create_postgres_checkpointer() -> AsyncPostgresSaver:
    """Create an AsyncPostgresSaver backed by a shared connection pool.

    The pool has min_size=2 to avoid cold-start latency, max_size=20 to cap
    concurrent DB work. Must be closed on shutdown via `close_pool()`.
    """
    global _pool
    conn_string = settings.database_url.replace("postgresql+psycopg://", "postgresql://")
    _pool = AsyncConnectionPool(
        conninfo=conn_string,
        min_size=2,
        max_size=20,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=False,
    )
    await _pool.open()
    checkpointer = AsyncPostgresSaver(_pool)
    await checkpointer.setup()
    return checkpointer


async def close_pool() -> None:
    """Close the shared connection pool (call from app shutdown)."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
