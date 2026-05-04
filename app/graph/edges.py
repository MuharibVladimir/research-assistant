"""Conditional edge functions for the Research Assistant graph.

These are the routing functions passed to add_conditional_edges().
Each receives the current state and returns a string node name.
"""

from app.config import settings
from app.graph.state import ResearchState


def route_after_reviewer(state: ResearchState) -> str:
    """Decide what to do after the reviewer node.

    - If review_feedback is set AND revision_count < max: go back to researcher
    - Otherwise: proceed to formatter
    """
    feedback = state.get("review_feedback", "")
    revision_count = state.get("revision_count", 0)

    if feedback and revision_count <= settings.max_revision_count:
        return "researcher"

    return "formatter"


def route_after_grader(state: ResearchState) -> str:
    """CRAG routing: after grader decides relevance of retrieved content.

    Three outcomes:
    * All relevant → `writer` (cache hit, skip expensive web fetch).
    * Any irrelevant AND we haven't exhausted adaptive depth → `adaptive_retrieval`
      (try a deeper pgvector pass before paying for Tavily).
    * Any irrelevant AND we've tried adaptive already → `web_search` (give up on
      cache, go to the web).
    """
    grades = _strip_meta(state.get("retrieval_grades", {}))

    if not grades:
        return "web_search"

    if all(g == "relevant" for g in grades.values()):
        return "writer"

    # G-8: prefer a deeper cache pass before the web if we have budget left.
    depth = int(state.get("retrieval_depth_count") or 0)
    if depth < settings.max_retrieval_depth:
        return "adaptive_retrieval"

    return "web_search"


def route_after_adaptive_retrieval(state: ResearchState) -> str:
    """After an adaptive pass, either accept the new cache state or fall through."""
    grades = _strip_meta(state.get("retrieval_grades", {}))
    if grades and all(g == "relevant" for g in grades.values()):
        return "writer"
    return "web_search"


def _strip_meta(grades: dict) -> dict:
    """Filter out the `_caller_hash` meta-field the planner uses (G-12).

    `retrieval_grades` is a dict-reducer field so we piggyback the caller's
    hash in it instead of adding a new state field. Edges must ignore it.
    """
    return {k: v for k, v in (grades or {}).items() if not k.startswith("_")}
