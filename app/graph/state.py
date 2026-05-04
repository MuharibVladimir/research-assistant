from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


def _replace(old: str | None, new: str | None) -> str | None:
    """Reducer that replaces the value (default langgraph behaviour for non-list fields)."""
    return new if new is not None else old


class ResearchState(TypedDict):
    # Input
    topic: str
    thread_id: str

    # Planner output — list of section titles
    plan: list[str]

    # Human-in-the-loop: set to True after user approves the plan
    human_approved: bool

    # Researcher output — raw search results per section
    # key = section title, value = raw text snippets joined
    search_results: Annotated[dict[str, str], lambda a, b: {**a, **b}]

    # Writer output — written section texts
    sections: Annotated[dict[str, str], lambda a, b: {**a, **b}]

    # Reviewer feedback
    review_feedback: str

    # How many revision loops have happened
    revision_count: int

    # Final assembled report
    final_report: str

    # CRAG: grader verdict per section — "relevant" | "irrelevant"
    retrieval_grades: Annotated[dict[str, str], lambda a, b: {**a, **b}]

    # Claim → source attribution. Each entry:
    # { "claim": str, "source_section": str, "score": float }
    # Populated by the citations node after formatter.
    citations: list[dict]

    # G-13 knowledge graph extracted from the final report. Structure:
    # {
    #   "entities":  [{"name": str, "type": "tool|company|concept|person|..."}],
    #   "relations": [{"from": str, "relation": str, "to": str}],
    # }
    knowledge_graph: dict

    # G-15 safety-classifier flags on the final report (speculative claims,
    # unverified facts about named people, etc).
    safety_flags: list[dict]

    # G-7 budget ceiling in USD — planner sees this and plans for it; reviewer
    # loop short-circuits when cost breaches it.
    budget_usd: float

    # G-8 adaptive retrieval: how many extra retrieval passes we've done on
    # sections the grader rated `irrelevant`. Capped by settings.max_retrieval_depth.
    retrieval_depth_count: int

    # Conversation-style messages (used for human-in-the-loop interrupt)
    messages: Annotated[list, add_messages]
