"""Tests for graph routing functions.

These edges are pure functions with no I/O, so they're trivial to test —
but they encode the control flow of the whole graph, so regressions here
would silently break behavior (e.g. skipping web_search or looping forever).
"""

import pytest

from app.config import settings
from app.graph.edges import route_after_grader, route_after_reviewer

# ---------------------------------------------------------------------------
# route_after_grader
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("grades", "depth", "expected"),
    [
        # all relevant → straight to writer (depth doesn't matter)
        ({"s1": "relevant", "s2": "relevant"}, 0, "writer"),
        # any irrelevant AND adaptive-depth left → try a deeper cache pass first
        ({"s1": "relevant", "s2": "irrelevant"}, 0, "adaptive_retrieval"),
        ({"s1": "irrelevant"}, 0, "adaptive_retrieval"),
        # any irrelevant AND adaptive exhausted → give up on cache, go to web
        ({"s1": "irrelevant"}, settings.max_retrieval_depth, "web_search"),
        # empty grades (defensive) → web search
        ({}, 0, "web_search"),
    ],
)
def test_route_after_grader(grades: dict[str, str], depth: int, expected: str) -> None:
    state = {"retrieval_grades": grades, "retrieval_depth_count": depth}
    assert route_after_grader(state) == expected


def test_route_after_grader_missing_key_defaults_to_web_search() -> None:
    """No retrieval_grades in state at all (shouldn't happen but must not crash)."""
    assert route_after_grader({}) == "web_search"


# ---------------------------------------------------------------------------
# route_after_reviewer
# ---------------------------------------------------------------------------


def test_route_after_reviewer_approved_goes_to_formatter() -> None:
    state = {"review_feedback": "", "revision_count": 0}
    assert route_after_reviewer(state) == "formatter"


def test_route_after_reviewer_with_feedback_loops_back() -> None:
    state = {"review_feedback": "needs more detail", "revision_count": 0}
    assert route_after_reviewer(state) == "researcher"


def test_route_after_reviewer_max_revisions_forces_formatter() -> None:
    """Once revision_count exceeds max, we must exit the loop even with feedback."""
    state = {
        "review_feedback": "still needs work",
        "revision_count": settings.max_revision_count + 1,
    }
    assert route_after_reviewer(state) == "formatter"


def test_route_after_reviewer_empty_feedback_skips_loop() -> None:
    """Empty-string feedback (approved) must not loop, regardless of revision_count."""
    state = {"review_feedback": "", "revision_count": 1}
    assert route_after_reviewer(state) == "formatter"


def test_route_after_reviewer_missing_fields_defaults_to_formatter() -> None:
    """State without feedback/revision keys should still route deterministically."""
    assert route_after_reviewer({}) == "formatter"
