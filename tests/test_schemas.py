"""Tests for the pydantic schemas used by `with_structured_output`.

These schemas define the contract between the LLM and the graph. A typo
or a relaxed constraint here would let malformed LLM output through.
"""

import pytest
from pydantic import ValidationError

from app.graph.nodes import GradeVerdict, ResearchPlan, ReviewVerdict


def test_research_plan_accepts_sections() -> None:
    plan = ResearchPlan(sections=["intro", "benchmarks", "conclusion"])
    assert plan.sections == ["intro", "benchmarks", "conclusion"]


def test_research_plan_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        ResearchPlan(sections=[])


@pytest.mark.parametrize("grade", ["relevant", "irrelevant"])
def test_grade_verdict_accepts_valid(grade: str) -> None:
    assert GradeVerdict(grade=grade).grade == grade


@pytest.mark.parametrize("grade", ["Relevant", "maybe", "yes", ""])
def test_grade_verdict_rejects_invalid(grade: str) -> None:
    with pytest.raises(ValidationError):
        GradeVerdict(grade=grade)


def test_review_verdict_approved_default_feedback() -> None:
    v = ReviewVerdict(approved=True)
    assert v.approved is True
    assert v.feedback == ""


def test_review_verdict_rejected_keeps_feedback() -> None:
    v = ReviewVerdict(approved=False, feedback="missing sources")
    assert v.approved is False
    assert v.feedback == "missing sources"
