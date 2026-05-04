"""Tests for scripts/eval_drift and scripts/eval_pairwise helpers.

We don't drive the full eval pipelines (they need real OpenAI); we cover
the pure helpers and the order-swap logic in pairwise evaluation.
"""

from __future__ import annotations

import pytest

from scripts.eval_drift import _nested_get
from scripts.eval_pairwise import (
    PairwiseResult,
    PairwiseVerdict,
    _pairwise_run,
    _render_markdown,
)

# ---------------------------------------------------------------------------
# eval_drift._nested_get
# ---------------------------------------------------------------------------


def test_nested_get_returns_leaf_value():
    payload = {"ragas": {"faithfulness": 0.82}}
    assert _nested_get(payload, "ragas.faithfulness") == 0.82


def test_nested_get_missing_path_is_none():
    payload = {"ragas": {"faithfulness": 0.8}}
    assert _nested_get(payload, "ragas.not_there") is None
    assert _nested_get(payload, "missing.top") is None


def test_nested_get_rejects_non_numeric_leaves():
    # Non-numeric leaves are returned as None so aggregation doesn't blow up.
    payload = {"judge": {"notes": "narrative text"}}
    assert _nested_get(payload, "judge.notes") is None


# ---------------------------------------------------------------------------
# eval_pairwise — order-swap logic + markdown render
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pairwise_run_swaps_order_on_alternating_topics(monkeypatch):
    """Even-indexed topics are judged as (v1, v2); odd-indexed as (v2, v1).
    The function must flip the winner back so results are reported in the
    caller's (v1, v2) frame."""
    calls: list[tuple[str, str]] = []

    async def fake_judge(topic, a, b):
        calls.append((a, b))
        # The "left-hand" argument always wins in the fake judge — that
        # lets us verify the swap-then-flip logic produces the right winner.
        return PairwiseVerdict(winner="v1", confidence=0.9, rationale="a wins")

    monkeypatch.setattr("scripts.eval_pairwise._judge_pair", fake_judge)

    v1 = {"topic-a": "REPORT-A1", "topic-b": "REPORT-B1", "topic-c": "REPORT-C1"}
    v2 = {"topic-a": "REPORT-A2", "topic-b": "REPORT-B2", "topic-c": "REPORT-C2"}

    results = await _pairwise_run(v1, v2)
    assert len(results) == 3

    # Topics are sorted in _pairwise_run, so order is a, b, c. i=0 no-swap,
    # i=1 swap, i=2 no-swap. Since the "left" always wins in the fake:
    #   i=0 (no swap): judge sees (v1, v2) → returns v1 → winner v1.
    #   i=1 (swap):    judge sees (v2, v1) → returns v1 (left) → flipped to v2.
    #   i=2 (no swap): v1 again.
    by_topic = {r.topic: r.verdict.winner for r in results}
    assert by_topic["topic-a"] == "v1"
    assert by_topic["topic-b"] == "v2"
    assert by_topic["topic-c"] == "v1"


@pytest.mark.asyncio
async def test_pairwise_run_skips_judge_failures(monkeypatch):
    call_count = {"n": 0}

    async def flaky_judge(topic, a, b):  # noqa: ARG001
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("judge blew up")
        return PairwiseVerdict(winner="tie", confidence=0.1, rationale="eh")

    monkeypatch.setattr("scripts.eval_pairwise._judge_pair", flaky_judge)
    results = await _pairwise_run(
        {"t1": "a", "t2": "b", "t3": "c"},
        {"t1": "x", "t2": "y", "t3": "z"},
    )
    # 1 failing judgement → dropped; 2 survive.
    assert len(results) == 2


def test_render_markdown_counts_and_tallies():
    results = [
        PairwiseResult(
            topic="a",
            verdict=PairwiseVerdict(winner="v1", confidence=0.9, rationale="ok"),
        ),
        PairwiseResult(
            topic="b",
            verdict=PairwiseVerdict(winner="v2", confidence=0.7, rationale="ok"),
        ),
        PairwiseResult(
            topic="c",
            verdict=PairwiseVerdict(winner="tie", confidence=0.3, rationale="ok"),
        ),
    ]
    md = _render_markdown(results)
    assert "| v1 | 1 | 33.3% |" in md
    assert "| v2 | 1 | 33.3% |" in md
    assert "| tie | 1 | 33.3% |" in md
    # Per-topic rows rendered.
    assert "| a | v1 |" in md


def test_render_markdown_handles_empty_results():
    """The div-by-zero guard leaves `total=1` in the header, which is fine
    for an empty run; we just need no exception and zero counts in every row.
    """
    md = _render_markdown([])
    # Division guard (`total = len(results) or 1`) prevents ZeroDivisionError.
    assert "| v1 | 0 | 0.0% |" in md
    assert "| v2 | 0 | 0.0% |" in md
    assert "| tie | 0 | 0.0% |" in md
