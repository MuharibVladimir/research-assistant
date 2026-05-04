"""Baseline save / compare logic — exercised without running the real graph."""

from __future__ import annotations

from app.eval.metrics import RagasScore
from scripts.eval import ReportScore, TopicResult, _render_diff, _to_json_payload


def _mk_result(topic: str, rel: int = 4, faith: float = 0.8) -> TopicResult:
    return TopicResult(
        topic=topic,
        score=ReportScore(relevance=rel, depth=rel, structure=rel, factuality=rel),
        ragas=RagasScore(
            faithfulness=faith, citation_precision=0.7, citation_recall=0.8, answer_relevance=0.75
        ),
        tokens=1000,
        cost_usd=0.001,
        latency_s=10.0,
    )


def test_to_json_payload_shape():
    r = _mk_result("t1")
    payload = _to_json_payload([r], RagasScore(0.8, 0.7, 0.8, 0.75))
    assert payload["topics"][0]["topic"] == "t1"
    assert payload["topics"][0]["judge"]["relevance"] == 4
    assert payload["aggregate"]["ragas"]["faithfulness"] == 0.8


def test_to_json_payload_empty_results():
    payload = _to_json_payload([], RagasScore(0.0, 0.0, 0.0, 0.0))
    assert payload["topics"] == []
    assert payload["aggregate"]["judge"]["relevance"] == 0


def test_render_diff_highlights_regression():
    baseline = {
        "aggregate": {
            "judge": {"relevance": 4.0, "depth": 4.0, "structure": 4.0, "factuality": 4.0},
            "ragas": {
                "faithfulness": 0.9,
                "citation_precision": 0.8,
                "citation_recall": 0.85,
                "answer_relevance": 0.8,
            },
        }
    }
    current = {
        "aggregate": {
            "judge": {"relevance": 3.5, "depth": 4.0, "structure": 4.0, "factuality": 4.0},
            "ragas": {
                "faithfulness": 0.7,
                "citation_precision": 0.8,
                "citation_recall": 0.85,
                "answer_relevance": 0.8,
            },
        }
    }
    diff = _render_diff(baseline, current)
    # Relevance dropped 4 → 3.5
    assert "-0.500" in diff
    # Faithfulness dropped 0.9 → 0.7
    assert "-0.200" in diff
    # Table header rendered
    assert "| Metric | Baseline | Current |" in diff


def test_render_diff_highlights_improvement():
    baseline = {
        "aggregate": {
            "judge": {"relevance": 3.0, "depth": 3.0, "structure": 3.0, "factuality": 3.0},
            "ragas": {
                "faithfulness": 0.5,
                "citation_precision": 0.5,
                "citation_recall": 0.5,
                "answer_relevance": 0.5,
            },
        }
    }
    current = {
        "aggregate": {
            "judge": {"relevance": 4.5, "depth": 4.5, "structure": 4.5, "factuality": 4.5},
            "ragas": {
                "faithfulness": 0.9,
                "citation_precision": 0.9,
                "citation_recall": 0.9,
                "answer_relevance": 0.9,
            },
        }
    }
    diff = _render_diff(baseline, current)
    # All positive deltas
    assert "+1.500" in diff  # judge metrics
    assert "+0.400" in diff  # ragas metrics
