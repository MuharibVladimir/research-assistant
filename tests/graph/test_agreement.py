"""Unit tests for inter-annotator agreement and golden correlation (G-1)."""

from __future__ import annotations

import json

from app.eval.agreement import (
    compute_agreement,
    fleiss_kappa,
    pearson_correlation,
)


def test_fleiss_kappa_perfect_agreement():
    # All annotators give 5 on every item — mathematically κ is undefined;
    # we define it as 1.0.
    assert fleiss_kappa([[5, 5, 5], [5, 5, 5], [5, 5, 5]]) == 1.0


def test_fleiss_kappa_random_disagreement():
    # 3 annotators × 6 items, labels spread across 1..5 — κ should be ≤ 0.2.
    data = [[1, 3, 5], [2, 4, 1], [5, 1, 3], [4, 2, 5], [1, 5, 2], [3, 1, 4]]
    assert fleiss_kappa(data) < 0.2


def test_fleiss_kappa_strong_agreement():
    # Most items see 2 of 3 annotators giving the same rating.
    data = [[4, 4, 3], [5, 5, 4], [3, 3, 4], [5, 5, 5], [4, 4, 4], [3, 4, 3]]
    assert fleiss_kappa(data) > 0.2


def test_pearson_linear():
    assert abs(pearson_correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) - 1.0) < 1e-9


def test_pearson_anticorrelated():
    r = pearson_correlation([1, 2, 3, 4, 5], [10, 8, 6, 4, 2])
    assert abs(r - (-1.0)) < 1e-9


def test_pearson_zero_variance():
    # One constant series → correlation falls back to 0.0
    assert pearson_correlation([3, 3, 3, 3], [1, 2, 3, 4]) == 0.0


def test_compute_agreement_end_to_end():
    # Mix of agreement patterns so Fleiss κ varies per metric.
    rows = []
    eval_scores = {}
    for i in range(5):
        topic = f"t{i}"
        # Annotators agree perfectly on faithfulness, spread a bit elsewhere.
        annots = [
            {
                "annotator_id": "a",
                "faithfulness": 4,
                "relevance": 4 if i % 2 == 0 else 5,
                "depth": 3 + (i % 3),
                "factuality": 4,
            },
            {
                "annotator_id": "b",
                "faithfulness": 4,
                "relevance": 4,
                "depth": 3 + (i % 3),
                "factuality": 4 if i < 3 else 5,
            },
        ]
        rows.append((topic, json.dumps(annots)))
        eval_scores[topic] = {
            "faithfulness": 4.0,
            "relevance": 4.2 + 0.1 * i,
            "depth": 3.0 + (i % 3),
            "factuality": 4.0,
        }

    reports = compute_agreement(rows, eval_scores)
    assert set(reports.keys()) == {"faithfulness", "relevance", "depth", "factuality"}

    # Faithfulness: all annotators pick 4 → κ=1.0 (perfect).
    assert reports["faithfulness"].fleiss_kappa == 1.0
    # Depth: both annotators follow the same (i % 3) pattern → perfect too.
    assert reports["depth"].fleiss_kappa == 1.0

    for r in reports.values():
        assert r.n_items == 5
        assert r.n_annotators == 2


def test_compute_agreement_no_eval_scores():
    """Correlation is None when no eval scores are provided."""
    rows = [
        (
            "t1",
            json.dumps(
                [
                    {"annotator_id": "a", "faithfulness": 4},
                    {"annotator_id": "b", "faithfulness": 4},
                ]
            ),
        ),
        (
            "t2",
            json.dumps(
                [
                    {"annotator_id": "a", "faithfulness": 3},
                    {"annotator_id": "b", "faithfulness": 3},
                ]
            ),
        ),
    ]
    reports = compute_agreement(rows)
    assert reports["faithfulness"].correlation_with_eval is None
