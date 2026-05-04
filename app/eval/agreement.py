"""Inter-annotator agreement + eval-vs-golden correlation (G-1).

Two things live here:

  * **IAA (Fleiss' κ)** — when multiple humans score the same report, how
    much do they agree? Low κ means the rubric itself is fuzzy and scores
    should be taken with salt. Computed per metric (faithfulness, relevance,
    depth, factuality).

  * **eval-vs-golden correlation** — Pearson correlation between our
    automated eval-harness scores and the human mean per topic. Tells us
    whether the judge LLM is a faithful proxy for the humans. Target ≥ 0.85.

No `nltk` dependency — we implement both metrics in ~30 LOC. They're
textbook statistics; pulling `nltk` just for `agreement.AnnotationTask`
would add ~40MB for five lines of use.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass


@dataclass
class AgreementReport:
    """Per-metric IAA + correlation against automated eval output."""

    metric: str
    n_items: int
    n_annotators: int
    fleiss_kappa: float
    human_mean: float
    correlation_with_eval: float | None  # None if no matching eval score given

    def as_dict(self) -> dict:
        return {
            "metric": self.metric,
            "n_items": self.n_items,
            "n_annotators": self.n_annotators,
            "fleiss_kappa": round(self.fleiss_kappa, 4),
            "human_mean": round(self.human_mean, 4),
            "correlation_with_eval": (
                round(self.correlation_with_eval, 4)
                if self.correlation_with_eval is not None
                else None
            ),
        }


def fleiss_kappa(ratings: list[list[int]]) -> float:
    """Fleiss' κ for categorical ratings.

    `ratings[i]` is a list of integer category labels from N annotators on
    item i — e.g. for a 1-5 scale, each entry is ∈ {1,2,3,4,5}.

    Implementation follows Wikipedia's formula:
      Pi = (sum_j(n_ij^2) - N) / (N * (N - 1))
      Pe = sum_j(p_j^2)
      κ  = (P_bar - Pe) / (1 - Pe)

    Returns κ ∈ [-1, 1]. Landis-Koch rough bands:
      < 0.2 poor, 0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial,
      > 0.8 almost perfect.
    """
    if not ratings:
        return 0.0
    N = len(ratings[0])
    if N < 2:
        return 0.0
    # Collect category universe
    categories = sorted({r for item in ratings for r in item})
    if len(categories) < 2:
        # Perfect agreement when everyone picks the same label — but Fleiss
        # is undefined mathematically (division by zero in 1 - Pe). Define
        # it as 1.0 by convention.
        return 1.0

    n_items = len(ratings)
    # Per-item agreement P_i
    P_is: list[float] = []
    cat_totals: Counter[int] = Counter()
    for item in ratings:
        counts = Counter(item)
        for c in categories:
            cat_totals[c] += counts[c]
        P_i = (sum(counts[c] ** 2 for c in categories) - N) / (N * (N - 1))
        P_is.append(P_i)

    P_bar = statistics.mean(P_is)
    total_ratings = N * n_items
    p_j = [cat_totals[c] / total_ratings for c in categories]
    P_e = sum(p * p for p in p_j)

    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)


def pearson_correlation(xs: list[float], ys: list[float]) -> float:
    """Pearson r. Returns 0.0 if the variance is zero on either side."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    denom = (dx * dy) ** 0.5
    return num / denom if denom else 0.0


# ---------------------------------------------------------------------------
# High-level API consumed by scripts/eval.py --against-golden
# ---------------------------------------------------------------------------


_METRICS = ("faithfulness", "relevance", "depth", "factuality")


def load_annotations(annotations_json: str) -> list[dict]:
    """Parse the JSON stored in `golden_reports.annotations_json`."""
    try:
        data = json.loads(annotations_json)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def compute_agreement(
    golden_rows: list[tuple[str, str]],
    eval_scores_by_topic: dict[str, dict[str, float]] | None = None,
) -> dict[str, AgreementReport]:
    """Compute per-metric IAA (and optional eval-correlation) across topics.

    Args:
        golden_rows: list of `(topic, annotations_json)` pairs.
        eval_scores_by_topic: optional `{topic: {metric: score}}` from the
            automated eval harness. When provided, computes Pearson r
            between the per-topic human mean and the eval score.

    Returns: `{metric: AgreementReport}`.
    """
    # ratings_by_metric[metric] = list[list[int]]  — per topic, per annotator
    ratings_by_metric: dict[str, list[list[int]]] = {m: [] for m in _METRICS}
    human_means_by_metric: dict[str, list[float]] = {m: [] for m in _METRICS}
    eval_vals_by_metric: dict[str, list[float]] = {m: [] for m in _METRICS}
    topics_order: list[str] = []

    for topic, annot_json in golden_rows:
        topics_order.append(topic)
        annots = load_annotations(annot_json)
        if not annots:
            continue
        for metric in _METRICS:
            values = [int(a[metric]) for a in annots if metric in a]
            if not values:
                continue
            ratings_by_metric[metric].append(values)
            human_means_by_metric[metric].append(statistics.mean(values))
            if eval_scores_by_topic is not None:
                eval_score = eval_scores_by_topic.get(topic, {}).get(metric)
                if eval_score is not None:
                    eval_vals_by_metric[metric].append(float(eval_score))

    reports: dict[str, AgreementReport] = {}
    for metric in _METRICS:
        items = ratings_by_metric[metric]
        if not items:
            continue
        kappa = fleiss_kappa(items)
        hm = statistics.mean(human_means_by_metric[metric])
        corr = (
            pearson_correlation(
                human_means_by_metric[metric][: len(eval_vals_by_metric[metric])],
                eval_vals_by_metric[metric],
            )
            if len(eval_vals_by_metric[metric]) >= 2
            else None
        )
        reports[metric] = AgreementReport(
            metric=metric,
            n_items=len(items),
            n_annotators=len(items[0]) if items else 0,
            fleiss_kappa=kappa,
            human_mean=hm,
            correlation_with_eval=corr,
        )
    return reports
