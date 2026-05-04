"""RAGAS-style quality metric tests.

Claim extraction and embeddings are stubbed so we can assert arithmetic
without hitting OpenAI.
"""

from __future__ import annotations

import pytest

from app.eval import metrics as metrics_mod
from app.eval.metrics import RagasScore, aggregate, compute_scores
from app.tools import attribution as attribution_mod
from app.tools.attribution import ExtractedClaim


@pytest.mark.asyncio
async def test_compute_scores_high_faithfulness(monkeypatch):
    """All extracted claims map to a source → faithfulness ≈ 1.0."""

    async def fake_extract(report):  # noqa: ARG001
        return [
            ExtractedClaim(claim="claim about climate policy", claim_type="factual"),
            ExtractedClaim(claim="claim about tech trends", claim_type="statistic"),
        ]

    def fake_embed(text):
        if "climate" in text:
            return [1.0, 0.0, 0.0]
        if "tech" in text:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]  # topic / report embedding

    monkeypatch.setattr(metrics_mod, "extract_claims", fake_extract)
    monkeypatch.setattr(metrics_mod, "embed_text", fake_embed)
    monkeypatch.setattr(attribution_mod, "embed_text", fake_embed)

    sources = {"climate": "climate body", "tech": "tech body"}
    score = await compute_scores(
        "topic", "# report\nclimate claim. tech claim.", sources, threshold=0.5
    )
    assert score.faithfulness == 1.0
    assert score.citation_recall == 1.0
    assert score.citation_precision >= 0.5


@pytest.mark.asyncio
async def test_compute_scores_hallucination_penalised(monkeypatch):
    """Claims that don't match any source → faithfulness < 1.0."""

    async def fake_extract(report):  # noqa: ARG001
        return [
            ExtractedClaim(claim="orthogonal hallucinated claim", claim_type="factual"),
        ]

    def fake_embed(text):
        # Claim orthogonal to the single source
        return [1.0, 0.0, 0.0] if "claim" in text else [0.0, 1.0, 0.0]

    monkeypatch.setattr(metrics_mod, "extract_claims", fake_extract)
    monkeypatch.setattr(metrics_mod, "embed_text", fake_embed)
    monkeypatch.setattr(attribution_mod, "embed_text", fake_embed)

    score = await compute_scores(
        "topic",
        "# report",
        {"s1": "unrelated source text"},
        threshold=0.8,  # high bar — won't match orthogonal
    )
    assert score.faithfulness == 0.0
    assert score.citation_recall == 0.0


@pytest.mark.asyncio
async def test_compute_scores_empty_report(monkeypatch):
    async def fake_extract(report):  # noqa: ARG001
        return []

    def fake_embed(text):  # noqa: ARG001
        return [0.0, 1.0, 0.0]

    monkeypatch.setattr(metrics_mod, "extract_claims", fake_extract)
    monkeypatch.setattr(metrics_mod, "embed_text", fake_embed)

    score = await compute_scores("topic", "", {})
    # No claims → faithfulness/recall are trivially 1.0 (vacuous)
    assert score.faithfulness == 1.0
    assert score.citation_recall == 1.0


def test_aggregate_averages():
    s1 = RagasScore(0.9, 0.8, 0.85, 0.7)
    s2 = RagasScore(0.5, 0.4, 0.45, 0.6)
    agg = aggregate([s1, s2])
    assert agg.faithfulness == 0.7
    assert agg.citation_precision == 0.6
    assert agg.citation_recall == 0.65
    assert agg.answer_relevance == 0.65


def test_aggregate_empty():
    agg = aggregate([])
    assert agg == RagasScore(0.0, 0.0, 0.0, 0.0)
