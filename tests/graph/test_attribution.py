"""Citation attribution unit tests.

Extraction and the LLM call are stubbed — we test the matching arithmetic
(embedding-similarity claim → source) and the footnote-injection shape.
"""

from __future__ import annotations

import pytest

from app.tools import attribution as attribution_mod
from app.tools.attribution import ExtractedClaim, inject_footnotes


@pytest.mark.asyncio
async def test_attribute_picks_highest_similarity(monkeypatch):
    # Deterministic embeddings: one-hot vectors per section + claim
    table = {
        "climate": [1.0, 0.0, 0.0],
        "economy": [0.0, 1.0, 0.0],
        "technology": [0.0, 0.0, 1.0],
        "climate-claim": [0.9, 0.1, 0.1],  # closest to "climate"
        "econ-claim": [0.1, 0.9, 0.1],  # closest to "economy"
        "tech-claim": [0.1, 0.1, 0.95],  # closest to "technology"
    }

    def fake_embed(text):
        return table[text]

    monkeypatch.setattr(attribution_mod, "embed_text", fake_embed)

    sources = {"climate": "climate", "economy": "economy", "technology": "technology"}
    claims = [
        ExtractedClaim(claim="climate-claim", claim_type="factual"),
        ExtractedClaim(claim="econ-claim", claim_type="statistic"),
        ExtractedClaim(claim="tech-claim", claim_type="definition"),
    ]

    result = await attribution_mod.attribute(claims, sources, threshold=0.5)
    by_claim = {r["claim"]: r["source_section"] for r in result}
    assert by_claim["climate-claim"] == "climate"
    assert by_claim["econ-claim"] == "economy"
    assert by_claim["tech-claim"] == "technology"


@pytest.mark.asyncio
async def test_attribute_flags_unsourced_below_threshold(monkeypatch):
    """An orthogonal claim/source pair must be marked unsourced."""

    def fake_embed(text):
        return [1.0, 0.0, 0.0] if "claim" in text else [0.0, 1.0, 0.0]

    monkeypatch.setattr(attribution_mod, "embed_text", fake_embed)

    claims = [ExtractedClaim(claim="a factual claim unrelated to sources", claim_type="factual")]
    result = await attribution_mod.attribute(
        claims, {"unrelated-section": "source body"}, threshold=0.99
    )
    assert result[0]["source_section"] is None
    assert result[0]["score"] < 0.99


def test_inject_footnotes_builds_sources_table():
    report = "# Title\n\nSome body paragraph."
    citations = [
        {"claim": "A fact", "claim_type": "factual", "source_section": "S1", "score": 0.8},
        {"claim": "Floater", "claim_type": "factual", "source_section": None, "score": 0.3},
    ]
    out = inject_footnotes(report, citations)
    assert "## Sources" in out
    assert "A fact" in out
    assert "Unsourced claims" in out
    assert "Floater" in out


def test_inject_footnotes_noop_on_empty_citations():
    report = "# Title\n\nBody."
    assert inject_footnotes(report, []) == report
