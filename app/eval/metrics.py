"""RAGAS-style quality metrics for generated reports.

Four metrics, each with a documented definition:

  * **faithfulness**   — fraction of extracted claims that are supported by
                         at least one source section (cosine sim ≥ threshold).
                         Measures "does the report stick to the sources" —
                         low = hallucination.

  * **citation_precision** — fraction of claims marked sourced that truly
                             match their chosen source (retrieval-side
                             correctness).

  * **citation_recall**    — fraction of factual/statistic claims that got
                             ANY source match at all. Measures coverage.

  * **answer_relevance**   — cosine similarity between topic embedding and
                             the report's overall embedding. Measures
                             "did the model stay on topic".

All metrics are in [0, 1]. Higher = better.

No external RAGAS library — we build these directly on top of the same
primitives the runtime uses (embed_text + extract_claims + attribute),
so eval scores reflect exactly what production would see.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass

import numpy as np

from app.config import settings
from app.tools.attribution import attribute, extract_claims
from app.tools.retriever import embed_text


@dataclass
class RagasScore:
    faithfulness: float
    citation_precision: float
    citation_recall: float
    answer_relevance: float

    def as_dict(self) -> dict:
        return asdict(self)


def _cosine(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
    return float(np.dot(va, vb) / denom)


async def compute_scores(
    topic: str,
    final_report: str,
    sources: dict[str, str],
    *,
    threshold: float | None = None,
) -> RagasScore:
    """Compute RAGAS-style metrics for one generated report.

    Args:
        topic: the original research topic.
        final_report: the generated markdown report.
        sources: dict of section -> supporting text (search_results).
        threshold: cosine threshold for "supported" (default from settings).
    """
    if threshold is None:
        threshold = settings.similarity_threshold

    claims = await extract_claims(final_report)
    factual = [c for c in claims if c.claim_type in ("factual", "statistic", "definition")]

    # faithfulness & citation_recall need per-claim source attribution
    citations = await attribute(factual, sources, threshold=threshold)

    if factual:
        supported = sum(1 for c in citations if c["source_section"])
        faithfulness = supported / len(factual)
        citation_recall = supported / len(factual)
    else:
        faithfulness = 1.0
        citation_recall = 1.0

    # citation_precision: of claims that picked a source, how many have a
    # strong cosine match? A claim with `score` just above threshold is a
    # weaker attribution than one deep above.
    sourced = [c for c in citations if c["source_section"]]
    precision = sum(1 for c in sourced if c["score"] >= 0.85) / len(sourced) if sourced else 0.0

    # answer_relevance: report embedding vs. topic embedding
    topic_vec, report_vec = await asyncio.gather(
        asyncio.to_thread(embed_text, topic),
        asyncio.to_thread(embed_text, final_report[:3000]),  # cap to avoid huge embed
    )
    answer_relevance = _cosine(topic_vec, report_vec)
    answer_relevance = max(0.0, min(1.0, answer_relevance))

    return RagasScore(
        faithfulness=round(faithfulness, 3),
        citation_precision=round(precision, 3),
        citation_recall=round(citation_recall, 3),
        answer_relevance=round(answer_relevance, 3),
    )


def aggregate(scores: list[RagasScore]) -> RagasScore:
    """Average a list of per-run scores into an aggregate."""
    if not scores:
        return RagasScore(0.0, 0.0, 0.0, 0.0)
    n = len(scores)
    return RagasScore(
        faithfulness=round(sum(s.faithfulness for s in scores) / n, 3),
        citation_precision=round(sum(s.citation_precision for s in scores) / n, 3),
        citation_recall=round(sum(s.citation_recall for s in scores) / n, 3),
        answer_relevance=round(sum(s.answer_relevance for s in scores) / n, 3),
    )
