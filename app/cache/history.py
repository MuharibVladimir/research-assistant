"""Per-user episodic memory — research history retrieval (G-12).

After every completed session we store a compact summary of the topic and
final report in `user_research_history`, keyed by `api_key_hash`. Before
the planner runs on a new topic, we pull the top-k semantically similar
prior summaries for the same user and inject them as context so the
planner can build on prior findings rather than duplicate them.

Anonymous callers (empty RESEARCH_API_KEY) are intentionally excluded —
we don't want to pollute one user's history with another's research.
"""

from __future__ import annotations

import logging
import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, text

from app.models.db import UserResearchHistory
from app.models.engine import SessionLocal
from app.tools.retriever import EMBEDDING_DIMS, embed_text

log = logging.getLogger(__name__)


# Summaries are short enough that a 3-sentence compression is enough —
# no LLM call needed if the final_report already has an executive-summary
# heading, but easy to add one via recursive_summarise if not.
_MAX_SUMMARY_CHARS = 2000


_LOOKUP_SQL = text(
    """
    SELECT id, topic, summary,
           1 - (topic_vec <=> :vec) AS similarity
    FROM user_research_history
    WHERE api_key_hash = :key_hash
      AND topic_vec IS NOT NULL
    ORDER BY topic_vec <=> :vec
    LIMIT :k
    """
).bindparams(
    bindparam("vec", type_=Vector(EMBEDDING_DIMS)),
    bindparam("key_hash"),
    bindparam("k"),
)


def record(
    api_key_hash: str,
    session_id: uuid.UUID | str,
    topic: str,
    summary: str,
) -> None:
    """Persist a compact record of a finished session for future reuse."""
    if not api_key_hash or not summary:
        return
    summary = summary[:_MAX_SUMMARY_CHARS]
    vec = embed_text(topic)
    sid = session_id if isinstance(session_id, uuid.UUID) else uuid.UUID(str(session_id))
    with SessionLocal() as db:
        db.add(
            UserResearchHistory(
                id=uuid.uuid4(),
                api_key_hash=api_key_hash,
                session_id=sid,
                topic=topic[:500],
                summary=summary,
                topic_vec=vec,
            )
        )
        db.commit()


def fetch_relevant(
    api_key_hash: str,
    topic: str,
    *,
    top_k: int = 3,
    min_similarity: float = 0.7,
) -> list[dict]:
    """Retrieve top-k semantically-similar prior researches for this user."""
    if not api_key_hash:
        return []
    vec = embed_text(topic)
    with SessionLocal() as db:
        rows = db.execute(
            _LOOKUP_SQL, {"vec": vec, "key_hash": api_key_hash, "k": top_k}
        ).fetchall()
    out: list[dict] = []
    for row in rows:
        sim = float(row.similarity)
        if sim < min_similarity:
            continue
        out.append(
            {
                "id": str(row.id),
                "topic": row.topic,
                "summary": row.summary,
                "similarity": sim,
            }
        )
    return out
