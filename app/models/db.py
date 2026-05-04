import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ResearchSession(Base):
    __tablename__ = "research_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="pending"
    )  # pending | planning | researching | writing | reviewing | done | error
    final_report: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Token usage metrics
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # SHA-256 hex of the X-API-Key that created this session.
    # Nullable so dev-mode (no auth) still works.
    api_key_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic: Mapped[str] = mapped_column(String(500), nullable=False)
    section: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_vec: Mapped[list | None] = mapped_column(Vector(1536), nullable=True)
    # G-9 parent-doc fields — let retrieval return chunk + surrounding context.
    parent_doc_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    parent_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_offset_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_offset_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class GoldenReport(Base):
    """Human-annotated evaluation ground truth (G-1).

    `annotations_json` is a list of `{annotator_id, faithfulness, relevance,
    depth, factuality, notes}` dicts. Multiple annotators per topic let us
    compute inter-annotator agreement (Fleiss' κ) and detect judge drift.
    """

    __tablename__ = "golden_reports"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    report: Mapped[str] = mapped_column(Text, nullable=False)
    annotations_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UserResearchHistory(Base):
    """Per-user episodic memory (G-12).

    Every completed research session persists a compact summary keyed by
    the caller's api_key_hash. Before the planner runs on a new topic, we
    pull the top-k semantically-similar prior summaries from the same user
    and include them as context — lets follow-up research build on earlier
    findings instead of repeating them.
    """

    __tablename__ = "user_research_history"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    api_key_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    topic: Mapped[str] = mapped_column(String(500), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    topic_vec: Mapped[list | None] = mapped_column(Vector(1536), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EvalRun(Base):
    """A single eval-harness run, recorded for drift detection (G-2)."""

    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    git_sha: Mapped[str | None] = mapped_column(String(40), nullable=True)
    metric_scores_json: Mapped[str] = mapped_column(Text, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class TopicCache(Base):
    """Full-report cache keyed by topic embedding.

    Skips the entire graph when a semantically similar topic was researched
    recently (within TTL). `hit_count` tracks cache-effectiveness telemetry.

    User scoping (C-3): `api_key_hash` restricts cache hits to the caller
    that originally populated the row. NULL means dev-mode (anonymous) or
    pre-migration legacy data — those rows are only served to anonymous
    callers, never cross-users.
    """

    __tablename__ = "topic_cache"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic: Mapped[str] = mapped_column(String(500), nullable=False)
    final_report: Mapped[str] = mapped_column(Text, nullable=False)
    citations_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    topic_vec: Mapped[list | None] = mapped_column(Vector(1536), nullable=True)
    api_key_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
