"""Integration tests that exercise the real pgvector/Postgres stack.

Spins up a pgvector-enabled Postgres container via testcontainers, runs
Alembic to head, and then drives the retriever + session ORM end-to-end.

These tests require Docker. They're marked `real_db` so CI can gate them:
`pytest -m real_db`.

Skipped if Docker is unavailable (e.g. CI without privileged runner).
"""

from __future__ import annotations

import uuid

import pytest

docker = pytest.importorskip("docker")  # skip if docker client not importable
try:
    from testcontainers.postgres import PostgresContainer
except ImportError:  # pragma: no cover
    pytest.skip("testcontainers not installed", allow_module_level=True)

pytestmark = pytest.mark.real_db


@pytest.fixture(scope="module")
def pg_container():
    """Start pgvector:pg17 once per module, tear down at the end."""
    container = PostgresContainer(
        image="pgvector/pgvector:pg17",
        username="research",
        password="research",
        dbname="research_db",
        driver="psycopg",  # psycopg3 driver marker for SQLAlchemy
    )
    try:
        container.start()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Docker not available: {e}")
    yield container
    container.stop()


@pytest.fixture
def pg_url(pg_container) -> str:
    # testcontainers builds the URL with +psycopg scheme because we set driver
    return pg_container.get_connection_url()


@pytest.fixture
def migrated_db(pg_url, monkeypatch):
    """Run Alembic migrations against the throwaway container."""
    from alembic import command
    from alembic.config import Config

    from app.config import settings

    monkeypatch.setattr(settings, "database_url", pg_url)
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", pg_url)
    command.upgrade(cfg, "head")
    return pg_url


def _make_engine(url: str):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(url, future=True)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


def test_research_session_round_trip(migrated_db):
    """Insert + fetch a ResearchSession against a real schema."""
    from app.models.db import ResearchSession

    _, Session = _make_engine(migrated_db)
    sid = uuid.uuid4()
    with Session() as db:
        db.add(
            ResearchSession(
                id=sid,
                topic="integration test topic",
                status="done",
                prompt_tokens=100,
                completion_tokens=200,
                total_tokens=300,
                cost_usd=0.0012,
                api_key_hash="a" * 64,
            )
        )
        db.commit()

    with Session() as db:
        loaded = db.get(ResearchSession, sid)
    assert loaded is not None
    assert loaded.topic == "integration test topic"
    assert loaded.total_tokens == 300
    assert loaded.api_key_hash == "a" * 64


def test_retriever_similarity_search(migrated_db, monkeypatch):
    """Insert a document with a known embedding, then retrieve it."""
    import app.models.engine as engine_mod
    from app.tools import retriever

    # Rewire SessionLocal to the test container
    _, Session = _make_engine(migrated_db)
    monkeypatch.setattr(engine_mod, "SessionLocal", Session)
    monkeypatch.setattr(retriever, "SessionLocal", Session)

    # Deterministic embeddings: a vector close to itself scores ~1.0 cosine.
    stored_vec = [1.0] + [0.0] * (retriever.EMBEDDING_DIMS - 1)
    query_vec = stored_vec  # identical → similarity = 1.0

    monkeypatch.setattr(retriever, "embed_text", lambda _t: query_vec)

    from app.models.db import Document

    doc_id = uuid.uuid4()
    with Session() as db:
        db.add(
            Document(
                id=doc_id,
                topic="pytest topic",
                section="pytest section",
                content="pytest cached content",
                embedding_vec=stored_vec,
            )
        )
        db.commit()

    # Force big threshold so only near-identical matches return
    hits = retriever.retrieve_relevant("any query", threshold=0.99)
    assert len(hits) == 1
    assert hits[0]["content"] == "pytest cached content"
    assert hits[0]["similarity"] > 0.99


def test_embedding_dim_validation_passes(migrated_db, monkeypatch):
    import app.models.engine as engine_mod
    from app.tools import retriever

    _, Session = _make_engine(migrated_db)
    monkeypatch.setattr(engine_mod, "SessionLocal", Session)
    monkeypatch.setattr(retriever, "SessionLocal", Session)
    # Should not raise — migration created a Vector(1536)
    retriever.validate_embedding_dimensions()
