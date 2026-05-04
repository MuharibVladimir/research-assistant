import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from sqlalchemy import text as sql_text

from app.config import settings
from app.logging_config import configure_logging
from app.models.engine import SessionLocal
from app.observability import configure_sentry, instrument_prometheus

log = logging.getLogger(__name__)


def _configure_langsmith() -> None:
    """Set LangSmith env vars from config so tracing activates automatically."""
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    _configure_langsmith()
    configure_sentry()

    # Warm up graph + checkpointer pool so the first request isn't cold.
    from app.api.routes import get_graph, rate_bucket_cleanup_task, shutdown_graph
    from app.rate_limit import get_limiter, shutdown_limiter

    await get_graph()
    await get_limiter()

    # Fail loud if pgvector column dim diverges from code.
    try:
        from app.tools.retriever import validate_embedding_dimensions

        validate_embedding_dimensions()
    except RuntimeError:
        log.exception("embedding_dim_validation_failed")
        raise

    # Background cleanup of the in-memory rate-limit bucket dict.
    cleanup_task = asyncio.create_task(rate_bucket_cleanup_task())

    log.info("app_started")
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        # Cancel any outstanding speculative research tasks (G-14).
        from app.graph.speculative import REGISTRY

        REGISTRY.clear()
        await shutdown_limiter()
        await shutdown_graph()
        log.info("app_stopped")


app = FastAPI(
    title="Research Assistant",
    description="Multi-agent research assistant built with LangGraph + LangChain.",
    version="0.1.0",
    debug=settings.debug,
    lifespan=lifespan,
)

# Prometheus /metrics endpoint + HTTP metrics (latency, codes).
instrument_prometheus(app)

from app.api.cache_routes import router as cache_router  # noqa: E402
from app.api.routes import router  # noqa: E402
from app.middleware import BodySizeLimitMiddleware, RequestIDMiddleware  # noqa: E402

# Middleware order matters: size check first (reject early), then request id.
app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(RequestIDMiddleware)
app.include_router(router, prefix="/research")
app.include_router(cache_router, prefix="/cache", tags=["cache"])


@app.get("/health")
async def health() -> dict:
    """Liveness probe — just verifies the process is up."""
    return {"status": "ok"}


@app.get("/health/ready")
async def ready() -> dict:
    """Readiness probe — verifies DB connectivity."""
    try:
        with SessionLocal() as db:
            db.execute(sql_text("SELECT 1"))
    except Exception as e:  # noqa: BLE001
        log.exception("readiness_probe_failed")
        return {"status": "error", "db": "unreachable", "error": str(e)}
    return {"status": "ok", "db": "ok"}
