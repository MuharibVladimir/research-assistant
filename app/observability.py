"""Observability wiring: Sentry for errors, Prometheus for metrics.

Usage (from app.main.lifespan):
    configure_sentry()          # enable Sentry error tracking if SENTRY_DSN set
    instrument_prometheus(app)  # expose /metrics and auto-HTTP metrics

Custom metrics exported to /metrics (consumed by Prometheus in docker-compose):
    graph_node_duration_seconds{node}      — histogram of per-node latency
    graph_nodes_total{node,status}         — counter (success|error)
    cache_hit_total / cache_miss_total     — vector cache effectiveness
    llm_tokens_total{model,type}           — prompt + completion tokens
    llm_cost_usd_total{model}              — accumulated $ cost

These counters are bumped from the graph callbacks and researcher node.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from prometheus_client import Counter, Gauge, Histogram

from app.config import settings

if TYPE_CHECKING:
    from fastapi import FastAPI

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom Prometheus metrics
# ---------------------------------------------------------------------------

GRAPH_NODE_DURATION = Histogram(
    "graph_node_duration_seconds",
    "Latency of a single graph node invocation.",
    labelnames=("node",),
    buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120),
)

GRAPH_NODES_TOTAL = Counter(
    "graph_nodes_total",
    "Total graph node invocations, split by outcome.",
    labelnames=("node", "status"),  # status: success | error
)

CACHE_HIT_TOTAL = Counter(
    "cache_hit_total",
    "Number of pgvector cache hits (relevant cached content found).",
)

CACHE_MISS_TOTAL = Counter(
    "cache_miss_total",
    "Number of pgvector cache misses (fell back to Tavily web search).",
)

LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total tokens consumed by LLM calls.",
    labelnames=("model", "type"),  # type: prompt | completion
)

LLM_COST_USD_TOTAL = Counter(
    "llm_cost_usd_total",
    "Accumulated LLM cost in USD.",
    labelnames=("model",),
)

CIRCUIT_BREAKER_STATE = Gauge(
    "circuit_breaker_state",
    "State of a named circuit breaker: 0=closed, 1=half_open, 2=open.",
    labelnames=("name",),
)


# ---------------------------------------------------------------------------
# Sentry
# ---------------------------------------------------------------------------


_SENSITIVE_KEYS = {
    "x_api_key",
    "x-api-key",
    "api_key",
    "apikey",
    "authorization",
    "password",
    "openai_api_key",
    "anthropic_api_key",
    "tavily_api_key",
    "telegram_bot_token",
    "langsmith_api_key",
    "sentry_dsn",
    "redis_password",
    "metrics_token",
    "mcp_write_token",
}


def _sentry_scrub(event: dict, hint: dict | None = None) -> dict | None:  # noqa: ARG001
    """M-6: strip API keys / secrets from Sentry events before upload.

    Covers three surfaces where secrets commonly leak:
      * request.headers (X-API-Key, Authorization)
      * exception frame locals (`x_api_key`, `api_key`, etc.)
      * top-level request/body cookies/query-string
    """
    req = event.get("request") or {}
    for field in ("headers", "cookies"):
        bag = req.get(field)
        if isinstance(bag, dict):
            for k in list(bag):
                if k.lower() in _SENSITIVE_KEYS:
                    bag[k] = "[REDACTED]"

    for exc in event.get("exception", {}).get("values", []):
        for frame in exc.get("stacktrace", {}).get("frames", []):
            for k in list(frame.get("vars", {})):
                if k.lower() in _SENSITIVE_KEYS:
                    frame["vars"][k] = "[REDACTED]"
    return event


def configure_sentry() -> None:
    """Initialize Sentry if SENTRY_DSN is configured. No-op otherwise."""
    if not settings.sentry_dsn:
        return
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration

        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.sentry_environment,
            traces_sample_rate=settings.sentry_traces_sample_rate,
            integrations=[
                FastApiIntegration(),
                StarletteIntegration(),
                SqlalchemyIntegration(),
                LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
            ],
            before_send=_sentry_scrub,
            send_default_pii=False,
        )
        log.info("sentry_configured environment=%s", settings.sentry_environment)
    except Exception:  # noqa: BLE001
        log.exception("sentry_init_failed")


# ---------------------------------------------------------------------------
# Prometheus FastAPI instrumentator
# ---------------------------------------------------------------------------


def instrument_prometheus(app: FastAPI) -> None:
    """Attach Prometheus instrumentation and expose `/metrics`.

    When `settings.metrics_token` is set (M-1), scrapes must send
    `Authorization: Bearer <token>` — otherwise `/metrics` leaks internal
    cost, cache, and error cardinality signals to anyone who can reach it.
    """
    from fastapi import HTTPException, Request
    from prometheus_fastapi_instrumentator import Instrumentator

    def _verify_metrics(request: Request) -> None:
        expected = settings.metrics_token
        if not expected:
            return  # open-mode
        header = request.headers.get("authorization", "")
        if not header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Bearer token required")
        token = header.removeprefix("Bearer ").strip()
        import hmac

        if not hmac.compare_digest(token, expected):
            raise HTTPException(status_code=403, detail="Invalid metrics token")

    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_instrument_requests_inprogress=True,
        inprogress_labels=True,
        excluded_handlers=["/metrics", "/health", "/health/ready"],
    ).instrument(app).expose(
        app,
        endpoint="/metrics",
        include_in_schema=False,
        dependencies=[__import__("fastapi").Depends(_verify_metrics)],
    )
