"""Structured logging setup for the Research Assistant.

Mode is controlled by `settings.log_format`:
    human  → colourful ConsoleRenderer (local dev)
    json   → JSON lines (production; easy to parse in Loki/ELK)

Both paths route stdlib `logging` records (fastapi, uvicorn, langchain, psycopg)
through structlog so they share the same processor chain (request_id, etc).

Request-scoped context (e.g. `request_id`, `thread_id`) is carried via
`contextvars` populated by `RequestIDMiddleware`. Anything bound with
`structlog.contextvars.bind_contextvars(...)` automatically appears in
every subsequent log in that request.
"""

import logging
import sys

import structlog

from app.config import settings


def configure_logging() -> None:
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    is_json = settings.log_format.lower() == "json"

    if is_json:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # Configure structlog loggers (structlog.get_logger())
    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging through the same formatter so library logs (uvicorn,
    # fastapi, langchain, psycopg) match the JSON / human format.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Clear any handlers installed by uvicorn/basicConfig so we don't double-print.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())

    # Tame chatty libs
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
