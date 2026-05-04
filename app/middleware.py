"""HTTP middleware.

RequestIDMiddleware:
    Reads/generates an `X-Request-ID`, binds it into structlog's contextvars
    so every log line emitted during the request is tagged with request_id,
    and echoes the ID back on the response.

BodySizeLimitMiddleware:
    Rejects requests whose Content-Length exceeds `settings.max_request_body_bytes`
    (default 1 MiB) with HTTP 413 before the handler is ever called —
    prevents trivial OOM DoS on /start with giant topics.
"""

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from app.config import settings

_REQUEST_ID_HEADER = "X-Request-ID"
log = structlog.get_logger(__name__)


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized request bodies.

    Two-layer check (M-5):
      1. Fast path — trust the client's `Content-Length` if present and
         refuse immediately. Handles the common well-behaved case in O(1).
      2. Streaming path — if no Content-Length (chunked transfer) or the
         header is malformed, consume the body ourselves counting bytes,
         and 413 as soon as the cumulative size crosses the limit. Prevents
         attackers bypassing the check by omitting `Content-Length`.
    """

    async def dispatch(self, request: Request, call_next):
        limit = settings.max_request_body_bytes
        cl = request.headers.get("content-length")
        declared: int | None = None
        if cl is not None:
            try:
                declared = int(cl)
            except ValueError:
                declared = None
        if declared is not None and declared > limit:
            return _too_large_response(limit)

        if declared is None and request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            # Read body ourselves, capping at `limit + 1` bytes to detect overflow.
            body = b""
            async for chunk in request.stream():
                body += chunk
                if len(body) > limit:
                    return _too_large_response(limit)
            # Re-inject the body so downstream handlers can read it.
            request._body = body  # noqa: SLF001 (starlette's supported re-entry point)
        return await call_next(request)


def _too_large_response(limit: int) -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={"detail": f"Request body too large. Max {limit} bytes."},
    )


class RequestIDMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get(_REQUEST_ID_HEADER) or str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=rid,
            method=request.method,
            path=request.url.path,
        )
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("request_failed")
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers[_REQUEST_ID_HEADER] = rid
        log.info(
            "request_completed",
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        return response
