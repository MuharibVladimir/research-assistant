"""Cache invalidation endpoints.

Both caches (documents, topic_cache) expire via TTL, but production needs
explicit invalidation for:
  * **Corrections** — a cached document turned out to contain outdated /
    incorrect facts; admin nukes it so future retrievals don't pick it up.
  * **GDPR / content removal** — user requests deletion of their content.
  * **Prompt-regression recovery** — a bad eval gate run contaminated the
    cache with low-quality summaries; wipe and let the system re-learn.

Auth: requires a SEPARATE `X-Admin-API-Key` header matching
`settings.admin_api_key`. Regular research API keys are NOT accepted for
these endpoints — leaking one to a user should not let them wipe the cache
for everyone else. When `admin_api_key` is empty, DELETE /cache/* is
disabled entirely (fail-closed).

Counts deleted so the caller can tell the op worked.
"""

from __future__ import annotations

import hmac
import logging
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.config import settings
from app.models.engine import SessionLocal

log = logging.getLogger(__name__)


async def require_admin(
    x_admin_api_key: str | None = Header(default=None, alias="X-Admin-API-Key"),
) -> None:
    """Gate mutating cache endpoints on a distinct admin key.

    Fails closed when `settings.admin_api_key` is empty — better to make
    cache invalidation require an explicit ops action than to let the bar
    drop silently in dev.

    Constant-time compare (hmac.compare_digest) to defeat timing attacks.
    """
    expected = settings.admin_api_key
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Cache-invalidation is disabled: set ADMIN_API_KEY in the server "
                "environment to enable it."
            ),
        )
    if not x_admin_api_key or not hmac.compare_digest(x_admin_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="X-Admin-API-Key missing or invalid.",
        )


router = APIRouter()


class DeleteResponse(BaseModel):
    deleted: int = Field(..., description="Number of rows removed.")


@router.delete("/documents", response_model=DeleteResponse)
async def invalidate_documents(
    request: Request,
    topic: str | None = Query(None, description="Delete all chunks for this topic."),
    section: str | None = Query(None, description="Limit to a section (requires topic)."),
    older_than_days: int | None = Query(
        None, ge=1, description="Delete only docs older than this many days."
    ),
    _: None = Depends(require_admin),
) -> DeleteResponse:
    """Delete documents from the chunk cache.

    Requires `X-Admin-API-Key` (RBAC). At least one filter is required —
    we refuse to wipe the whole table. Writes an audit log line (L-1).
    """
    if topic is None and older_than_days is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one filter: topic=... and/or older_than_days=N",
        )
    if section is not None and topic is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="`section` filter requires `topic`.",
        )

    where: list[str] = []
    params: dict = {}
    if topic is not None:
        where.append("topic = :topic")
        params["topic"] = topic
    if section is not None:
        where.append("section = :section")
        params["section"] = section
    if older_than_days is not None:
        where.append("created_at < now() - make_interval(days => :days)")
        params["days"] = older_than_days

    stmt = text(f"DELETE FROM documents WHERE {' AND '.join(where)}")
    with SessionLocal() as db:
        result = db.execute(stmt, params)
        db.commit()
    deleted = result.rowcount or 0
    log.info(
        "cache_invalidation endpoint=documents topic=%r section=%r older_than_days=%r "
        "deleted=%d client_ip=%s",
        topic,
        section,
        older_than_days,
        deleted,
        request.client.host if request.client else "unknown",
    )
    return DeleteResponse(deleted=deleted)


@router.delete("/topics", response_model=DeleteResponse)
async def invalidate_topic_cache(
    request: Request,
    topic: str | None = Query(None, description="Delete rows with this topic."),
    cache_id: str | None = Query(None, description="Delete a specific cache_id."),
    older_than_days: int | None = Query(None, ge=1),
    _: None = Depends(require_admin),
) -> DeleteResponse:
    """Delete topic_cache rows (full-report semantic cache). Requires admin key."""
    if topic is None and cache_id is None and older_than_days is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one filter: topic, cache_id, or older_than_days.",
        )

    where: list[str] = []
    params: dict = {}
    if topic is not None:
        where.append("topic = :topic")
        params["topic"] = topic
    if cache_id is not None:
        try:
            params["cache_id"] = uuid.UUID(cache_id)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Malformed cache_id (expected UUID).",
            ) from e
        where.append("id = :cache_id")
    if older_than_days is not None:
        where.append("created_at < now() - make_interval(days => :days)")
        params["days"] = older_than_days

    stmt = text(f"DELETE FROM topic_cache WHERE {' AND '.join(where)}")
    with SessionLocal() as db:
        result = db.execute(stmt, params)
        db.commit()
    deleted = result.rowcount or 0
    log.info(
        "cache_invalidation endpoint=topics topic=%r cache_id=%r older_than_days=%r "
        "deleted=%d client_ip=%s",
        topic,
        cache_id,
        older_than_days,
        deleted,
        request.client.host if request.client else "unknown",
    )
    return DeleteResponse(deleted=deleted)
