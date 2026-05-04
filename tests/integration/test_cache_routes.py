"""Cache invalidation endpoints.

After RBAC hardening, DELETE endpoints require a distinct `X-Admin-API-Key`
header matching `settings.admin_api_key`. A regular research API key (even
a valid one) must be rejected. Dev-mode (empty admin_api_key) disables
the endpoints entirely — fail-closed.
"""

import pytest

from app.config import settings

_ADMIN_HEADERS = {"X-Admin-API-Key": "admin-secret"}


@pytest.fixture(autouse=True)
def _set_admin_key(monkeypatch):
    """Configure a distinct admin key for the tests below."""
    monkeypatch.setattr(settings, "admin_api_key", "admin-secret")
    monkeypatch.setattr(settings, "research_api_key", "user-secret")


@pytest.mark.asyncio
async def test_delete_documents_requires_filter(async_client):
    resp = await async_client.delete("/cache/documents", headers=_ADMIN_HEADERS)
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_delete_documents_with_topic_filter(async_client):
    resp = await async_client.delete("/cache/documents?topic=X", headers=_ADMIN_HEADERS)
    assert resp.status_code == 200
    assert "deleted" in resp.json()


@pytest.mark.asyncio
async def test_delete_documents_section_requires_topic(async_client):
    resp = await async_client.delete("/cache/documents?section=S", headers=_ADMIN_HEADERS)
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_delete_topics_requires_filter(async_client):
    resp = await async_client.delete("/cache/topics", headers=_ADMIN_HEADERS)
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_delete_topics_malformed_uuid(async_client):
    resp = await async_client.delete("/cache/topics?cache_id=not-a-uuid", headers=_ADMIN_HEADERS)
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_delete_topics_by_ttl(async_client):
    resp = await async_client.delete("/cache/topics?older_than_days=14", headers=_ADMIN_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 0  # fake DB returns no rows


# ---------------------------------------------------------------------------
# RBAC tests — regular key rejected, missing key rejected, wrong key rejected.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_rejects_user_api_key(async_client):
    """A valid *research* API key must NOT be accepted on admin endpoints."""
    resp = await async_client.delete(
        "/cache/documents?topic=X", headers={"X-API-Key": "user-secret"}
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_delete_rejects_missing_admin_header(async_client):
    resp = await async_client.delete("/cache/documents?topic=X")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_delete_rejects_wrong_admin_key(async_client):
    resp = await async_client.delete(
        "/cache/documents?topic=X", headers={"X-Admin-API-Key": "not-the-admin-key"}
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_delete_fails_closed_when_admin_key_unset(async_client, monkeypatch):
    """No ADMIN_API_KEY configured → endpoints report 503, not 200."""
    monkeypatch.setattr(settings, "admin_api_key", "")
    resp = await async_client.delete(
        "/cache/documents?topic=X", headers={"X-Admin-API-Key": "anything"}
    )
    assert resp.status_code == 503
