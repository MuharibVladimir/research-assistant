"""PII redaction and safety classifier tests (G-15)."""

from __future__ import annotations

import pytest

from app.config import settings
from app.tools import safety as safety_mod


@pytest.fixture(autouse=True)
def _enable_pii(monkeypatch):
    """Most tests below exercise the regex redaction path — force presidio-off
    by raising ImportError when safety tries to load it, so fallback runs."""
    monkeypatch.setattr(settings, "pii_redaction_enabled", True)

    # Force the presidio import in redact_pii to fail cleanly — the fallback
    # regex path is what we're exercising.
    def _boom():
        raise ImportError("presidio unavailable")

    monkeypatch.setattr(safety_mod, "_get_presidio", _boom)
    yield


def test_redact_disabled_returns_input_unchanged(monkeypatch):
    monkeypatch.setattr(settings, "pii_redaction_enabled", False)
    text = "Email me at alice@example.com"
    out, counts = safety_mod.redact_pii(text)
    assert out == text
    assert counts == {}


def test_redact_email():
    out, counts = safety_mod.redact_pii("Contact me: alice@example.com please")
    assert "alice@example.com" not in out
    assert "[REDACTED_EMAIL]" in out
    assert counts == {"EMAIL": 1}


def test_redact_phone_us():
    out, counts = safety_mod.redact_pii("Call 415-555-1234 any time")
    assert "415-555-1234" not in out
    assert "[REDACTED_PHONE]" in out
    assert counts.get("PHONE") == 1


def test_redact_ssn():
    out, counts = safety_mod.redact_pii("SSN: 123-45-6789")
    assert "123-45-6789" not in out
    assert "[REDACTED_SSN]" in out
    assert counts.get("SSN") == 1


def test_redact_credit_card():
    out, counts = safety_mod.redact_pii("Card 4111 1111 1111 1111 expires 12/30")
    assert "4111 1111 1111 1111" not in out
    assert "[REDACTED_CREDIT_CARD]" in out
    assert counts.get("CREDIT_CARD") == 1


def test_redact_multiple_types_at_once():
    out, counts = safety_mod.redact_pii("Reach me at a@b.co or 415-555-9999; SSN 999-88-7777.")
    assert "[REDACTED_EMAIL]" in out
    assert "[REDACTED_PHONE]" in out
    assert "[REDACTED_SSN]" in out
    assert counts.get("EMAIL") == 1
    assert counts.get("PHONE") == 1
    assert counts.get("SSN") == 1


def test_redact_is_noop_on_empty():
    assert safety_mod.redact_pii("") == ("", {})


@pytest.mark.asyncio
async def test_audit_report_disabled_returns_empty(monkeypatch):
    monkeypatch.setattr(settings, "safety_classifier_enabled", False)
    flags = await safety_mod.audit_report("a " * 300)
    assert flags == []


@pytest.mark.asyncio
async def test_audit_report_short_input_skips_llm(monkeypatch):
    monkeypatch.setattr(settings, "safety_classifier_enabled", True)

    class Boom:
        async def ainvoke(self, *_a, **_kw):
            raise AssertionError("should not be called")

    monkeypatch.setattr(safety_mod, "_safety_chain", Boom())
    flags = await safety_mod.audit_report("too short")
    assert flags == []


@pytest.mark.asyncio
async def test_audit_report_returns_flags(monkeypatch):
    monkeypatch.setattr(settings, "safety_classifier_enabled", True)

    class FakeReport:
        @staticmethod
        def _flag():
            from app.tools.safety import SafetyFlag

            return SafetyFlag(
                kind="speculative_claim",
                quote="The CEO definitely ate the paperwork",
                explanation="Unsupported claim about a named person",
            )

        def __init__(self):
            self.flags = [self._flag()]

    class FakeChain:
        async def ainvoke(self, *_a, **_kw):
            return FakeReport()

    monkeypatch.setattr(safety_mod, "_safety_chain", FakeChain())
    out = await safety_mod.audit_report("x" * 500)
    assert len(out) == 1
    assert out[0]["kind"] == "speculative_claim"
