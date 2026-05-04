"""Model router + fallback wiring.

Two things live here:

1. **Per-role model selection** — different graph nodes use different models.
   Reasoning-heavy roles (planner, reviewer) use a stronger model; cheap
   roles (summariser, grader) use a faster/cheaper one. Configure via
   `settings.MODEL_FOR_*`.

2. **Provider fallback** — if the primary provider (OpenAI) fails with a
   retryable error that tenacity already exhausted, the chain automatically
   falls back to Anthropic Claude. Implemented via
   `langchain_core.runnables.RunnableWithFallbacks`.

Why not `with_fallbacks` on each chain? Because each node has different
structured-output schemas. It's cleaner to return the LLM-with-fallback
from here and let each node compose its own `prompt | llm | parser`.
"""

from __future__ import annotations

import logging
from enum import StrEnum

import openai
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import settings

log = logging.getLogger(__name__)


class NodeRole(StrEnum):
    """Semantic role of the LLM call — drives model selection."""

    PLANNER = "planner"  # reasoning, outline — use strong model
    RESEARCHER = "researcher"  # summarising search results — cheap OK
    GRADER = "grader"  # deterministic yes/no — cheap + temp=0
    WRITER = "writer"  # creative prose — mid-tier
    REVIEWER = "reviewer"  # nuanced judgement — strong
    FORMATTER = "formatter"  # mechanical assembly — cheap
    JUDGE = "judge"  # eval LLM-as-judge — deterministic


_RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


def _model_for(role: NodeRole) -> str:
    """Look up the model name for a role, falling back to settings.openai_model."""
    attr = f"model_{role.value}"
    return getattr(settings, attr, None) or settings.openai_model


def _openai(role: NodeRole, *, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        model=_model_for(role),
        api_key=settings.openai_api_key,
        temperature=temperature,
        timeout=60,
    )


def _anthropic(temperature: float) -> BaseChatModel | None:
    """Build the Anthropic fallback if a key is configured."""
    if not settings.anthropic_api_key:
        return None
    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        temperature=temperature,
        timeout=60,
    )


def get_llm(role: NodeRole, *, deterministic: bool = False) -> BaseChatModel:
    """Return an LLM for the given node role, wrapped in provider fallback.

    The returned object is a LangChain `Runnable`, so callers can compose it
    freely with `with_structured_output`, `with_config`, or `prompt | llm | parser`.
    """
    temp = settings.llm_temperature_deterministic if deterministic else settings.llm_temperature
    primary = _openai(role, temperature=temp)

    fallback = _anthropic(temp)
    if fallback is None:
        return primary

    # `with_fallbacks` auto-switches on `exceptions_to_handle`. tenacity retries
    # each provider internally; here we only fall across providers.
    return primary.with_fallbacks(
        fallbacks=[fallback],
        exceptions_to_handle=_RETRYABLE,
    )
