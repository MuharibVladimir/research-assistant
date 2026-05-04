"""OpenAI model pricing table — USD per 1000 tokens.

Source: OpenAI pricing page (input/output rates). Keep this in one place so
changing `OPENAI_MODEL` in .env automatically picks up the right cost.

If a model is not in the table we fall back to `DEFAULT_PRICE` and log a
warning — the user still gets *a* cost estimate, just potentially wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPrice:
    prompt_per_1k: float  # input tokens
    completion_per_1k: float  # output tokens


# NOTE: Values below are the mainstream OpenAI list prices as of early 2026.
# Adjust if your account has different rates (enterprise, volume discounts).
_CHAT_MODELS: dict[str, ModelPrice] = {
    # gpt-4o family
    "gpt-4o": ModelPrice(0.0025, 0.01),
    "gpt-4o-2024-08-06": ModelPrice(0.0025, 0.01),
    "gpt-4o-mini": ModelPrice(0.00015, 0.0006),
    "gpt-4o-mini-2024-07-18": ModelPrice(0.00015, 0.0006),
    # gpt-4.1 family
    "gpt-4.1": ModelPrice(0.002, 0.008),
    "gpt-4.1-mini": ModelPrice(0.0004, 0.0016),
    "gpt-4.1-nano": ModelPrice(0.0001, 0.0004),
    # older
    "gpt-4-turbo": ModelPrice(0.01, 0.03),
    "gpt-3.5-turbo": ModelPrice(0.0005, 0.0015),
}

_EMBEDDING_MODELS: dict[str, float] = {
    "text-embedding-3-small": 0.00002,  # per 1k
    "text-embedding-3-large": 0.00013,
    "text-embedding-ada-002": 0.0001,
}

DEFAULT_PRICE = ModelPrice(0.00015, 0.0006)  # gpt-4o-mini — cheapest well-known


def chat_price(model: str) -> ModelPrice:
    """Return (prompt, completion) per-1k USD rates for a chat model."""
    price = _CHAT_MODELS.get(model)
    if price is None:
        log.warning("unknown_chat_model_falling_back model=%s", model)
        return DEFAULT_PRICE
    return price


def embedding_price_per_1k(model: str) -> float:
    """Return per-1k USD rate for an embedding model."""
    price = _EMBEDDING_MODELS.get(model)
    if price is None:
        log.warning("unknown_embedding_model_falling_back model=%s", model)
        return 0.00002
    return price


def chat_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute USD cost for a single chat completion."""
    p = chat_price(model)
    return (prompt_tokens / 1000) * p.prompt_per_1k + (
        completion_tokens / 1000
    ) * p.completion_per_1k


def embedding_cost(model: str, tokens: int) -> float:
    return (tokens / 1000) * embedding_price_per_1k(model)
