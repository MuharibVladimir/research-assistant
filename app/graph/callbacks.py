"""Token usage callback for tracking OpenAI API costs.

UsageCallback collects token counts across all LLM calls in a single graph
run AND pushes them to Prometheus counters so dashboards aggregate cost
over time. Per-model pricing comes from `app.pricing.chat_price` — change
`OPENAI_MODEL` in .env and the cost calculation follows automatically.

A side-channel registry `_SESSION_TOTALS` exposes the running token count
by thread_id so nodes (e.g. `reviewer_node`) can enforce the H-4 session
cost cap without threading the callback object through graph state.
"""

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.config import settings
from app.observability import LLM_COST_USD_TOTAL, LLM_TOKENS_TOTAL
from app.pricing import chat_cost

# thread_id → running token total (populated by UsageCallback when the
# callback was registered with a `thread_id` hint). Read-only for nodes.
_SESSION_TOTALS: dict[str, int] = {}


def get_session_tokens(thread_id: str) -> int:
    """Return the cumulative token count observed so far for this thread."""
    return _SESSION_TOTALS.get(thread_id, 0)


def reset_session_tokens(thread_id: str) -> None:
    _SESSION_TOTALS.pop(thread_id, None)


class UsageCallback(BaseCallbackHandler):
    """Accumulates token usage across multiple LLM calls.

    Pass `thread_id=` to have the running total mirrored into the module-level
    `_SESSION_TOTALS` map so graph nodes can enforce per-session budgets
    without shuttling the callback object around.
    """

    def __init__(self, thread_id: str | None = None) -> None:
        super().__init__()
        self.thread_id = thread_id
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.cost_usd = 0.0

    def _track(self, prompt_tokens: int, completion_tokens: int) -> None:
        model = settings.openai_model
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        cost = chat_cost(model, prompt_tokens, completion_tokens)
        self.cost_usd += cost

        LLM_TOKENS_TOTAL.labels(model=model, type="prompt").inc(prompt_tokens)
        LLM_TOKENS_TOTAL.labels(model=model, type="completion").inc(completion_tokens)
        LLM_COST_USD_TOTAL.labels(model=model).inc(cost)

        if self.thread_id is not None:
            _SESSION_TOTALS[self.thread_id] = (
                _SESSION_TOTALS.get(self.thread_id, 0) + prompt_tokens + completion_tokens
            )

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        saw_usage = False
        for generations in response.generations:
            for gen in generations:
                usage = (
                    getattr(gen.message, "usage_metadata", None)
                    if hasattr(gen, "message")
                    else None
                )
                if usage:
                    self._track(usage.get("input_tokens", 0), usage.get("output_tokens", 0))
                    saw_usage = True

        # Fallback: older OpenAI `llm_output.token_usage` format
        if not saw_usage and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {}) or {}
            if token_usage:
                self._track(
                    token_usage.get("prompt_tokens", 0),
                    token_usage.get("completion_tokens", 0),
                )

    @property
    def usage(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }
