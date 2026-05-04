"""Telegram bot interface for the Research Assistant.

Flow:
    User sends any text message  →  treated as research topic
    Bot calls POST /research/start  →  gets thread_id, sends plan with Approve / Edit
    User taps Approve  →  bot calls POST /approve (lightweight), then connects
                          to GET /stream to drive the graph + receive live events
    User taps Edit     →  bot asks to send a corrected plan
    Bot sends final report  →  markdown formatted, with token/cost summary from /metrics

FSM states:
    waiting_topic         — waiting for a topic
    waiting_plan_approval — plan generated, waiting for user decision
    researching           — full pipeline running
"""

import asyncio
import contextlib
import json
import logging

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties  # type: ignore[import]
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from app.config import settings

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# L-4: redact the Telegram bot token from every log record. Without this,
# aiogram's own logs or a traceback through httpx may surface the full
# token in plain text, which anyone grepping the log aggregator can steal.
class _TokenRedactionFilter(logging.Filter):
    def __init__(self, secrets: list[str]) -> None:
        super().__init__()
        self._secrets = [s for s in secrets if s]

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._secrets:
            return True
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return True
        for s in self._secrets:
            if s in msg:
                record.msg = msg.replace(s, "[REDACTED_TOKEN]")
                record.args = ()
        return True


_redact_filter = _TokenRedactionFilter([settings.telegram_bot_token, settings.research_api_key])
logging.getLogger().addFilter(_redact_filter)
for name in ("aiogram", "httpx", "httpcore"):
    logging.getLogger(name).addFilter(_redact_filter)

# ---------------------------------------------------------------------------
# FSM states
# ---------------------------------------------------------------------------


class ResearchFlow(StatesGroup):
    waiting_topic = State()
    waiting_plan_approval = State()
    researching = State()


# ---------------------------------------------------------------------------
# HTTP client (with API key header if configured)
# ---------------------------------------------------------------------------

_http: httpx.AsyncClient | None = None


def get_http() -> httpx.AsyncClient:
    global _http
    if _http is None:
        headers = {}
        if settings.research_api_key:
            headers["X-API-Key"] = settings.research_api_key
        _http = httpx.AsyncClient(
            base_url=settings.api_base_url,
            timeout=300,
            headers=headers,
        )
    return _http


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plan_keyboard(thread_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Approve", callback_data=f"approve:{thread_id}"),
                InlineKeyboardButton(text="✏️ Edit plan", callback_data=f"edit:{thread_id}"),
            ]
        ]
    )


def _format_plan(plan: list[str]) -> str:
    lines = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan))
    return f"📋 *Research plan:*\n\n{lines}"


def _split_long_message(text: str, limit: int = 4000) -> list[str]:
    """Telegram max message length is 4096 chars. Split by paragraphs."""
    if len(text) <= limit:
        return [text]
    parts = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break
        split_at = text.rfind("\n\n", 0, limit)
        if split_at == -1:
            split_at = limit
        parts.append(text[:split_at])
        text = text[split_at:].lstrip()
    return parts


_NODE_LABELS = {
    "planner": "📝 *planner* — outlining...",
    "researcher": "🔎 *researcher* — searching & summarizing...",
    "grader": "🧪 *grader* — checking relevance...",
    "web_search": "🌐 *web_search* — re-fetching fresh data...",
    "writer": "✍️ *writer* — writing sections...",
    "reviewer": "🧐 *reviewer* — reviewing quality...",
    "formatter": "📄 *formatter* — assembling final report...",
}


async def _stream_progress(thread_id: str, progress_msg: Message) -> None:
    """Consume the SSE stream and update a single progress message per node.

    Handles three event kinds from the API:
      * `progress` — node boundary (start/end). We edit the message label.
      * `token`    — LLM delta; we keep a counter and show it as "writing… N chars".
                     Telegram rate-limits message edits, so we only re-render
                     the tokens counter every ~1s.
      * `done` / `error` — terminal, return.
    """
    http = get_http()
    seen: set[str] = set()
    token_count = 0
    last_token_edit = 0.0
    async with http.stream("GET", f"/research/{thread_id}/stream") as resp:
        event_name: str | None = None
        async for raw_line in resp.aiter_lines():
            if not raw_line:
                continue
            if raw_line.startswith("event:"):
                event_name = raw_line.removeprefix("event:").strip()
                continue
            if raw_line.startswith("data:"):
                payload = raw_line.removeprefix("data:").strip()
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if event_name == "progress":
                    node = data.get("node")
                    label = _NODE_LABELS.get(node)
                    if label and node not in seen and data.get("phase") != "end":
                        seen.add(node)
                        with contextlib.suppress(Exception):
                            await progress_msg.edit_text(label)
                elif event_name == "token":
                    token_count += len(data.get("delta", ""))
                    now = asyncio.get_event_loop().time()
                    # throttle edits to ≤1/sec (Telegram would 429 otherwise)
                    if now - last_token_edit > 1.0:
                        last_token_edit = now
                        with contextlib.suppress(Exception):
                            await progress_msg.edit_text(
                                f"✍️ *writing* — {token_count:,} chars streamed…"
                            )
                elif event_name in ("done", "error"):
                    return


# ---------------------------------------------------------------------------
# Bot + Dispatcher
# ---------------------------------------------------------------------------

bot = Bot(
    token=settings.telegram_bot_token,
    default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
)
dp = Dispatcher(storage=MemoryStorage())


# /start
@dp.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.set_state(ResearchFlow.waiting_topic)
    await message.answer(
        "👋 *Research Assistant*\n\n"
        "Send me any topic and I'll produce a structured research report.\n\n"
        "Example: _LangGraph vs CrewAI in 2026_"
    )


# Any text while waiting for a topic
@dp.message(ResearchFlow.waiting_topic)
async def handle_topic(message: Message, state: FSMContext) -> None:
    topic = message.text.strip()
    await message.answer(f"🔍 Generating research plan for:\n*{topic}*\n\nPlease wait...")

    http = get_http()
    resp = await http.post("/research/start", json={"topic": topic})
    if resp.status_code != 200:
        await message.answer(f"❌ Failed to start research: {resp.text}")
        return

    thread_id = resp.json()["thread_id"]

    plan_resp = await http.get(f"/research/{thread_id}/plan")
    plan: list[str] = plan_resp.json().get("plan", []) if plan_resp.status_code == 200 else []

    await state.update_data(thread_id=thread_id, topic=topic)
    await state.set_state(ResearchFlow.waiting_plan_approval)

    await message.answer(
        _format_plan(plan),
        reply_markup=_plan_keyboard(thread_id),
    )


# Approve button
@dp.callback_query(F.data.startswith("approve:"))
async def handle_approve(callback: CallbackQuery, state: FSMContext) -> None:
    thread_id = callback.data.split(":", 1)[1]
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer(
        "✅ Plan approved! Starting research...\n\n_This takes ~1-2 minutes_"
    )
    await callback.answer()

    await state.set_state(ResearchFlow.researching)

    http = get_http()
    progress_msg = await callback.message.answer("⏳ starting...")

    # Step 1: mark approved (lightweight)
    approve_resp = await http.post(f"/research/{thread_id}/approve", json={})
    if approve_resp.status_code != 200:
        await progress_msg.edit_text(f"❌ Approve failed: {approve_resp.text}")
        await state.set_state(ResearchFlow.waiting_topic)
        return

    # Step 2: drive the graph via SSE, updating the progress message live
    try:
        await _stream_progress(thread_id, progress_msg)
    except Exception as e:
        log.exception("stream failed")
        await progress_msg.edit_text(f"❌ Stream error: {e}")
        await state.set_state(ResearchFlow.waiting_topic)
        return

    # Fetch final result
    result_resp = await http.get(f"/research/{thread_id}/result")
    if result_resp.status_code != 200:
        await progress_msg.edit_text("❌ Research failed.")
        await state.set_state(ResearchFlow.waiting_topic)
        return

    final_report = result_resp.json().get("final_report", "")
    metrics_resp = await http.get(f"/research/{thread_id}/metrics")
    metrics = metrics_resp.json() if metrics_resp.status_code == 200 else {}

    await progress_msg.edit_text("✅ *Done!*")

    for part in _split_long_message(final_report):
        await callback.message.answer(part)

    if metrics:
        await callback.message.answer(
            f"📊 *Usage stats:*\n"
            f"  Tokens: `{metrics.get('total_tokens', 0):,}`\n"
            f"  Cost: `${metrics.get('cost_usd', 0):.5f}`"
        )

    await state.set_state(ResearchFlow.waiting_topic)
    await callback.message.answer("Send me another topic to research 🚀")


# Edit button — ask user to send corrected plan
@dp.callback_query(F.data.startswith("edit:"))
async def handle_edit(callback: CallbackQuery, state: FSMContext) -> None:
    thread_id = callback.data.split(":", 1)[1]
    await callback.message.edit_reply_markup(reply_markup=None)
    await state.update_data(thread_id=thread_id)
    await callback.message.answer("✏️ Send me the corrected plan — one section per line:")
    await callback.answer()
    await state.set_state(ResearchFlow.waiting_plan_approval)


# Handle edited plan text
@dp.message(ResearchFlow.waiting_plan_approval)
async def handle_edited_plan(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    thread_id = data.get("thread_id")

    new_plan = [line.strip() for line in message.text.splitlines() if line.strip()]
    if not new_plan:
        await message.answer("❌ Plan is empty. Send at least one section.")
        return

    await message.answer(
        _format_plan(new_plan) + "\n\nLooks good?",
        reply_markup=_plan_keyboard(thread_id),
    )


# Fallback — any message outside of states
@dp.message()
async def fallback(message: Message, state: FSMContext) -> None:
    await state.set_state(ResearchFlow.waiting_topic)
    await message.answer("Send me a research topic to get started!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _on_shutdown() -> None:
    """Close HTTP client + bot session on shutdown."""
    global _http
    if _http is not None:
        await _http.aclose()
        _http = None
    await bot.session.close()
    log.info("bot_stopped")


async def main() -> None:
    log.info("Starting Telegram bot...")
    dp.shutdown.register(_on_shutdown)
    try:
        await dp.start_polling(bot, skip_updates=True)
    finally:
        await _on_shutdown()


if __name__ == "__main__":
    asyncio.run(main())
