# Research Assistant

> Multi-agent research pipeline built on **LangGraph + LangChain**, served via **FastAPI** and driven from a **Telegram bot**.
> Uses a **CRAG** (Corrective RAG) pattern over `pgvector` to cache past research and avoid redundant web searches.

![python](https://img.shields.io/badge/python-3.14-blue)
![langgraph](https://img.shields.io/badge/LangGraph-async-green)
![fastapi](https://img.shields.io/badge/FastAPI-SSE-009688)
![postgres](https://img.shields.io/badge/Postgres-pgvector-336791)

## What it does

You send a topic → the planner drafts a research outline → you approve it (human-in-the-loop) → agents retrieve from a `pgvector` cache, fall back to web search when cache hits are weak, write the report section-by-section, self-review, and emit a polished Markdown document with inline citations, a knowledge-graph snapshot, and token-cost metrics.

The whole run is checkpointed in Postgres, so a crash mid-pipeline resumes where it left off, and revisions can re-enter the graph without losing prior state.

## Architecture

### Graph topology

```mermaid
flowchart TD
    START([START]) --> planner
    planner --> await_approval{{await_approval<br/>interrupt_before — human approves plan}}
    await_approval --> researcher[researcher<br/>pgvector cache lookup]
    researcher --> grader{grader<br/>LLM validates cache hits}
    grader -- all relevant --> writer
    grader -- mixed quality --> adaptive_retrieval[adaptive_retrieval<br/>query rewrite + retry]
    grader -- all irrelevant --> web_search[web_search<br/>Tavily]
    adaptive_retrieval -- still weak --> web_search
    adaptive_retrieval -- recovered --> writer
    web_search --> writer
    writer --> reviewer{reviewer}
    reviewer -- needs revision<br/>max 2x --> researcher
    reviewer -- approved --> formatter
    formatter --> citations[citations<br/>claim → source attribution]
    citations --> knowledge_graph[knowledge_graph<br/>entity + relation extraction]
    knowledge_graph --> END([END])
```

ASCII fallback:

```
START
 └─> planner
       └─[interrupt_before await_approval — human approves plan]
             └─> researcher  (pgvector cache lookup)
                   └─> grader  (LLM validates cache-hit quality)
                         ├─(all relevant)──────> writer
                         ├─(mixed)─> adaptive_retrieval ─┬─> writer
                         │                                └─> web_search ─> writer
                         └─(all irrelevant)──> web_search ─> writer
                                                              └─> reviewer
                                                                    ├─(needs revision, max 2x)─> researcher
                                                                    └─(approved)──────────────> formatter
                                                                                                     └─> citations
                                                                                                            └─> knowledge_graph
                                                                                                                   └─> END
```

### Components

| Layer | Tech |
|---|---|
| Graph | LangGraph (`AsyncPostgresSaver` + `psycopg_pool.AsyncConnectionPool`) |
| LLM | OpenAI (any model; per-model pricing table) + `with_structured_output` |
| Search | Tavily with tenacity retry |
| Vector cache | Postgres 17 + `pgvector` (cosine similarity, TTL filter, dim validation) |
| Rate limit | Redis sliding-window (Lua script, distributed across replicas) |
| API | FastAPI + SSE (`sse-starlette`) — per-node boundaries AND LLM token streaming |
| Bot | aiogram 3 FSM, talks to API over HTTP |
| Migrations | Alembic |
| Observability | structlog (JSON), Sentry, Prometheus + provisioned Grafana dashboard |
| Quality gate | LLM-as-judge evaluation harness (`scripts/eval.py`) |
| Tests | pytest + httpx + asgi-lifespan + testcontainers (real pgvector) — 75% cov |
| CI | GitHub Actions (lint/type/test/docker-build + gitleaks + pip-audit) |
| Deploy | `fly.toml` for FastAPI service + `fly.bot.toml` for the bot worker |

## Quick start

```bash
cp .env.example .env   # fill in OPENAI_API_KEY, TAVILY_API_KEY, TELEGRAM_BOT_TOKEN

docker compose up -d
# app        → http://localhost:8000  (+ /docs, /metrics, /health/ready)
# grafana    → http://localhost:3000  (admin/admin, dashboard "Research Assistant")
# prometheus → http://localhost:9090
```

`docker compose` brings up: Postgres (pgvector), a one-shot Alembic migrator, the FastAPI app, the Telegram bot, Prometheus, and Grafana with a provisioned dashboard.

## API

```bash
# 1. Start a session — planner runs and pauses before the interrupt
curl -X POST http://localhost:8000/research/start \
    -H "X-API-Key: $RESEARCH_API_KEY" \
    -d '{"topic": "LangGraph vs CrewAI in 2026"}'

# 2. Review the generated plan
curl http://localhost:8000/research/<thread_id>/plan -H "X-API-Key: ..."

# 3. Approve (optionally with edited plan)
curl -X POST http://localhost:8000/research/<thread_id>/approve \
    -H "X-API-Key: ..." -d '{}'

# 4. Drive the graph + stream live node updates via SSE
curl -N http://localhost:8000/research/<thread_id>/stream -H "X-API-Key: ..."

# 5. Fetch the final markdown report + metrics
curl http://localhost:8000/research/<thread_id>/result -H "X-API-Key: ..."
curl http://localhost:8000/research/<thread_id>/metrics -H "X-API-Key: ..."
```

Full endpoint reference: [docs/api.md](docs/api.md).

## LangChain surface used

Pretty much the full LangChain stack sits under the LangGraph orchestration layer:

| Component | Where |
|---|---|
| **LCEL chains** (`prompt \| llm \| parser`) | every agent node — `app/graph/nodes.py` |
| **`with_structured_output`** → Pydantic | `ResearchPlan`, `GradeVerdict`, `ReviewVerdict` for fragile-parse-free decisions |
| **`ChatOpenAI`** | `langchain-openai` LLM wrapper, retried via tenacity |
| **`OpenAIEmbeddings`** | `langchain-openai` embedding model for pgvector cache |
| **`BaseRetriever`** | custom `VectorCacheRetriever` in `app/tools/retriever.py` — standard Runnable, composable into any LCEL chain |
| **`RecursiveCharacterTextSplitter`** | `app/tools/splitter.py` — chunks Tavily output before embedding so retrieval matches facts, not blurred summaries |
| **LangChain Runnable `.abatch()`** | researcher + web_search fan out queries and summariser calls in parallel with built-in concurrency throttling |
| **`BaseCallbackHandler`** | custom `UsageCallback` — token + per-model cost accounting, also pushed to Prometheus |
| **`astream_events(version="v2")`** | `app/api/routes.py::/stream` — token-level SSE filtered by Runnable tags (`writer`, `formatter`) |
| **`with_config(tags=[...])`** | writer/formatter chains tagged so only their tokens are streamed to clients |
| **`langchain-tavily`** | Tavily search tool via LangChain community |

## MCP Integration

The project ships with a **Model Context Protocol** server that exposes
this codebase's tools to any MCP-compatible client (Claude Desktop, Cursor,
LangGraph agents, custom Anthropic SDK loops). Other agents can reuse our
retrieval + quality tools without linking against our code.

### Tools exposed

| Tool | Purpose |
|---|---|
| `search_cache(query, top_k, threshold)` | pgvector semantic search (TTL-filtered) |
| `search_hybrid(query, top_k)` | BM25 + vector fused via Reciprocal Rank Fusion |
| `search_bm25(query, top_k)` | Postgres `ts_rank_cd` lexical search |
| `save_to_cache(topic, section, content)` | Embed + persist a chunk |
| `search_web(query, max_results)` | Live Tavily web search |
| `rerank(query, documents, top_n)` | LLM cross-encoder reranker |
| `extract_claims(report)` | Structured claim extraction for hallucination audits |
| `attribute_claims(claims, sources, threshold)` | Embedding-based claim → source matching |

### Run locally

```bash
# stdio transport (Claude Desktop, LangGraph agent integrations)
uv run python -m app.mcp_server

# networked transports
uv run python -m app.mcp_server --transport sse --port 8765
uv run python -m app.mcp_server --transport streamable-http --port 8765
```

### Client configuration — Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "research-assistant": {
      "command": "uv",
      "args": ["run", "python", "-m", "app.mcp_server"],
      "cwd": "/absolute/path/to/research_assistant",
      "env": {
        "OPENAI_API_KEY": "...",
        "TAVILY_API_KEY": "...",
        "DATABASE_URL": "postgresql+psycopg://research:research@localhost:5432/research_db"
      }
    }
  }
}
```

### Client configuration — LangGraph / LangChain

Use `langchain-mcp-adapters` to pull MCP tools into any LangGraph agent:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "research-assistant": {
            "command": "uv",
            "args": ["run", "python", "-m", "app.mcp_server"],
            "transport": "stdio",
        }
    }
)
tools = await client.get_tools()  # now usable with any LangGraph agent
```

### Why this matters

- The tools (`retriever`, `reranker`, `attribution`, `claim extractor`) are
  genuinely reusable — exposing them via MCP means a separate QA agent, a
  citation-verifier bot, or a follow-up-research planner can all share
  one pgvector cache and one embedding pool.
- Separating the *tools* (MCP server) from the *agent* (LangGraph graph)
  pushes toward the same interoperability story the agentic-systems
  ecosystem is converging on — agents compose tools, tools don't assume
  a single agent.

## Technical highlights (what to talk about)

- **LangGraph `StateGraph` with a dedicated `await_approval` node** so `interrupt_before` fires exactly once after the planner; revision cycles loop back to `researcher` and skip approval.
- **CRAG**: researcher checks pgvector first, LLM grader (temperature=0.0) validates the quality of cache hits, and the `web_search` node re-fetches only the sections graded irrelevant.
- **Structured output** (Pydantic schemas `ResearchPlan` / `GradeVerdict` / `ReviewVerdict`) on every decision node — no string parsing, no regex.
- **Async-first**: all LLM calls, embeddings, DB access, and the checkpointer are async; `asyncio.gather` parallelises per-section work in researcher / grader / writer.
- **Reliability**: tenacity retry with exponential backoff on OpenAI/Tavily transient errors, `asyncio.timeout` around every graph invocation, `asyncio.Semaphore` to cap concurrent embedding calls, `return_exceptions=True` so one failing section doesn't kill the rest.
- **Security**: SHA-256 access control (sessions tagged with the hash of the creating API key — cross-user access returns 403), prompt-injection hardening (all user text wrapped in XML tags with system prompts instructing the model to treat tag contents as data), sliding-window rate limit with periodic GC, `bindparam(type_=Vector)` so embeddings are never string-interpolated into SQL.
- **Observability**: structlog JSON logs with per-request `X-Request-ID`, Sentry (FastAPI/SQLAlchemy integrations), Prometheus custom metrics (`graph_node_duration_seconds`, `cache_hit_total`, `cache_miss_total`, `llm_tokens_total`, `llm_cost_usd_total`) with a provisioned Grafana dashboard.
- **Distributed rate limiting**: Redis-backed sliding-window via a single Lua script — survives restarts and scales across replicas. Graceful fallback to in-process limiter if `REDIS_URL` is unset.
- **Per-node hard timeouts**: each graph node (planner, researcher, grader, writer, …) has its own `asyncio.timeout` budget. A stuck LLM call on one phase can't starve others.
- **Per-model cost table** (`app/pricing.py`): swap `OPENAI_MODEL` in `.env` and USD cost calculation updates automatically — no silent billing drift.
- **LLM streaming to the client**: `astream_events(version="v2")` on the graph emits `token` events over SSE for writer/formatter chains, so clients can render the report as it's being written.
- **LLM-as-judge regression testing** (`scripts/eval.py`): runs a fixture of 10 topics end-to-end, scores each report across 4 rubrics with a separate deterministic judge model, and exits non-zero if average relevance drops below the quality floor. Wire it into CI to block PRs that regress LLM quality.
- **Real-DB integration tests** (`tests/integration/test_real_db.py`): spin up `pgvector/pgvector:pg17` via testcontainers, run Alembic to head, exercise the retriever + ORM against real schema — catches migration bugs and embedding-dim drift that mocked tests would miss.
- **Defence in depth**: body-size middleware (1 MiB cap), embedding dimension validation at startup, SHA-256 access control, prompt-injection XML wrapping, SQL bindparam.
- **Tests**: 94 fast tests + 3 testcontainers real-DB tests — 76% line coverage on `app/`.
- **MCP server** exposes `search_cache`, `search_hybrid`, `search_web`, `rerank`, `extract_claims`, `attribute_claims` as reusable tools for any MCP-compatible agent (Claude Desktop, Cursor, LangGraph agents). See the "MCP Integration" section.

## Project layout

```
research_assistant/
├── app/
│   ├── main.py            # FastAPI + lifespan (logging, sentry, prometheus, graph warm-up)
│   ├── config.py          # pydantic-settings
│   ├── errors.py          # typed HTTPException subclasses
│   ├── logging_config.py  # structlog JSON/human switch
│   ├── middleware.py      # RequestIDMiddleware
│   ├── observability.py   # Sentry + Prometheus custom metrics
│   ├── api/routes.py      # /research endpoints
│   ├── graph/
│   │   ├── state.py       # ResearchState (TypedDict + reducers)
│   │   ├── nodes.py       # planner, researcher, grader, web_search, writer, reviewer, formatter
│   │   ├── edges.py       # route_after_grader, route_after_reviewer
│   │   ├── graph.py       # StateGraph assembly + AsyncPostgresSaver pool
│   │   └── callbacks.py   # UsageCallback (tokens + cost → Prometheus)
│   ├── tools/
│   │   ├── retriever.py   # pgvector cache (TTL-filtered similarity + bindparam)
│   │   └── search.py      # Tavily wrapper
│   └── models/            # SQLAlchemy ORM + engine
├── bot/bot.py             # aiogram bot — HTTP client to FastAPI
├── migrations/            # Alembic
├── monitoring/            # prometheus.yml + Grafana provisioning + dashboard JSON
├── tests/                 # integration/ + graph/ + edges/schemas unit tests
├── docs/                  # architecture.md, api.md
├── Dockerfile             # multi-stage (python:3.14-slim + uv)
├── docker-compose.yml     # postgres + migrator + app + bot + prometheus + grafana
└── pyproject.toml
```

## Running tests locally

```bash
uv sync

# fast unit + mocked-integration suite (default)
uv run pytest --cov=app --cov-fail-under=70

# heavy — spins up a real pgvector container via testcontainers (needs Docker)
uv run pytest -m real_db
```

## LLM quality evaluation

```bash
uv run python -m scripts.eval --out eval_report.md
# Runs 10 fixture topics end-to-end, scores each report with a judge LLM,
# writes a markdown table of relevance/depth/structure/factuality + cost.
# Non-zero exit if average relevance < 3.5 — wire into CI to block regressions.
```

## Deploying to Fly.io

```bash
# Prerequisites: flyctl installed, authenticated (`flyctl auth login`)
flyctl launch --no-deploy                       # first-time setup
flyctl postgres create --name research-pg       # pgvector-capable
flyctl postgres attach research-pg
flyctl redis create --name research-redis       # distributed rate limit
flyctl secrets set OPENAI_API_KEY=... TAVILY_API_KEY=... \
                   RESEARCH_API_KEY=... SENTRY_DSN=...
./scripts/deploy.sh                              # deploys app + bot
```

## Further reading

- [docs/architecture.md](docs/architecture.md) — full graph topology and key decisions
- [docs/api.md](docs/api.md) — API reference with curl examples
