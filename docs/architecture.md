# Architecture

## Graph topology

```
START
 └─> planner                    structured output → ResearchPlan
       │
       └─[interrupt_before await_approval]  ← fires exactly once
             │
             └─> researcher    pgvector cache hit → "relevant"
                   │          cache miss → Tavily + save + "irrelevant"
                   │          (retry + bounded concurrency on embeddings)
                   │
                   └─> grader              LLM (temperature=0) re-verifies hits
                         │                 → {"relevant" | "irrelevant"} per section
                         │
                         ├─(all relevant)────────────> writer
                         │
                         └─(any irrelevant)─> web_search  only re-fetches irrelevant
                                                   │      sections; merges results
                                                   │
                                                   └──> writer
                                                         │
                                                         └─> reviewer  structured verdict
                                                               │
                                                               ├─(approved)──────> formatter → END
                                                               └─(needs revision, count<max)─> researcher
```

### Why a dedicated `await_approval` node?

`interrupt_before=["researcher"]` would fire on **every** revision cycle, which is wrong — the user only needs to approve the plan once. Putting the interrupt on a no-op `await_approval` node that `planner → await_approval → researcher` flows through lets us interrupt exactly once. The `reviewer → researcher` back-edge skips `await_approval` entirely.

### Why a separate `grader` + `web_search` (CRAG)?

- **pgvector cache hits are fast but can be stale / off-topic.** A deterministic LLM grader (temperature=0) decides per-section whether to trust the cached content.
- **Cache misses came straight from Tavily** and are already fresh, so we label them `"irrelevant"` at retrieval time to skip the grader LLM call and route through `web_search` unconditionally (the web_search node does the summarisation and also saves to cache for next time).

### State & reducers

`ResearchState` is a `TypedDict` with `Annotated` reducers for the three dict fields that nodes contribute to in parallel:

```python
search_results:   Annotated[dict, lambda a,b: {**a, **b}]
sections:         Annotated[dict, lambda a,b: {**a, **b}]
retrieval_grades: Annotated[dict, lambda a,b: {**a, **b}]
```

This lets `asyncio.gather`'d node tasks each return a partial dict that LangGraph merges — no in-node locking required.

## Checkpointing

`AsyncPostgresSaver` backed by a shared `psycopg_pool.AsyncConnectionPool(min=2, max=20, autocommit=True, prepare_threshold=0)`. The pool is created lazily on first `get_graph()` call, warmed up in `lifespan`, and closed on shutdown. This gives the graph thread-level persistence (resumable from any `thread_id`) without the cost of opening a connection per node.

## Security

### Access control

Every `ResearchSession` is tagged with `api_key_hash = sha256(X-API-Key)` at creation. `_check_ownership` runs before every read/write on `{thread_id}/*` endpoints — a key that didn't create the session gets 403. Dev mode (empty `RESEARCH_API_KEY`) skips this check.

### Prompt injection

All user-controlled strings (`topic`, `section`, search results) are wrapped in XML tags inside prompts. System prompts instruct the model to treat tag contents as data only. Example:

```
<topic>LangGraph vs CrewAI</topic>
```

This doesn't make LLMs safe against motivated attackers, but stops the trivial "Ignore previous instructions" class of attack and raises the bar.

### SQL

pgvector queries use `sqlalchemy.bindparam(name, type_=Vector(1536))`, so the embedding array is bound as a typed parameter — never string-interpolated.

## Reliability patterns

| Concern | Mechanism |
|---|---|
| OpenAI 429 / 5xx / timeout | `tenacity` retry with exponential backoff, max 3 attempts |
| Tavily transient failures | same retry policy |
| Stuck graph | `asyncio.timeout(settings.graph_timeout_seconds)` (default 300s) wrapping every `ainvoke` / `astream` |
| Slow embedding concurrency | `asyncio.Semaphore(settings.embedding_max_concurrency)` shared by retriever calls |
| One bad section | `asyncio.gather(..., return_exceptions=True)` + structured log + skip |
| Rate-bucket memory leak | periodic cleanup task started in `lifespan` |
| Connection pool lifecycle | `create_postgres_checkpointer` opens; `close_pool` in lifespan `finally` closes |

## Observability

- **Logs**: `structlog` with JSON renderer in prod, colourful console in dev. Every log carries `request_id` (from `RequestIDMiddleware`), and graph nodes log `node_start` / `node_end` with duration.
- **Tracing**: Sentry (FastAPI + SQLAlchemy integrations). LangSmith tracing is activated automatically when `LANGSMITH_API_KEY` is set.
- **Metrics**: Prometheus instrumentator exposes default HTTP metrics on `/metrics`. Custom metrics:
  - `graph_node_duration_seconds{node}` — histogram
  - `graph_nodes_total{node,status}` — counter
  - `cache_hit_total` / `cache_miss_total` — counter
  - `llm_tokens_total{model,type}` — counter
  - `llm_cost_usd_total{model}` — counter
- **Dashboard**: `monitoring/grafana/dashboards/research-assistant.json` is auto-provisioned into Grafana. Panels: request rate, error rate, graph node latency P50/P95/P99, cache hit rate, LLM cost/hr, tokens/sec, node errors.
