# API Reference

All `/research/*` endpoints require the `X-API-Key` header when `RESEARCH_API_KEY` is set. In dev mode (empty env var) the header is optional.

Base URL: `http://localhost:8000` (default).

---

## `POST /research/start`

Kick off a new research session. Runs the planner, pauses at the `await_approval` interrupt.

**Request**

```json
{"topic": "LangGraph vs CrewAI in 2026"}
```

- `topic`: 3-500 chars. Longer topics → 422.

**Response 200**

```json
{
  "thread_id": "8e5a6df2-3a4f-4d6e-9b77-a4a5f25a4a8e",
  "message": "Plan generated. GET /research/{thread_id}/plan to review."
}
```

**Errors**

- 401 — missing/wrong API key
- 422 — invalid `topic`
- 429 — rate limit hit (see `Retry-After` header)
- 504 — planner timed out

---

## `GET /research/{thread_id}/plan`

Fetch the generated plan for review.

**Response 200**

```json
{
  "thread_id": "...",
  "plan": ["Market landscape", "Feature comparison", "Pricing", "Community"],
  "status": "waiting_approval"
}
```

**Errors**

- 400 — malformed `thread_id`
- 403 — session belongs to another API key
- 404 — session or plan missing

---

## `POST /research/{thread_id}/approve`

Mark the plan approved. **Does not run the graph** — you must connect to `/stream` next.

**Request** (either is fine)

```json
{}
```

or with an edited plan:

```json
{
  "plan": ["New section 1", "New section 2"]
}
```

- `plan`: optional, max 10 items, each max 200 chars.

**Response 200**

```json
{
  "thread_id": "...",
  "message": "Plan approved. GET /research/{thread_id}/stream to run it."
}
```

---

## `GET /research/{thread_id}/stream`

Drives the graph to completion and streams live node updates via **Server-Sent Events**.

If the session is already complete, replays checkpoint history for late subscribers.

**Events**

```
event: progress
data: {"node":"researcher","plan":["..."],"sections_written":[],"revision_count":0,"review_feedback":""}

event: progress
data: {"node":"grader",...}

event: done
data: {"final_report_ready":true}
```

On failure:

```
event: error
data: {"message":"Graph execution timed out"}
```

**Errors**

- 409 — approve the plan first
- 403 — other API key's session

---

## `GET /research/{thread_id}/result`

Return the final markdown report.

**Response 200**

```json
{
  "thread_id": "...",
  "final_report": "# Research Report: ...\n\n## Executive summary\n..."
}
```

**Errors**

- 202 — report not ready yet (check `/stream`)

---

## `GET /research/{thread_id}/metrics`

Token usage + cost for the session.

**Response 200**

```json
{
  "thread_id": "...",
  "prompt_tokens": 12345,
  "completion_tokens": 6789,
  "total_tokens": 19134,
  "cost_usd": 0.00423,
  "status": "done"
}
```

---

## Infrastructure endpoints

- `GET /health` — liveness probe (always 200)
- `GET /health/ready` — readiness; returns `db:"ok"` iff `SELECT 1` succeeds
- `GET /metrics` — Prometheus exposition format
- `GET /docs` — OpenAPI / Swagger UI
