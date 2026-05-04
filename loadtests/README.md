# Load tests

Profile the Research Assistant API under concurrent user load.

## Prerequisites

```bash
docker compose up -d                # whole stack (api, bot, redis, postgres, prometheus, grafana)
# seed some cached documents first if you want cache-hit ratios to be meaningful
```

## Run

Interactive (web UI at http://localhost:8089):

```bash
uv run locust -f loadtests/locustfile.py --host http://localhost:8000
```

Headless, fixed load for 5 minutes:

```bash
uv run locust -f loadtests/locustfile.py \
    --host http://localhost:8000 \
    --users 5 --spawn-rate 1 --run-time 5m --headless --csv out
```

CSV outputs `out_stats.csv` / `out_failures.csv` — upload as CI artifact or
diff against a previous run to catch regressions.

## What to watch in Grafana during a load run

- **graph_node_duration_seconds** p95/p99 — should stay well under per-node
  timeouts (planner 60s, researcher 150s).
- **cache_hit_total / cache_miss_total** — hit ratio climbs as users repeat
  similar topics. If it stays 0%, the splitter / threshold is mis-tuned.
- **http_request_duration_seconds{handler="/research/:id/stream"}** — tail
  latency of streaming. If it spikes, one of the graph nodes is hanging.
- **llm_cost_usd_total{model}** — sanity-check that per-node model routing
  is actually hitting cheaper models for cheap roles.
- **Postgres pool saturation** — `AsyncConnectionPool` is min=2, max=20. At
  ~50 concurrent graph invocations you'll see connections queue; that's the
  signal to bump `max_size` or shard.

## Cost warning

Every iteration hits real OpenAI + Tavily. Factor ~$0.01-0.05 per iteration.
Keep `--users` small unless you've pre-warmed the semantic cache with
`scripts/eval.py` runs.
