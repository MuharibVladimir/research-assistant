"""Load-test profile for the Research Assistant API.

Runs synthetic users that mimic the real UX:
  /start → /plan → /approve → /stream (drain SSE) → /result → /metrics

This is NOT a correctness test (that's in tests/) — it profiles:
  * where latency goes (/start vs /stream vs /followup)
  * 95/99 percentile tail behaviour under concurrent load
  * whether the Postgres connection pool saturates (max_size=20)
  * whether the per-key rate limit (10/min default) actually fires
  * whether the graph timeout catches long LLM tail latency

How to run locally (requires the stack up via `docker compose up -d`):

    cd research_assistant
    uv run locust -f loadtests/locustfile.py \
        --host http://localhost:8000 \
        --users 5 --spawn-rate 1 --run-time 5m --headless

For CI (just smoke — 2 users for 30s):

    uv run locust -f loadtests/locustfile.py \
        --host http://localhost:8000 \
        --users 2 --spawn-rate 1 --run-time 30s --headless --only-summary

Environment variables:
    LOAD_TEST_API_KEY — X-API-Key header if the server has RESEARCH_API_KEY set.

Cost note: each iteration hits real OpenAI + Tavily. Factor ~$0.01-0.05 per
iteration depending on plan length. Keep --users low unless you have a
sandbox OpenAI key.
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, between, events, task

TOPICS = [
    "LangGraph vs CrewAI in 2026",
    "Corrective RAG (CRAG) design patterns",
    "pgvector vs Pinecone at production scale",
    "Streaming token UX patterns for LLM apps",
    "Multi-agent coordination with supervisor graphs",
    "Structured output vs function calling tradeoffs",
    "Prompt injection defences in agentic systems",
]

FOLLOWUPS = [
    "Can you deepen the cost comparison?",
    "Add a section on open-source alternatives.",
    "Summarise the key tradeoffs in 3 bullet points.",
]


@events.init.add_listener
def _on_init(environment, **kw):  # noqa: ARG001
    print("Load-test starting. API key configured:", bool(os.environ.get("LOAD_TEST_API_KEY")))


class ResearchUser(HttpUser):
    """Simulates one end-user running a research session + one follow-up."""

    # Real users don't fire back-to-back requests — realistic think-time.
    wait_time = between(2, 5)

    def on_start(self) -> None:
        key = os.environ.get("LOAD_TEST_API_KEY")
        if key:
            self.client.headers.update({"X-API-Key": key})

    @task(4)
    def full_flow(self) -> None:
        topic = random.choice(TOPICS)
        with self.client.post(
            "/research/start",
            json={"topic": topic},
            name="/research/start",
            catch_response=True,
        ) as start:
            if start.status_code != 200:
                start.failure(f"start failed: {start.status_code} {start.text[:200]}")
                return
            body = start.json()
            tid = body["thread_id"]
            if body.get("cached"):
                start.success()
                return  # short-circuit — cached path, nothing more to do

        # Review plan
        self.client.get(f"/research/{tid}/plan", name="/research/:id/plan")

        # Approve
        self.client.post(f"/research/{tid}/approve", json={}, name="/research/:id/approve")

        # Drive the graph via SSE; drain the stream so connections close cleanly.
        with self.client.get(
            f"/research/{tid}/stream",
            name="/research/:id/stream",
            stream=True,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"stream {resp.status_code}")
                return
            for _ in resp.iter_lines():
                pass
            resp.success()

        # Read final report + metrics
        self.client.get(f"/research/{tid}/result", name="/research/:id/result")
        self.client.get(f"/research/{tid}/metrics", name="/research/:id/metrics")

    @task(1)
    def followup_on_fresh(self) -> None:
        """Kick off a session, drain it, then fire a follow-up turn."""
        topic = random.choice(TOPICS)
        r = self.client.post("/research/start", json={"topic": topic})
        if r.status_code != 200:
            return
        tid = r.json()["thread_id"]
        if r.json().get("cached"):
            # Cached → /followup works immediately on the cached report
            self.client.post(
                f"/research/{tid}/followup",
                json={"question": random.choice(FOLLOWUPS)},
                name="/research/:id/followup",
            )
            return
        self.client.post(f"/research/{tid}/approve", json={})
        with self.client.get(f"/research/{tid}/stream", stream=True) as resp:
            for _ in resp.iter_lines():
                pass
        self.client.post(
            f"/research/{tid}/followup",
            json={"question": random.choice(FOLLOWUPS)},
            name="/research/:id/followup",
        )

    @task(2)
    def healthcheck(self) -> None:
        """Cheap endpoint — simulates LB probes, should never degrade."""
        self.client.get("/health/ready", name="/health/ready")
