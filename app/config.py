from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Works whether uvicorn is launched from project root or a subdirectory
_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"  # default for any role without override
    llm_temperature: float = 0.3
    llm_temperature_deterministic: float = 0.0

    # Per-node model overrides. Empty string = use `openai_model`.
    # Reasoning-heavy roles get a stronger model; cheap roles stay small.
    model_planner: str = ""  # outline generation — reasoning
    model_researcher: str = ""  # summarising search results
    model_grader: str = ""  # deterministic yes/no verdicts
    model_writer: str = ""  # creative prose
    model_reviewer: str = ""  # nuanced judgement
    model_formatter: str = ""  # mechanical assembly
    model_judge: str = ""  # eval harness judge model

    # Anthropic fallback — used when OpenAI is rate-limited / down.
    # Leave anthropic_api_key empty to disable fallback.
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    tavily_api_key: str = ""
    tavily_max_results: int = 3

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    database_url: str = "postgresql+psycopg://research:research@localhost:5432/research_db"

    # ------------------------------------------------------------------
    # App
    # ------------------------------------------------------------------
    debug: bool = False
    max_revision_count: int = 2
    graph_timeout_seconds: int = 300
    # Hard ceiling on total tokens consumed per session (H-4). When exceeded,
    # the revision loop exits early even if the reviewer wants another pass,
    # preventing cost-amplification DoS via adversarial plans.
    max_tokens_per_session: int = 150_000
    # Cap on final_report length (L-2) — formatter truncates beyond this.
    max_report_length: int = 50_000
    # Idle-event timeout on SSE /stream (H-6). If no event fires for this many
    # seconds, close the connection to prevent slow-loris attacks.
    sse_idle_timeout_seconds: int = 120

    # Per-node hard timeouts (seconds). A node taking longer than its budget
    # is cancelled and its failure surfaced through the graph error counter.
    planner_timeout_seconds: int = 60
    researcher_timeout_seconds: int = 150
    grader_timeout_seconds: int = 60
    web_search_timeout_seconds: int = 150
    writer_timeout_seconds: int = 120
    reviewer_timeout_seconds: int = 60
    formatter_timeout_seconds: int = 60
    citations_timeout_seconds: int = 90
    refine_timeout_seconds: int = 120

    # Adaptive retrieval depth (G-8). If the grader says "irrelevant" for a
    # section, re-retrieve with top_k doubled, up to this many extra passes.
    max_retrieval_depth: int = 2

    # ------------------------------------------------------------------
    # Retrieval cache (CRAG)
    # ------------------------------------------------------------------
    similarity_threshold: float = 0.75
    cache_ttl_days: int = 30
    retriever_top_k: int = 3
    embedding_max_concurrency: int = 5
    # Text-splitter parameters — see app/tools/splitter.py
    chunk_size: int = 800
    chunk_overlap: int = 100

    # Semantic full-report cache — see app/cache/semantic.py
    # Threshold is intentionally stricter than the chunk-level similarity_threshold:
    # we only skip the graph entirely if the topic embeddings are near-identical.
    semantic_cache_threshold: float = 0.95
    semantic_cache_ttl_days: int = 7

    # In-flight request dedup — see app/cache/dedup.py (G-4).
    dedup_window_seconds: int = 30
    dedup_similarity_threshold: float = 0.95

    # Reranker backend — "llm" (default, zero extra deps) or "cross_encoder"
    # (needs sentence-transformers; ~100-1000× cheaper/faster at top-10).
    reranker_backend: str = "llm"
    # Only used when reranker_backend == "cross_encoder".
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # HyDE (G-10): expand query with a hypothetical answer before embedding.
    # Disabled by default because it costs one extra LLM call per section.
    hyde_enabled: bool = False
    # Blend ratio between query and hypothetical embeddings (0.0 = pure query,
    # 1.0 = pure hypothetical). 0.3 is the paper's recommended default.
    hyde_blend: float = 0.3

    # Speculative research (G-14): kick off researcher in the background as
    # soon as the planner returns, under the assumption the user will approve.
    # On edit/reject we cancel the task. Trades a small amount of wasted
    # work on rejection for ~40% lower perceived latency on the approve path.
    speculative_execution_enabled: bool = True

    # PII + safety (G-15).
    pii_redaction_enabled: bool = False  # off by default — presidio is a heavy dep
    safety_classifier_enabled: bool = True

    # ------------------------------------------------------------------
    # API auth + rate-limit
    # ------------------------------------------------------------------
    research_api_key: str = ""
    # Separate key required for destructive cache-invalidation endpoints (RBAC).
    # When empty, DELETE /cache/* is disabled entirely — any cache clearing
    # must go through `alembic` / direct DB access. Set to a distinct secret
    # so leaking a regular research_api_key can't be used to wipe the cache.
    admin_api_key: str = ""
    rate_limit_per_minute: int = 10
    # When set, rate-limit state is shared across replicas via Redis.
    # Leave empty to use single-process in-memory limiter (dev / single pod).
    redis_url: str = ""
    # Max request body size (bytes) — 1 MiB default, rejects oversize POSTs.
    max_request_body_bytes: int = 1_048_576

    # ------------------------------------------------------------------
    # Telegram bot
    # ------------------------------------------------------------------
    telegram_bot_token: str = ""
    api_base_url: str = "http://127.0.0.1:8000"

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    log_format: str = "human"  # human | json
    log_level: str = "INFO"

    sentry_dsn: str = ""
    sentry_traces_sample_rate: float = 0.1
    sentry_environment: str = "dev"

    # Prometheus /metrics scrape token (M-1). When set, clients must pass it
    # as `Authorization: Bearer <token>`. Empty = open (dev / local compose).
    metrics_token: str = ""

    # LangSmith
    langsmith_tracing: bool = False
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: str = ""
    langsmith_project: str = "research-assistant"


settings = Settings()
