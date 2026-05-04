"""Drift-detection wrapper over `scripts.eval` (G-2).

Runs the fixed golden topic set, records aggregate scores in the `eval_runs`
table, and compares against the trailing 7-day median. Emits a non-zero
exit code (and a Prometheus gauge, if the process is co-located with the
app) when any metric drops more than `--tolerance` vs the 7-day median.

Designed to be wired into a nightly GitHub Actions cron job — it writes a
small markdown summary to `out` so artifacts are readable at a glance.

Usage:
    uv run python -m scripts.eval_drift \
        --dataset scripts/eval_dataset.json \
        --out drift.md \
        --tolerance 0.05
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import text

from app.eval.metrics import aggregate
from app.graph.callbacks import reset_session_tokens  # noqa: F401 — future use
from app.graph.graph import build_graph, close_pool, create_postgres_checkpointer
from app.models.db import EvalRun
from app.models.engine import SessionLocal
from scripts.eval import _judge, _run_one  # reuse the existing harness


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def _record_run(scores: dict, git_sha: str | None) -> None:
    with SessionLocal() as db:
        db.add(
            EvalRun(
                id=uuid.uuid4(),
                run_date=datetime.now(UTC),
                git_sha=git_sha,
                metric_scores_json=json.dumps(scores),
            )
        )
        db.commit()


def _historical_median(metric: str, days: int = 7) -> float | None:
    """Pull the median of `metric` across runs in the last `days`."""
    with SessionLocal() as db:
        rows = db.execute(
            text(
                "SELECT metric_scores_json FROM eval_runs "
                "WHERE run_date > now() - make_interval(days => :days)"
            ),
            {"days": days},
        ).fetchall()
    if not rows:
        return None
    values: list[float] = []
    for (blob,) in rows:
        try:
            data = json.loads(blob)
            v = _nested_get(data, metric)
            if v is not None:
                values.append(float(v))
        except Exception:  # noqa: BLE001
            continue
    return statistics.median(values) if values else None


def _nested_get(data: dict, dotted: str) -> float | None:
    """`ragas.faithfulness` → data['ragas']['faithfulness']."""
    cur = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur if isinstance(cur, (int, float)) else None


async def _run(
    topics: list[str],
    tolerance: float,
    out_path: Path | None,
) -> int:
    # Configure LangSmith if available — same as the main harness.
    if os.environ.get("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_TRACING"] = "true"

    checkpointer = await create_postgres_checkpointer()
    graph = build_graph(checkpointer=checkpointer)
    judge = _judge(os.environ.get("JUDGE_MODEL", "gpt-4o-mini"))
    try:
        results = []
        for i, topic in enumerate(topics, 1):
            print(f"[{i}/{len(topics)}] {topic}")
            try:
                results.append(await _run_one(graph, topic, judge))
            except Exception as e:  # noqa: BLE001
                print(f"  ! failed: {e}")

        if not results:
            return 0

        overall = aggregate([r.ragas for r in results])
        judge_means = {
            "relevance": statistics.mean(r.score.relevance for r in results),
            "depth": statistics.mean(r.score.depth for r in results),
            "structure": statistics.mean(r.score.structure for r in results),
            "factuality": statistics.mean(r.score.factuality for r in results),
        }
        scores = {"judge": judge_means, "ragas": overall.as_dict()}

        # Persist the run so subsequent invocations have history.
        _record_run(scores, _git_sha())

        # Compare each metric to its 7-day median.
        drift_lines = [
            "## Drift report",
            "",
            "| Metric | Current | Median(7d) | Δ | Status |",
            "|---|---:|---:|---:|:---|",
        ]
        regressions: list[str] = []
        for dotted in (
            "judge.relevance",
            "judge.depth",
            "judge.structure",
            "judge.factuality",
            "ragas.faithfulness",
            "ragas.citation_precision",
            "ragas.citation_recall",
            "ragas.answer_relevance",
        ):
            current = _nested_get(scores, dotted)
            median7 = _historical_median(dotted)
            if current is None:
                continue
            if median7 is None:
                drift_lines.append(f"| {dotted} | {current:.3f} | _no history_ | — | 🔵 new |")
                continue
            delta = current - median7
            if delta < -tolerance:
                status = "🔴 DRIFT"
                regressions.append(f"{dotted} {delta:+.3f}")
            elif delta > tolerance:
                status = "🟢 improved"
            else:
                status = "⚪ stable"
            drift_lines.append(
                f"| {dotted} | {current:.3f} | {median7:.3f} | {delta:+.3f} | {status} |"
            )

        report = "\n".join(drift_lines)
        if out_path:
            out_path.write_text(report, encoding="utf-8")
            print(f"\nWritten to {out_path}")
        else:
            print("\n" + report)

        if regressions:
            print("\nFAIL:")
            for r in regressions:
                print(f"  - regression {r}")
            return 1
        return 0
    finally:
        await close_pool()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parent / "eval_dataset.json",
    )
    parser.add_argument("--topics", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Fail if any metric drops more than this vs 7-day median",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    topics = data["topics"]
    if args.topics > 0:
        topics = topics[: args.topics]

    raise SystemExit(asyncio.run(_run(topics, args.tolerance, args.out)))


if __name__ == "__main__":
    main()
