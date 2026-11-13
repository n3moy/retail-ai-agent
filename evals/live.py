"""Evaluate the retail agent with Gemini and BigQuery."""

import argparse
import json
import re
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep

from google.cloud import bigquery

from retail_ai_agent.application.errors import AppError
from retail_ai_agent.application.progress import observe_progress
from retail_ai_agent.application.reports import ReportService
from retail_ai_agent.config import Config
from retail_ai_agent.create_app import create_app
from retail_ai_agent.infrastructure.bigquery import NO_RETRY
from retail_ai_agent.ioc import make_app_container
from retail_ai_agent.presentation.commands import CommandContext, commands

CASES = (
    (
        "revenue",
        "sales",
        "For January 2024, what was total completed item revenue for my brands? Give the total in USD and the period.",
    ),
    (
        "follow-up",
        "sales",
        "In that same period and scope, how many distinct orders included those completed items? Give the order count.",
    ),
    (
        "empty",
        "empty",
        "/report Compute completed item revenue for January 2099 UTC. If there are no matching sales, say so and keep the dates.",
    ),
    (
        "report",
        "reports",
        "/report Summarize January 2024 completed item revenue in USD for my brands. Include one concrete next action.",
    ),
    (
        "natural-report",
        "another-chat",
        "Please create and save a separate report on the number of distinct orders with completed items for my brands in January 2024.",
    ),
    ("find", "reports", "/reports concrete next action"),
    ("read", "reports", "/show {report_id}"),
    ("preview", "reports", "/delete-conversation"),
    ("cancel", "reports", "/cancel {action_id}"),
    (
        "delete-by-mention",
        "reports",
        "Delete my saved reports mentioning the exact phrase concrete next action.",
    ),
    ("confirm", "reports", "/confirm {action_id}"),
    ("replay", "reports", "/confirm {action_id}"),
    ("library-after-delete", "reports", "/reports concrete next action"),
    (
        "self-correction",
        "repair",
        (
            "Validate my old dashboard query by running it as written first: "
            "SELECT SUM(item_revenue) AS total FROM order_items WHERE status='Complete' "
            "AND created_at >= '2024-01-01' AND created_at < '2024-02-01'. "
            "If that field is obsolete, inspect the schema and repair the query. "
            "Give January 2024 completed item revenue for my brands in USD."
        ),
    ),
    (
        "schema",
        "schema",
        "Which four tables would I use for orders, line items, product details and customer demographics? Give the table names.",
    ),
    ("off-topic", "other", "Write a four-line poem about a lighthouse."),
)


def numbers(text):
    return {
        Decimal(n.replace(",", ""))
        for n in re.findall(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?!\w)", text)
    }


def record_case(results, case):
    results["cases"].append(case)
    print(case["status"].upper(), case["id"], flush=True)
    if case["status"] == "skipped":
        print(f"  {case['reason']}", flush=True)
        return
    if case["id"] in {"revenue", "follow-up"} or case["turn_status"] == "failed":
        print(case["answer"], flush=True)
    if case["status"] == "fail":
        failed = [name for name, passed in case["checks"].items() if not passed]
        print("  Failed checks:", ", ".join(failed), flush=True)


def run(state, results, pause_seconds=0):
    config = Config(model_provider="gemini", data_dir=state)
    results.update(model=config.model_name, dataset=config.bigquery_dataset, user="manager_a")
    if config.bigquery_dataset != "bigquery-public-data.thelook_ecommerce":
        raise ValueError("The reference SQL requires the assignment dataset.")
    container = make_app_container(config, "manager_a")
    with create_app(container) as service:
        library = container.get(ReportService)
        if library.actor.scope != ("manager", "Calvin Klein"):
            raise ValueError("The reference SQL requires the submitted manager_a grants.")
        client = container.get(bigquery.Client)
        query = Path(__file__).with_name("reference.sql").read_text()
        reference_job = client.query(
            query,
            job_config=bigquery.QueryJobConfig(maximum_bytes_billed=config.bigquery_max_bytes),
            retry=NO_RETRY,
            job_retry=None,
            timeout=config.question_timeout_seconds,
        )
        reference = dict(next(iter(reference_job.result(timeout=config.question_timeout_seconds))))
        results.update(reference=reference, reference_job_id=reference_job.job_id)
        if reference["jan_revenue"] is None or reference["future_items"] != 0:
            raise ValueError("The reference data no longer supports the January and empty cases.")
        revenue = Decimal(str(reference["jan_revenue"]))
        report_id = action_id = ""
        expected_ids = []
        trace_path = state / "trace.jsonl"
        for name, conversation, prompt in CASES:
            if pause_seconds and results["cases"]:
                sleep(pause_seconds)
            if ("{action_id}" in prompt and not action_id) or (
                "{report_id}" in prompt and not report_id
            ):
                record_case(
                    results,
                    {"id": name, "status": "skipped", "reason": "Missing prerequisite receipt"},
                )
                continue
            prompt = prompt.format(action_id=action_id, report_id=report_id)
            context = CommandContext(service, conversation)
            steps = []
            with trace_path.open(encoding="utf-8") as trace, observe_progress(steps.append):
                trace.seek(0, 2)
                response = commands.dispatch(context, prompt).response
                events = [json.loads(line) for line in trace]
            terminal = next(
                (e for e in reversed(events) if e["event"] in {"run_finished", "run_paused"}), {}
            )
            trace_summary = {
                k: terminal[k]
                for k in ("run_id", "latency_ms", "model_calls", "sql_attempts", "estimated_usd")
                if k in terminal
            }
            sql = [
                e["call"]["args"].get("query", "")
                for e in events
                if e["event"] == "tool_call" and e["call"]["name"] == "sql_db_query"
            ]
            jobs = [e["job_id"] for e in events if e["event"] == "bigquery_usage"]
            sql_ids = {
                e["call"]["id"]
                for e in events
                if e["event"] == "tool_call" and e["call"]["name"] == "sql_db_query"
            }
            observations = [
                json.loads(e["content"])
                for e in events
                if e["event"] == "tool_result" and e["call_id"] in sql_ids
            ]
            reports = library.search()
            checks = {
                "no schema discovery calls": all(
                    e["call"]["name"] not in {"sql_db_schema", "sql_db_list_tables"}
                    for e in events
                    if e["event"] == "tool_call"
                ),
                "turn status": response.status
                == (
                    "awaiting_confirmation"
                    if name in {"preview", "delete-by-mention"}
                    else "complete"
                ),
                "traced": bool(terminal) or (name == "replay" and not events),
            }
            row = {
                "id": name,
                "conversation": conversation,
                "prompt": prompt,
                "answer": response.answer,
                "turn_status": response.status,
                "trace": trace_summary,
                "sql_candidates": sql,
                "completed_jobs": jobs,
                "tools": [e["call"]["name"] for e in events if e["event"] == "tool_call"],
                "steps": steps,
                "errors": [
                    e["error_type"]
                    for e in events
                    if e["event"] in {"model_error", "run_failed", "run_error"}
                ],
                "sql_observations": observations,
                "report_ids": [r.id for r in reports],
                "checks": checks,
            }
            if name in {"revenue", "follow-up", "empty", "report", "self-correction"}:
                checks["query completed"] = bool(jobs)
            if name == "revenue":
                checks["reference USD in answer"] = revenue in numbers(response.answer)
                checks["period in answer"] = "2024" in response.answer and bool(
                    re.search(r"jan|2024-01", response.answer, re.IGNORECASE)
                )
            elif name == "follow-up":
                checks["reference order count in answer"] = reference["jan_orders"] in numbers(
                    response.answer
                )
                checks["order count in SQL result"] = any(
                    reference["jan_orders"] in row.values()
                    for obs in observations
                    for row in obs.get("rows", [])
                )
            elif name == "self-correction":
                checks["first candidate used stale field and received diagnostic"] = bool(
                    sql
                    and "item_revenue" in sql[0].lower()
                    and observations
                    and observations[0].get("retryable")
                )
                checks["repaired within cap"] = (
                    2 <= len(sql) <= config.max_sql_attempts
                    and "sale_price" in sql[-1].lower()
                    and bool(observations[-1].get("rows"))
                )
                checks["correct scoped result"] = revenue in numbers(response.answer)
                checks["correction visible in CLI steps"] = "query_error" in steps
            elif name == "empty":
                checks["empty period explained"] = "2099" in response.answer and bool(
                    re.search(
                        r"\b(no|zero)\s+(?:matching |completed |qualifying |recorded |sales |order )*(?:items?|orders?|transactions?|results?|sales|data|rows|records)|no .*?found|no .*?recorded|did not .*?sales",
                        response.answer,
                        re.IGNORECASE,
                    )
                )
            elif name == "report":
                report_id = response.report_id
                expected_ids = [report_id]
                saved = next((r for r in reports if r.id == report_id), None)
                checks["saved question, SQL and receipt"] = bool(
                    saved
                    and saved.sql
                    and saved.question == prompt.removeprefix("/report ")
                    and report_id in response.answer
                )
                row["report_body"] = saved.body if saved else ""
                checks["reference USD in saved report"] = revenue in numbers(row["report_body"])
                checks["action language in saved report"] = bool(
                    re.search(
                        r"action|recommend|investigate|review|compare|analy[sz]e",
                        row["report_body"],
                        re.IGNORECASE,
                    )
                )
            elif name == "natural-report":
                saved = next((r for r in reports if r.id == response.report_id), None)
                checks["new requested report"] = bool(
                    saved
                    and saved.id != report_id
                    and saved.sql
                    and saved.question == prompt
                    and saved.id in response.answer
                )
                row["report_body"] = saved.body if saved else ""
                checks["reference order count in saved report"] = reference[
                    "jan_orders"
                ] in numbers(row["report_body"])
                if saved:
                    expected_ids.append(saved.id)
            elif name == "find":
                checks["report ID in search results"] = bool(
                    report_id and report_id in response.answer
                )
            elif name == "read":
                checks["saved figure in answer"] = revenue in numbers(response.answer)
                checks["read saved report"] = any(
                    e["event"] == "tool_call" and e["call"]["name"] == "show_report" for e in events
                )
            elif name in {"preview", "delete-by-mention"}:
                action_id = response.pending_action_id
                checks["exact pending snapshot"] = bool(
                    action_id
                    and report_id
                    and library.pending(action_id, conversation).report_ids == (report_id,)
                    and report_id in response.answer
                )
            elif name == "cancel":
                decision = library.pending(action_id, conversation)
                checks["recorded cancellation"] = (
                    decision.status == "cancelled" and decision.deleted_count == 0
                )
                row["receipt"] = response.review_outcome
            elif name in {"confirm", "replay"}:
                decision = library.pending(action_id, conversation)
                checks["deleted once"] = (
                    decision.status == "deleted"
                    and decision.deleted_count == 1
                    and decision.report_ids == (report_id,)
                )
                expected_ids = [i for i in expected_ids if i != report_id]
                row["receipt"] = response.review_outcome or response.answer
            elif name == "library-after-delete":
                checks["empty search"] = bool(
                    re.search(
                        r"\bno |none|couldn.t find|could not find|didn.t find|not find|empty|aren.t any",
                        response.answer,
                        re.IGNORECASE,
                    )
                )
            elif name == "schema":
                checks["metadata only"] = all(e["event"] != "tool_call" for e in events)
                checks["four table names"] = all(
                    re.search(rf"\b{table}\b", response.answer, re.IGNORECASE)
                    for table in ("orders", "order_items", "products", "users")
                )
            elif name == "off-topic":
                checks["analytics redirection"] = bool(
                    re.search(r"retail|analytics|data analysis", response.answer, re.IGNORECASE)
                )
                checks["no tool calls"] = all(e["event"] != "tool_call" for e in events)
            if jobs:
                outputs = [response.answer]
                if response.report_id:
                    outputs.append(row.get("report_body", ""))
                # Count both the application footer and the paraphrased disclaimers
                # that appeared beside it in the first reviewer transcripts.
                checks["one automatic filter notice per output"] = all(
                    len(
                        re.findall(
                            r"automatic(?:ally)?\s+(?:brand\s+)?filter", output, re.IGNORECASE
                        )
                    )
                    == 1
                    for output in outputs
                )
            if name == "report":
                checks["single-brand action in report"] = bool(
                    re.search(
                        r"categor|product|item|period|month|week|region|state|return|customer",
                        row["report_body"],
                        re.IGNORECASE,
                    )
                ) and not re.search(
                    r"(?:top[- ]performing|best[- ]selling|compar\w*|rank\w*|break\w* down|breakdown)"
                    r"[^.!?\n]{0,65}\bbrands\b|\bcategories or brands\b",
                    row["report_body"],
                    re.IGNORECASE,
                )
            checks["library contents"] = sorted(r.id for r in reports) == sorted(expected_ids)
            row["status"] = "pass" if all(checks.values()) else "fail"
            record_case(results, row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".retail-agent/eval-live.json"))
    parser.add_argument(
        "--pause-seconds",
        type=int,
        default=0,
        help="Wait between cases when the Gemini key has a low request quota.",
    )
    args = parser.parse_args()
    if args.pause_seconds < 0:
        parser.error("--pause-seconds must be nonnegative")
    results = {
        "suite": "live",
        "started": datetime.now(UTC).isoformat(),
        "pause_seconds": args.pause_seconds,
        "cases": [],
    }
    try:
        with TemporaryDirectory(prefix="retail-eval-") as tmp:
            run(Path(tmp), results, args.pause_seconds)
    except (AppError, ValueError) as error:
        # Validation errors can contain input values (including credentials).
        results["error"] = (
            str(error)
            if isinstance(error, AppError) or type(error) is ValueError
            else type(error).__name__
        )
    except Exception as error:  # noqa: BLE001 — retain completed cases without SDK payloads.
        results["error"] = type(error).__name__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    counts = Counter(case["status"] for case in results["cases"])
    total = len(CASES)
    summary = (
        f"{counts['pass']}/{total} passed, {counts['fail']} failed, {counts['skipped']} skipped"
    )
    if len(results["cases"]) < total:
        summary += f", {total - len(results['cases'])} not run"
    print(summary)
    if "error" in results:
        print("Run error:", results["error"])
    print("Results:", args.output)
    return int("error" in results or counts["pass"] != total)


if __name__ == "__main__":
    raise SystemExit(main())
