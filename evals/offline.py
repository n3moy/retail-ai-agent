"""Run the submission's deterministic acceptance cases without credentials."""

import argparse
import json
import sys
import unittest
from pathlib import Path

# The existing unittest suite imports its fixture module relative to tests/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

# IDs link back to docs/error_handling_test_plan.md. Some checks cover only the
# CLI portion of the wider production scenario, as documented in the review guide.
CASES = {
    "EH-01": (
        "SQL syntax repair",
        "test_graph.GraphTests.test_syntax_error_and_empty_result_can_be_corrected_within_one_budget",
    ),
    "EH-02+11": (
        "Stale user SQL, reflected schema and scoped repair",
        "test_graph.GraphTests.test_stale_user_query_is_corrected_against_schema_and_scoped_rows",
    ),
    "EH-03": (
        "Stop repeated invalid SQL",
        "test_graph.GraphTests.test_repeated_invalid_candidates_exhaust_shared_sql_limit",
    ),
    "EH-04": (
        "Empty SUM vs real zero, including publication",
        "test_graph.GraphTests.test_empty_scalar_report_and_real_zero",
    ),
    "EH-07": (
        "429 during token counting and recovery",
        "test_middleware.MiddlewareTests.test_google_count_failure_does_not_reserve_or_generate_and_429_can_retry",
    ),
    "EH-08": (
        "Persistent transient failure and bounded backoff",
        "test_middleware.MiddlewareTests.test_retry_exhaustion_is_bounded_and_does_not_release_reservations",
    ),
    "EH-09": (
        "Malformed model call",
        "test_graph.GraphTests.test_malformed_tool_call_is_not_a_final_answer",
    ),
    "EH-10": (
        "403 and unavailable warehouse",
        "test_runtime.RuntimeTests.test_permission_failure_and_outage_do_not_replay_a_paid_job_or_leak_details",
    ),
    "EH-12": (
        "Sanitized warehouse diagnostics",
        "test_runtime.RuntimeTests.test_query_error_never_returns_raw_provider_payload",
    ),
    "EH-13": (
        "Timeout with a submitted job",
        "test_runtime.RuntimeTests.test_timeout_attempts_cancellation_and_keeps_cost_reservation",
    ),
    "EH-14": (
        "Scan limit before execution",
        "test_runtime.RuntimeTests.test_dry_run_scan_limit_prevents_execution",
    ),
    "EH-17": (
        "Failed report publication and retry",
        "test_reports.ReportTests.test_failed_save_is_not_published_and_can_be_retried",
    ),
    "EH-18": (
        "Lost deletion receipt and replay",
        "test_graph.GraphTests.test_failure_after_tool_commit_recovers_the_persisted_outcome",
    ),
    "EH-19": (
        "Frozen snapshot and changed scope",
        "test_reports.ReportTests.test_confirmation_binds_owner_conversation_scope_and_exact_snapshot",
    ),
    "EH-21": (
        "CLI accepts next question after error",
        "test_chat.ChatTests.test_cli_keeps_session_after_failure_and_can_start_a_new_one",
    ),
    "EH-22": (
        "Gemini payloads, trace, report and history redacted",
        "test_gemini.GeminiAdapterTests.test_free_text_pii_is_absent_from_count_generation_and_final_answer",
    ),
    "save-intent": (
        "Unsolicited save after real SQL rows",
        "test_graph.GraphTests.test_unrequested_report_tool_after_valid_rows_cannot_publish",
    ),
    "brand-grants": (
        "Recommendation context for zero, one, multiple and unrestricted brands",
        "test_graph.GraphTests.test_grant_count_is_in_the_prompt_before_sql",
    ),
    "filter-notice": (
        "One access notice in the answer and saved Markdown",
        "test_graph.GraphTests.test_scoped_report_and_answer_each_render_one_filter_notice",
    ),
    "cli-steps": (
        "Observable steps on stderr in both CLI modes",
        "test_cli.CliTests.test_steps_show_operations_on_redirected_stderr_only",
    ),
    "schema-context": (
        "Reflected tables and column types by role, including new tables",
        "test_schema.SchemaTests.test_context_exposes_reflected_schema_for_current_role_without_sample_rows",
    ),
    "null-report": (
        "A NULL-only adapter row cannot publish a report",
        "test_graph.GraphTests.test_null_aggregate_observation_cannot_publish",
    ),
    "scope": (
        "Unrelated /report, uncooperative generator",
        "test_graph.GraphTests.test_report_command_cannot_bypass_off_topic_refusal",
    ),
    "invalid-intent": (
        "Classifier returns a poem",
        "test_graph.GraphTests.test_unclassified_model_prose_is_never_displayed",
    ),
    "report-request": (
        "Requested report, evidence and scope",
        "test_graph.GraphTests.test_context_schema_query_and_save_report",
    ),
    "library": (
        "Find reports and preview deletion in chat",
        "test_graph.GraphTests.test_natural_language_report_library_uses_tools_without_sql",
    ),
    "mention": (
        "Delete a report by ID independently of text matches",
        "test_graph.GraphTests.test_delete_a_discussed_report_by_id_skips_reports_that_mention_it",
    ),
    "cancel": (
        "Cancel a frozen preview",
        "test_graph.GraphTests.test_cancellation_and_snapshot_exclude_new_reports",
    ),
    "confirm": (
        "Confirm and replay",
        "test_graph.GraphTests.test_delete_interrupt_can_be_confirmed_in_session_and_is_idempotent",
    ),
    "owner": (
        "Another employee cannot access a report",
        "test_reports.ReportTests.test_ceo_cannot_read_or_delete_another_users_reports",
    ),
    "restart": (
        "Report survives restart; stale confirmation expires",
        "test_graph.GraphTests.test_restart_discards_history_and_pending_pause_but_keeps_reports",
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".retail-agent/eval-offline.json"))
    args = parser.parse_args()
    rows = []
    for case, (title, test) in CASES.items():
        suite = unittest.defaultTestLoader.loadTestsFromName(test)
        result = unittest.TestResult()
        suite.run(result)
        passed = result.wasSuccessful() and result.testsRun == 1 and not result.skipped
        row = {"id": case, "scenario": title, "test": test, "status": "pass" if passed else "fail"}
        if not passed:
            row["diagnostic"] = [
                message for _, message in result.failures + result.errors + result.skipped
            ]
        rows.append(row)
        print(f"{row['status'].upper():4}  {case:15} {title}")
    passed = sum(row["status"] == "pass" for row in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"suite": "offline", "passed": passed, "cases": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{passed}/{len(rows)} passed. {args.output}")
    return int(passed != len(rows))


if __name__ == "__main__":
    raise SystemExit(main())
