from typing import Annotated, Literal, NotRequired, TypedDict
from uuid import uuid4

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from langgraph.func import task
from langgraph.types import interrupt

from retail_ai_agent.application.errors import AppError
from retail_ai_agent.application.progress import progress
from retail_ai_agent.application.reports import ReportService
from retail_ai_agent.domain.models import Report
from retail_ai_agent.infrastructure.agent.context import RunContext
from retail_ai_agent.infrastructure.privacy import LocalSanitizer
from retail_ai_agent.infrastructure.text_privacy import REDACTION
from retail_ai_agent.infrastructure.tracing import JsonlTrace


class ReviewDecision(TypedDict):
    type: Literal["approve", "reject", "select", "options"]
    report_ids: NotRequired[list[str]]


def publish_analysis(
    reports: ReportService, trace: JsonlTrace, context: RunContext, body: str
) -> str:
    if not context.report_requested:
        return "Report not saved. Give the analysis in chat unless the user requested a new report."
    if context.report_id:
        return f"This request already saved report {context.report_id}."
    sql = [query.sql for query in context.queries[: context.observed_queries] if query.rows]
    if not sql or not body.strip():
        return (
            "Report not saved. A non-empty analysis and query results from this turn are required."
        )
    progress("save_report")
    report = reports.save(
        context.run_id,
        context.request.conversation_id,
        context.request.question,
        sql,
        context.with_warnings(body),
    )
    context.report_id = report.id
    progress("report_saved")
    trace.record("report_saved", run_id=context.run_id, report_id=report.id)
    return f"Report saved: {report.id}"


def report_tools(
    reports: ReportService, sanitizer: LocalSanitizer, trace: JsonlTrace
) -> list[BaseTool]:
    @tool
    def list_reports(
        runtime: ToolRuntime[RunContext], search: str = "", conversation_only: bool = False
    ) -> str:
        """Find your saved reports, newest first. Search matches literal text in the question or body."""
        request = runtime.context.request
        if request.action == "list":
            search, conversation_only = request.search, request.conversation_only
        conversation = request.conversation_id if conversation_only else ""
        if REDACTION.search(search):
            # Older reports can still contain raw contacts. Compare their sanitized
            # text locally and present all matching IDs for the employee to choose.
            matches = [
                r
                for r in reports.search("", conversation)
                if search.casefold() in sanitizer.clean(r.question + "\n" + r.body).casefold()
            ]
        else:
            matches = reports.search(search, conversation)
        return (
            "\n".join(f"{report.id} — {report.question}" for report in matches)
            or "No reports found."
        )

    @tool
    def show_report(report_id: str, runtime: ToolRuntime[RunContext]) -> str:
        """Read the full text of your saved report by ID."""
        request = runtime.context.request
        report = reports.get(request.report_id if request.action == "show" else report_id)
        return f"{report.question}\n\n{report.body}"

    @tool
    def save_report(body: str, runtime: ToolRuntime[RunContext]) -> str:
        """Save the explicitly requested Markdown analysis with this turn's executed SQL."""
        return publish_analysis(reports, trace, runtime.context, body)

    def report_rows(matches: list[Report]) -> str:
        return sanitizer.clean("\n".join(f"{r.id} — {r.question}" for r in matches))

    def preview(action_id: str, matches: list[Report]) -> dict:
        return {
            "type": "tool_approval",
            "action_id": action_id,
            "description": f"Selected reports ({len(matches)}):\n{report_rows(matches)}",
        }

    @task
    def selection_options() -> str:
        return report_rows(reports.search())

    @task
    def prepare(
        run_id: str, conversation: str, search: str, conversation_only: bool, report_id: str
    ) -> dict:
        matches = reports.search(
            search, conversation if conversation_only else "", report_id=report_id
        )
        if not matches:
            return {}
        action = reports.prepare(uuid4().hex, conversation, matches)
        trace.record("deletion_prepared", run_id=run_id, report_ids=action.report_ids)
        return preview(action.id, matches)

    @task
    def reselect(action_id: str, conversation: str, ids: list[str], run_id: str) -> dict:
        try:
            report_ids = list(dict.fromkeys(ids))
            matches = [reports.get(report_id) for report_id in report_ids]
            new_action_id = uuid4().hex
            updated = preview(new_action_id, matches)
            trace.record("deletion_selection_requested", run_id=run_id, report_ids=report_ids)
            reports.reselect(action_id, conversation, new_action_id, matches)
            return updated
        except AppError as error:
            return {"review_error": sanitizer.clean(str(error))}
        except OSError:
            return {"storage_error": "Could not update the selection. Check the preview and retry."}

    @task
    def resolve(action_id: str, conversation: str, approved: bool) -> dict:
        try:
            reports.confirm(action_id, conversation, approved=approved)
            action = reports.pending(action_id, conversation)
            return {
                "status": action.status,
                "count": action.deleted_count,
                "report_ids": action.report_ids,
            }
        except AppError as error:
            return {"review_error": sanitizer.clean(str(error))}
        except OSError:
            return {
                "storage_error": "Could not record the decision. Retry confirmation or cancellation."
            }

    @tool
    def delete_reports(
        runtime: ToolRuntime[RunContext],
        report_id: str = "",
        search: Annotated[
            str, "Literal keyword or phrase copied from the request, e.g. Client X."
        ] = "",
        conversation_only: bool = False,
        all_reports: bool = False,
    ) -> str:
        """Prepare a deletion preview. For 'delete reports mentioning Client X', pass search='Client X'.
        Use the phrase alone, without command words or surrounding quotes. Use conversation_only
        for reports created here, optionally together with search. Use report_id for one known
        report, or all_reports alone for an explicit whole-library request. The employee can
        change the selection and must confirm the final IDs.
        """
        context = runtime.context
        request = context.request
        if request.action == "delete":
            search, conversation_only = request.search, request.conversation_only
            report_id = request.report_id
            all_reports = not (search or conversation_only or report_id)
        if (REDACTION.search(request.question) or REDACTION.search(search)) and not (
            request.action == "delete" and request.report_id
        ):
            # A model may omit the masked mention or pick an ID on its own. Only an
            # employee-selected ID resolves the ambiguity. Never prepare a broad plan.
            candidates = reports.search("", request.conversation_id if conversation_only else "")
            choices = "\n".join(f"{r.id} — {sanitizer.clean(r.question)}" for r in candidates)
            return (
                "Personal details cannot identify a report for deletion. "
                "Select an ID with /delete REPORT_ID.\n" + (choices or "No reports found.")
            )
        if (search and not search.strip()) or not (
            search or report_id or conversation_only or all_reports
        ):
            return "Specify a literal phrase, a report ID, this conversation, or explicitly all reports."
        if (all_reports and (search or conversation_only or report_id)) or (
            report_id and (search or conversation_only)
        ):
            return (
                "Use a single report ID, a phrase and/or conversation filter, or all reports alone."
            )
        search = search.strip()
        if search and request.action != "delete":
            question = request.question.strip().casefold()
            phrase = search.casefold()
            if phrase not in question or phrase.rstrip(".!?") == question.rstrip(".!?"):
                return (
                    "Pass only the literal keyword or phrase from the request in search, "
                    "for example Client X. If unclear, ask the employee to quote it."
                )
        review = prepare(
            context.run_id, request.conversation_id, search, conversation_only, report_id
        ).result()
        if not review:
            return (
                "No matching reports of your own were found. To select manually, "
                "use /reports and /delete <report ID>."
            )
        # LangGraph checkpoints the snapshot and each commit attempt. Replaying a failed
        # decision retrieves its error and pauses again before attempting any new write.
        result = {}
        while "status" not in result:
            decision = interrupt({**review, **result}, response_schema=ReviewDecision)
            if decision["type"] == "options":
                options = selection_options().result()
                result = {"description": f"{review['description']}\nAvailable reports:\n{options}"}
                continue
            if decision["type"] == "select":
                result = reselect(
                    review["action_id"],
                    request.conversation_id,
                    decision.get("report_ids", []),
                    context.run_id,
                ).result()
                if "action_id" in result:
                    review, result = result, {}
                continue
            result = resolve(
                review["action_id"], request.conversation_id, decision["type"] == "approve"
            ).result()
        trace.record("deletion_resolved", run_id=context.run_id, **result)
        outcome = (
            f"Reports deleted: {result['count']}."
            if result["status"] == "deleted"
            else "Deletion cancelled."
        )
        if result["status"] == "deleted" and context.report_id in result["report_ids"]:
            context.report_id = ""
        context.deletion_outcomes.append(outcome)
        return outcome

    return [list_reports, show_report, save_report, delete_reports]
