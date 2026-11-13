import json
from collections.abc import Callable
from dataclasses import replace
from typing import cast

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.errors import is_invalid_resume
from langgraph.types import Command
from pydantic import BaseModel, ValidationError

from retail_ai_agent.application.errors import (
    AccessDenied,
    AppError,
    QueryFailed,
    ServiceUnavailable,
)
from retail_ai_agent.application.ports import Analytics
from retail_ai_agent.application.progress import progress
from retail_ai_agent.infrastructure.agent.context import RunContext
from retail_ai_agent.infrastructure.privacy import LocalSanitizer
from retail_ai_agent.infrastructure.sql import SafeSql
from retail_ai_agent.infrastructure.tracing import JsonlTrace


class ToolPolicyMiddleware(AgentMiddleware[AgentState, RunContext]):
    """Validate registered tool calls and execute SQL under the request policy."""

    def __init__(
        self,
        trace: JsonlTrace,
        sql: SafeSql,
        analytics: Analytics,
        sanitizer: LocalSanitizer,
        report_timeout_seconds: float,
    ):
        self.trace = trace
        self.sql = sql
        self.analytics = analytics
        self.sanitizer = sanitizer
        self.report_timeout_seconds = report_timeout_seconds

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
    ) -> ToolMessage | Command:
        # ToolCallRequest does not preserve the middleware's context type.
        context = cast(RunContext, request.runtime.context)
        call = request.tool_call
        call_id = call["id"]
        if context.stopped:
            raise context.stopped
        context.budget.remaining()
        if not call_id:
            raise QueryFailed("A tool call ID is required.")
        if request.tool is None or call["name"] == "sql_db_query_checker":
            raise AccessDenied("The tool is not available.")
        if call["name"] == "sql_db_query":
            context.budget.sql_attempt()
        # Validate only model-visible fields. LangChain injects ToolRuntime.
        schema = cast(type[BaseModel], request.tool.tool_call_schema)
        args = schema.model_validate(call["args"], strict=True, extra="forbid").model_dump()
        if call["name"] != "sql_db_query":
            return handler(request)

        if context.request.action == "auto" and args.get("for_report"):
            context.report_requested = True
            # The deadline is measured from the original start of this turn.
            context.budget.timeout = self.report_timeout_seconds

        # The toolkit's default executor cannot enforce our BigQuery job budget.
        prepared = self.sql.prepare(args["query"])
        if prepared.sql in context.empty_sql:
            raise QueryFailed(
                "No rows for this query earlier in this request. Check the schema or dates "
                "before changing the query, or explain that no data matched."
            )
        result = self.analytics.execute(prepared, context.budget)
        # An adapter or nested scalar subquery can still return a row of NULLs.
        # Report the missing measurement before the result can become report evidence.
        null_only = bool(result.rows) and all(
            value is None for row in result.rows for value in row.values()
        )
        result = replace(
            result,
            rows=tuple(
                self.sanitizer.mask_row(row, prepared.pii_columns, prepared.customer_ids)
                for row in (() if null_only else result.rows)
            ),
        )
        context.queries.append(result)
        payload: dict[str, object] = {"rows": result.rows, "truncated": result.truncated}
        if prepared.brand_filtered:
            payload["brand_filtered"] = True
            context.brand_filtered = True
        if not result.rows:
            progress("no_data")
            context.empty_sql.add(prepared.sql)
            payload["notice"] = (
                "No rows. Verify the dates, joins and filters against the question and schema. Correct a mistake once if possible; keep the requested scope. Otherwise report no matching data."
            )
            if null_only:
                payload["notice"] = (
                    "The query returned only NULL values. No measured value is available. "
                    "Check the matching row count and filters; do not report zero."
                )
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False, default=str),
            tool_call_id=call_id,
            name=call["name"],
        )

    def on_error(self, error: Exception, request: ToolCallRequest) -> str | None:
        context = cast(RunContext, request.runtime.context)
        if is_invalid_resume(error):
            return None
        if isinstance(error, ValidationError):
            return '{"error": "Invalid tool arguments. Check the tool schema.", "retryable": true}'
        if isinstance(error, AppError):
            if isinstance(error, QueryFailed):
                progress("query_error")
            else:
                context.stopped = error
            return json.dumps({"error": str(error), "retryable": isinstance(error, QueryFailed)})
        self.trace.record("tool_error", run_id=context.run_id, error_type=type(error).__name__)
        context.stopped = ServiceUnavailable(
            "Could not access the requested data or report. Try again later."
        )
        return json.dumps({"error": str(context.stopped), "retryable": False})
