"""Build the single ReAct agent used for every chat turn."""

from typing import Any

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.agents.middleware.types import InputAgentState, OutputAgentState
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from retail_ai_agent.application.ports import Analytics
from retail_ai_agent.application.reports import ReportService
from retail_ai_agent.config import Config
from retail_ai_agent.infrastructure.agent.context import RunContext
from retail_ai_agent.infrastructure.agent.middleware import create_middleware
from retail_ai_agent.infrastructure.agent.middleware.history import HistoryMiddleware
from retail_ai_agent.infrastructure.agent.prompts import AGENT_PROMPT, REPORT_INSTRUCTION
from retail_ai_agent.infrastructure.agent.report_tools import report_tools
from retail_ai_agent.infrastructure.privacy import LocalSanitizer
from retail_ai_agent.infrastructure.sql import SafeSql
from retail_ai_agent.infrastructure.tracing import JsonlTrace

ChatGraph = CompiledStateGraph[AgentState[Any], RunContext, InputAgentState, OutputAgentState[Any]]


class AnalysisQuery(BaseModel):
    query: str = Field(description="One GoogleSQL SELECT against the schema in context.")
    for_report: bool = Field(
        default=False,
        description="Set true when this query supports the user's explicit request to create or save a new report.",
    )


@dynamic_prompt
def _prompt(request: ModelRequest[RunContext]) -> str:
    context = request.runtime.context
    instruction = {
        "report": REPORT_INSTRUCTION,
        "list": "The user requested a report search with list_reports.",
        "show": "The user requested the full text of a report with show_report.",
        "delete": "The user requested a deletion preview with delete_reports.",
    }.get(context.request.action, "")
    if context.deletion_outcomes:
        instruction += (
            "\nThe user has already responded in the CLI to the following deletion reviews. "
            "The recorded outcomes are final. Report the decision in the user's language. "
            "Do not request confirmation again for a resolved action. "
            + " ".join(context.deletion_outcomes)
        )
    scope = (
        "The employee has access to all brands."
        if context.brand_count is None
        else f"The employee has access to exactly {context.brand_count} distinct brand(s)."
    )
    return f"{AGENT_PROMPT}\n{scope}\nAvailable schema (column names and types):\n{context.schema}\n{instruction}"


def create_chat_graph(
    model: BaseChatModel,
    analytics: Analytics,
    reports: ReportService,
    config: Config,
    sanitizer: LocalSanitizer,
    trace: JsonlTrace,
    checkpointer: BaseCheckpointSaver,
    sql: SafeSql,
) -> ChatGraph:
    return create_agent(
        model,
        tools=[
            QuerySQLDatabaseTool(
                db=sql.catalog.database,
                description="Execute one GoogleSQL SELECT against the schema in context.",
                args_schema=AnalysisQuery,
            ),
            *report_tools(reports, sanitizer, trace),
        ],
        middleware=[
            _prompt,
            *create_middleware(config, sanitizer, trace, sql, analytics),
            HistoryMiddleware(config, reports, trace),
        ],
        context_schema=RunContext,
        checkpointer=checkpointer,
    ).with_config(recursion_limit=config.max_model_calls * 5 + 20, max_concurrency=1)
