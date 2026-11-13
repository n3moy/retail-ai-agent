from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from retail_ai_agent.application.errors import ServiceUnavailable
from retail_ai_agent.application.reports import ReportService
from retail_ai_agent.config import Config
from retail_ai_agent.infrastructure.agent.context import RunContext
from retail_ai_agent.infrastructure.agent.report_tools import publish_analysis
from retail_ai_agent.infrastructure.tracing import JsonlTrace


def previous_turns(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Everything before the active question is already compacted into complete turns."""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[:index]
    return []


class HistoryMiddleware(AgentMiddleware[AgentState, RunContext]):
    def __init__(self, config: Config, reports: ReportService, trace: JsonlTrace):
        self.config = config
        self.reports = reports
        self.trace = trace

    def after_agent(self, state: AgentState, runtime: Runtime[RunContext]) -> dict:
        context = runtime.context
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.text.strip():
            raise ServiceUnavailable("The model did not generate an answer. Repeat your question.")
        answer = context.with_warnings(last.text)
        # An explicit request authorizes publication. Some models return the finished
        # draft directly. Apply the same evidence and idempotency checks as the tool.
        if (
            context.request.action == "report"
            and not context.report_id
            and not context.deletion_outcomes
            and any(q.rows for q in context.queries[: context.observed_queries])
        ):
            publish_analysis(self.reports, self.trace, context, last.text)
        if context.report_id and context.report_id not in answer:
            answer += f"\nReport saved: {context.report_id}"
        history = previous_turns(state["messages"])
        question = state["messages"][len(history)]
        history.extend([question, AIMessage(content=answer)])
        history = history[-2 * self.config.history_max_turns :]
        while (
            len(history) > 2 and sum(len(m.text) for m in history) > self.config.history_max_chars
        ):
            history = history[2:]
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *history]}
