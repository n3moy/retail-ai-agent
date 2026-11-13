"""Invoke the chat agent and resume a reviewed tool call on the same conversation."""

import hashlib
import json
from dataclasses import replace
from time import monotonic
from uuid import uuid4

from langchain.agents.middleware.types import InputAgentState
from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command, Interrupt
from langsmith import tracing_context

from retail_ai_agent.application.errors import AppError, InvalidRequest
from retail_ai_agent.application.models import ChatRequest, ChatResponse
from retail_ai_agent.application.reports import ReportService
from retail_ai_agent.config import Config
from retail_ai_agent.domain.models import Actor
from retail_ai_agent.infrastructure.agent.context import RunContext, RunContextFactory
from retail_ai_agent.infrastructure.agent.graph import ChatGraph
from retail_ai_agent.infrastructure.agent.middleware.history import previous_turns
from retail_ai_agent.infrastructure.agent.report_tools import ReviewDecision
from retail_ai_agent.infrastructure.privacy import LocalSanitizer
from retail_ai_agent.infrastructure.tracing import JsonlTrace


class LangGraphRunner:
    def __init__(
        self,
        graph: ChatGraph,
        reports: ReportService,
        actor: Actor,
        config: Config,
        trace: JsonlTrace,
        contexts: RunContextFactory,
        sanitizer: LocalSanitizer,
    ) -> None:
        self.graph = graph
        self.reports = reports
        self.actor = actor
        self.config = config
        self.trace = trace
        self.contexts = contexts
        self.sanitizer = sanitizer
        # Like the checkpointer, pending turns belong to this CLI process.
        self._paused: dict[str, tuple[RunContext, float]] = {}

    def _graph_config(self, conversation_id: str) -> RunnableConfig:
        identity = json.dumps([self.actor.user_id, self.actor.scope, conversation_id])
        return {"configurable": {"thread_id": hashlib.sha256(identity.encode()).hexdigest()}}

    def run(self, request: ChatRequest) -> ChatResponse:
        conversation = request.conversation_id
        configuration = self._graph_config(conversation)
        snapshot = self.graph.get_state(configuration)
        if snapshot.interrupts:
            return self._preview(conversation, snapshot.interrupts[0])
        self._paused.pop(conversation, None)
        # The first graph checkpoint and the report index consume this same request.
        try:
            request = replace(
                request,
                question=self.sanitizer.clean(request.question),
                search=self.sanitizer.clean(request.search),
            )
        except AppError as error:
            return ChatResponse(str(error), conversation, status="failed")
        messages: list[AnyMessage] = [HumanMessage(content=request.question)]
        if snapshot.next:
            # A failed tool/model turn may have left an unmatched tool call in the checkpoint.
            history = previous_turns(snapshot.values["messages"])
            messages = [RemoveMessage(id=REMOVE_ALL_MESSAGES), *history, *messages]
        context = self.contexts.create(uuid4().hex, request)
        return self._invoke({"messages": messages}, context, configuration)

    def resume(self, conversation_id: str, action_id: str, approved: bool) -> ChatResponse:
        return self._review(
            conversation_id, action_id, {"type": "approve" if approved else "reject"}
        )

    def reselect(self, conversation_id: str, action_id: str, ids: list[str]) -> ChatResponse:
        if not ids:
            return self._review(conversation_id, action_id, {"type": "options"})
        return self._review(conversation_id, action_id, {"type": "select", "report_ids": ids})

    def _review(
        self, conversation_id: str, action_id: str, decision: ReviewDecision
    ) -> ChatResponse:
        action = self.reports.pending(action_id, conversation_id)
        configuration = self._graph_config(conversation_id)
        snapshot = self.graph.get_state(configuration)
        pause = next((p for p in snapshot.interrupts if p.value["action_id"] == action_id), None)
        suspended = self._paused.get(conversation_id)
        if not pause or not suspended:
            if action.status != "pending":
                return ChatResponse(
                    f"Action already completed: {action.status}; deleted {action.deleted_count}.",
                    conversation_id,
                )
            raise InvalidRequest(
                "No such confirmation exists in the current session. Repeat the deletion request."
            )
        context, paused_at = suspended
        context.budget.started += monotonic() - paused_at
        previous_outcomes = len(context.deletion_outcomes)
        response = self._invoke(Command(resume={pause.id: decision}), context, configuration)
        return replace(
            response, review_outcome="\n".join(context.deletion_outcomes[previous_outcomes:])
        )

    def _invoke(
        self, value: InputAgentState | Command, context: RunContext, configuration: RunnableConfig
    ) -> ChatResponse:
        run_id = context.run_id
        conversation = context.request.conversation_id
        budget = context.budget
        self.trace.record(
            "run_resumed" if isinstance(value, Command) else "run_started",
            run_id=run_id,
            user_id=self.actor.user_id,
            conversation_id=conversation,
            scope=self.actor.scope,
            provider=self.config.model_provider,
            model=self.config.model_name,
        )
        response = None
        try:
            with tracing_context(enabled=False):
                result = self.graph.invoke(value, config=configuration, context=context)
            if pauses := result.get("__interrupt__"):
                self._paused[conversation] = (context, monotonic())
                pause = pauses[0]
                if error := pause.value.get("review_error"):
                    raise InvalidRequest(error)
                if error := pause.value.get("storage_error"):
                    response = ChatResponse(
                        error,
                        conversation,
                        status="failed",
                        pending_action_id=pause.value["action_id"],
                    )
                else:
                    response = self._preview(conversation, pause)
            else:
                self._paused.pop(conversation, None)
                response = ChatResponse(
                    self.sanitizer.clean(result["messages"][-1].text),
                    conversation,
                    report_id=context.report_id,
                )
        except AppError as error:
            self.trace.record(
                "run_failed",
                run_id=run_id,
                error_type=type(error).__name__,
                message=self.sanitizer.clean(str(error)),
            )
            if isinstance(value, Command) and isinstance(error, InvalidRequest):
                raise
            response = ChatResponse(self.sanitizer.clean(str(error)), conversation, status="failed")
        except Exception as error:  # noqa: BLE001 — contain graph/SDK failures at the runner boundary.
            self.trace.record("run_error", run_id=run_id, error_type=type(error).__name__)
            response = ChatResponse(
                "Could not complete the request. Retry it or start a new conversation with /new.",
                conversation,
                status="failed",
            )
        finally:
            if isinstance(value, Command) and conversation in self._paused:
                if self.graph.get_state(configuration).interrupts:
                    self._paused[conversation] = (context, monotonic())
                else:
                    self._paused.pop(conversation)
            self.trace.record(
                "run_paused" if conversation in self._paused else "run_finished",
                run_id=run_id,
                status=response.status if response else "failed",
                latency_ms=round((monotonic() - budget.started) * 1000),
                model_calls=budget.model_calls,
                sql_attempts=budget.sql_attempts,
                model_usd=budget.model_usd,
                bigquery_usd=budget.bigquery_usd,
                unresolved_reserve_usd=round(budget.reserved_usd, 12),
                estimated_usd=round(budget.spent_usd, 6),  # Legacy budget consumption field.
            )
        if response.status == "failed" and (context.report_id or context.deletion_outcomes):
            # The final model call can fail after a tool has already committed.
            outcomes = list(context.deletion_outcomes)
            if context.report_id:
                outcomes.append(f"Report saved: {context.report_id}")
            response = replace(
                response,
                answer="\n".join([response.answer, *outcomes]),
                report_id=context.report_id,
            )
        return response

    @staticmethod
    def _preview(conversation: str, pause: Interrupt) -> ChatResponse:
        return ChatResponse(
            pause.value["description"],
            conversation,
            status="awaiting_confirmation",
            pending_action_id=pause.value["action_id"],
        )
