from collections.abc import Callable
from dataclasses import replace
from time import monotonic
from typing import cast

from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from retail_ai_agent.application.errors import ServiceUnavailable
from retail_ai_agent.application.progress import progress
from retail_ai_agent.infrastructure.agent.context import RunContext
from retail_ai_agent.infrastructure.privacy import LocalSanitizer
from retail_ai_agent.infrastructure.tracing import JsonlTrace


class TraceMiddleware(AgentMiddleware[AgentState, RunContext]):
    """Record individual model attempts and all tool observations."""

    def __init__(self, trace: JsonlTrace, sanitizer: LocalSanitizer):
        self.sanitizer = sanitizer
        self.trace = trace

    def wrap_model_call(
        self,
        request: ModelRequest[RunContext],
        handler: Callable[[ModelRequest[RunContext]], ModelResponse],
    ) -> ModelResponse:
        context = request.runtime.context
        started = monotonic()
        self.trace.record(
            "model_request",
            run_id=context.run_id,
            attempt=context.model_attempt,
            messages=[{"role": m.type, "content": m.content} for m in request.messages],
        )
        try:
            response = handler(request)
        except Exception as error:  # Log the type, never provider payloads.
            self.trace.record(
                "model_error",
                run_id=context.run_id,
                error_type=type(error).__name__,
                attempt=context.model_attempt,
            )
            raise
        if any(isinstance(m, AIMessage) and m.invalid_tool_calls for m in response.result):
            self.trace.record(
                "model_error",
                run_id=context.run_id,
                error_type="InvalidToolCall",
                attempt=context.model_attempt,
            )
            return response  # Account for provider usage before rejecting this response.
        self.trace.record(
            "model_response",
            run_id=context.run_id,
            usage=next(
                (
                    m.usage_metadata
                    for m in response.result
                    if isinstance(m, AIMessage) and m.usage_metadata
                ),
                None,
            ),
            latency_ms=round((monotonic() - started) * 1000),
            messages=[
                {"role": m.type, "text": m.text, "tool_calls": getattr(m, "tool_calls", [])}
                for m in response.result
            ],
        )
        return response

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
    ) -> ToolMessage | Command:
        context = cast(RunContext, request.runtime.context)
        call = request.tool_call
        # Publication emits its own events after authorization, including /report fallback.
        if call["name"] != "save_report":
            progress(call["name"])
        self.trace.record("tool_call", run_id=context.run_id, call=call)
        result = handler(request)
        match result:
            case ToolMessage() as observation:
                return self._record_observation(context.run_id, observation)
            case Command(update={"messages": [ToolMessage() as observation]} as update):
                # write_todos updates graph state and returns an observation in the same command.
                message = self._record_observation(context.run_id, observation)
                return replace(result, update={**update, "messages": [message]})
            case _:
                raise ServiceUnavailable("Unsupported tool result.")

    def _record_observation(self, run_id: str, message: ToolMessage) -> ToolMessage:
        message = message.model_copy(update={"content": self.sanitizer.value(message.content)})
        self.trace.record(
            "tool_result", run_id=run_id, call_id=message.tool_call_id, content=message.content
        )
        return message
