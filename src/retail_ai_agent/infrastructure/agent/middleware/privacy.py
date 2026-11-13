from collections.abc import Callable

from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage

from retail_ai_agent.application.errors import ServiceUnavailable
from retail_ai_agent.infrastructure.agent.context import RunContext
from retail_ai_agent.infrastructure.privacy import LocalSanitizer


def clean_message[MessageT: BaseMessage](
    message: MessageT, sanitizer: LocalSanitizer, tool_names: frozenset[str] = frozenset()
) -> MessageT:
    updates = {"content": sanitizer.value(message.content)}
    if isinstance(message, AIMessage):
        # Keep malformed calls rejectable by tool policy. ToolMessage needs a string ID
        # to attach the error observation to the same call.
        updates["tool_calls"] = [
            {
                **call,
                "id": call["id"] or "",
                "name": call["name"]
                if call["name"] in tool_names
                else sanitizer.clean(call["name"]),
                "args": sanitizer.value(call["args"]),
            }
            for call in message.tool_calls
        ]
        updates["invalid_tool_calls"] = sanitizer.value(message.invalid_tool_calls)
        # Gemini can serialize legacy function_call.arguments when normalized calls are
        # absent. Keep opaque provider metadata (including thought signatures) verbatim.
        extra = message.additional_kwargs.copy()
        for key in ("function_call", "tool_calls"):
            if key in extra:
                if message.tool_calls:
                    del extra[key]
                else:
                    extra[key] = sanitizer.value(extra[key])
        updates["additional_kwargs"] = extra
    return message.model_copy(update=updates)


class ModelInputMiddleware(AgentMiddleware[AgentState, RunContext]):
    """Redact history before cost estimation, tracing, and provider calls."""

    def __init__(self, sanitizer: LocalSanitizer):
        self.sanitizer = sanitizer

    def wrap_model_call(
        self,
        request: ModelRequest[RunContext],
        handler: Callable[[ModelRequest[RunContext]], ModelResponse],
    ) -> ModelResponse:
        context = request.runtime.context
        # A sibling tool call in the same model response cannot have seen new SQL rows yet.
        context.observed_queries = len(context.queries)
        tool_names = frozenset(t.name for t in request.tools)
        request = request.override(
            messages=[clean_message(m, self.sanitizer, tool_names) for m in request.messages]
        )
        response = handler(request)
        last = response.result[-1]
        if (
            isinstance(last, AIMessage)
            and not last.tool_calls
            and not last.invalid_tool_calls
            and not last.text.strip()
        ):
            response = handler(request)  # Retry an empty reply within the same model budget.
        if any(isinstance(m, AIMessage) and m.invalid_tool_calls for m in response.result):
            raise ServiceUnavailable("The model returned an invalid tool call.")
        return response


class ResponseMiddleware(AgentMiddleware[AgentState, RunContext]):
    """Redact model results before they enter the trace or graph state."""

    def __init__(self, sanitizer: LocalSanitizer):
        self.sanitizer = sanitizer

    def wrap_model_call(
        self,
        request: ModelRequest[RunContext],
        handler: Callable[[ModelRequest[RunContext]], ModelResponse],
    ) -> ModelResponse:
        response = handler(request)
        tool_names = frozenset(t.name for t in request.tools)
        response.result = [clean_message(m, self.sanitizer, tool_names) for m in response.result]
        return response
