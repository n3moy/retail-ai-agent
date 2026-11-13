import json
from collections.abc import Callable
from hashlib import sha256

from google.genai.types import HttpOptions
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from retail_ai_agent.application.errors import BudgetExceeded
from retail_ai_agent.application.progress import progress
from retail_ai_agent.config import Config
from retail_ai_agent.infrastructure.agent.context import InputTokenCount, RunContext
from retail_ai_agent.infrastructure.agent.gemini_tokens import count_gemini_input
from retail_ai_agent.infrastructure.pricing import TokenRates, gemini_rates
from retail_ai_agent.infrastructure.tracing import JsonlTrace


def estimate_input(request: ModelRequest[RunContext]) -> InputTokenCount:
    """Use measured input for an identical prefix, and reserve one token per new UTF-8 byte."""
    header = json.dumps(
        {
            "system": request.system_message.content if request.system_message else None,
            "tools": [convert_to_openai_tool(tool) for tool in request.tools],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    # A measurement belongs to the specific model that processed these inputs.
    digest = sha256(str(id(request.model)).encode() + b"\n" + header)
    tokens = len(header)
    measured = request.runtime.context.measured_input
    for message in request.messages:
        # The legacy estimator excludes compatibility copies of normalized tool calls.
        exclude = None
        if isinstance(message, AIMessage) and message.tool_calls:
            exclude = {"additional_kwargs": {"function_call", "tool_calls"}}
        content = message.model_dump_json(
            include={"type", "name", "content", "tool_calls", "tool_call_id", "additional_kwargs"},
            exclude=exclude,
            exclude_none=True,
        ).encode()
        digest.update(b"\n" + content)
        tokens += len(content)
        if measured and digest.digest() == measured.fingerprint:
            tokens = measured.tokens
    return InputTokenCount(digest.digest(), tokens)


class ModelBudgetMiddleware(AgentMiddleware[AgentState, RunContext]):
    """Reserve each attempt, bound its timeout, and reconcile successful usage."""

    def __init__(self, config: Config, trace: JsonlTrace):
        self.config = config
        self.trace = trace
        self.gemini = config.model_provider == "gemini"
        self.rates = (
            gemini_rates(config.model_name, config.gemini_billing_tier)
            if self.gemini
            else TokenRates(
                config.input_usd_per_million,
                config.input_usd_per_million,
                config.output_usd_per_million,
            )
        )

    def wrap_model_call(
        self,
        request: ModelRequest[RunContext],
        handler: Callable[[ModelRequest[RunContext]], ModelResponse],
    ) -> ModelResponse:
        context = request.runtime.context
        budget = context.budget
        budget.check_model_call()
        if context.model_attempt:
            progress("retry")
        else:
            progress("model" if context.intent else "classify")
        prompt = None
        if self.gemini:
            assert self.config.google_ai_studio_api_key is not None
            input_tokens = count_gemini_input(
                request,
                self.config.google_ai_studio_api_key,
                min(budget.remaining(), self.config.model_timeout_seconds),
            )
        else:
            prompt = estimate_input(request)
            input_tokens = prompt.tokens
        # The cache hit count is known only after generation. Reserve the uncached tariff.
        estimate = self.rates.cost(input_tokens, 0, self.config.max_output_tokens)
        self.trace.record(
            "model_budget_check",
            run_id=context.run_id,
            input_tokens=input_tokens,
            count_source="google_count_tokens" if self.gemini else "estimate",
            max_output_tokens=self.config.max_output_tokens,
            required_usd=estimate,
            available_usd=budget.max_usd - budget.spent_usd,
        )
        timeout = min(self.config.model_timeout_seconds, budget.remaining())
        settings = {**request.model_settings, "timeout": timeout}
        if self.gemini and timeout < 10:
            # The SDK truncates to milliseconds and treats zero as an unlimited timeout.
            if timeout < 0.001:
                raise BudgetExceeded(
                    "The request timed out before Gemini could generate a response."
                )
            # Gemini requires at least 10 seconds on the server. Keep the shorter client timeout.
            options = HttpOptions.model_validate(settings.get("http_options") or {})
            settings["http_options"] = options.model_copy(
                update={"headers": {**(options.headers or {}), "X-Server-Timeout": "10"}}
            )
        budget.model_call(estimate)
        response = handler(request.override(model_settings=settings))
        usage = next(
            (
                m.usage_metadata
                for m in response.result
                if isinstance(m, AIMessage) and m.usage_metadata
            ),
            None,
        )
        if usage:
            cached = usage.get("input_token_details", {}).get("cache_read", 0)
            actual = self.rates.cost(usage["input_tokens"], cached, usage["output_tokens"])
            budget.reserved_usd -= estimate
            budget.model_usd += actual
            self.trace.record(
                "model_usage",
                run_id=context.run_id,
                model=self.config.model_name,
                tariff=self.config.gemini_billing_tier if self.gemini else "configured",
                input_tokens=usage["input_tokens"],
                cached_input_tokens=cached,
                output_tokens=usage["output_tokens"],
                input_rate=self.rates.input,
                cached_rate=self.rates.cached,
                output_rate=self.rates.output,
                cost_usd=actual,
            )
            if prompt and usage["input_tokens"] > 0:
                context.measured_input = InputTokenCount(prompt.fingerprint, usage["input_tokens"])
        budget.remaining()
        return response
