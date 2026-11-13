from collections.abc import Callable
from typing import Never

import httpx
from langchain.agents.middleware import ModelRequest, ModelResponse, ModelRetryMiddleware
from langchain_core.exceptions import ModelConnectionError, ModelTimeoutError
from langgraph.runtime import get_runtime

from retail_ai_agent.application.errors import AppError, ServiceUnavailable
from retail_ai_agent.application.progress import progress
from retail_ai_agent.infrastructure.agent.context import RunContext


def transient(error: Exception) -> bool:
    # OpenAI status errors can have code=None and wrap an HTTPStatusError.
    return any(
        isinstance(
            cause,
            (
                TimeoutError,
                ConnectionError,
                httpx.TransportError,
                ModelConnectionError,
                ModelTimeoutError,
            ),
        )
        or getattr(cause, "status_code", None) in {408, 429, 500, 502, 503, 504}
        or getattr(getattr(cause, "response", None), "status_code", None)
        in {408, 429, 500, 502, 503, 504}
        or getattr(cause, "code", None) in {408, 429, 500, 502, 503, 504}
        for cause in (error, error.__cause__)
        if cause is not None
    )


class BudgetedModelRetry(ModelRetryMiddleware):
    """Use LangChain's retry loop with the shared request deadline and safe errors."""

    def __init__(self, max_retries: int):
        super().__init__(
            max_retries=max_retries,
            retry_on=self.retry_within_budget,
            on_failure=self.unavailable,
            initial_delay=1,
            backoff_factor=2,
            max_delay=4,
            jitter=False,
        )

    @staticmethod
    def unavailable(error: Exception) -> Never:
        raise ServiceUnavailable(
            "The model is unavailable or returned an invalid response. Try again later."
        ) from None

    def wrap_model_call(
        self, request: ModelRequest[RunContext], handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        context = request.runtime.context
        if context.stopped:
            raise context.stopped
        context.model_attempt = 0
        return super().wrap_model_call(request, handler)

    def retry_within_budget(self, error: Exception) -> bool:
        if isinstance(error, AppError):
            return False
        if not transient(error):
            self.unavailable(error)
        context = get_runtime(RunContext).context
        if context.model_attempt == self.max_retries:
            return True  # Let LangChain invoke on_failure without reserving another backoff.
        delay = min(self.initial_delay * self.backoff_factor**context.model_attempt, self.max_delay)
        if context.budget.remaining() <= delay:
            raise ServiceUnavailable("The model timed out. Try again later.") from None
        context.model_attempt += 1
        progress("backoff")
        return True
