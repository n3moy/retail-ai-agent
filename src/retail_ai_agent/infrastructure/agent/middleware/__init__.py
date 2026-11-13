from langchain.agents.middleware import AgentMiddleware, AgentState, ToolErrorMiddleware

from retail_ai_agent.application.ports import Analytics
from retail_ai_agent.config import Config
from retail_ai_agent.infrastructure.agent.context import RunContext
from retail_ai_agent.infrastructure.agent.middleware.budget import ModelBudgetMiddleware
from retail_ai_agent.infrastructure.agent.middleware.planning import SqlPlanningMiddleware
from retail_ai_agent.infrastructure.agent.middleware.privacy import (
    ModelInputMiddleware,
    ResponseMiddleware,
)
from retail_ai_agent.infrastructure.agent.middleware.retry import BudgetedModelRetry
from retail_ai_agent.infrastructure.agent.middleware.tools import ToolPolicyMiddleware
from retail_ai_agent.infrastructure.agent.middleware.tracing import TraceMiddleware
from retail_ai_agent.infrastructure.privacy import LocalSanitizer
from retail_ai_agent.infrastructure.sql import SafeSql
from retail_ai_agent.infrastructure.tracing import JsonlTrace


def create_middleware(
    config: Config,
    sanitizer: LocalSanitizer,
    trace: JsonlTrace,
    sql: SafeSql,
    analytics: Analytics,
) -> list[AgentMiddleware[AgentState, RunContext]]:
    """Build wrappers from outermost to innermost; order is part of the runtime contract.

    Model: clean input -> retry -> reserve attempt -> trace -> clean response.
    Tool: trace -> safe error observations -> SQL policy and result masking.
    Settle reported usage before rejecting malformed responses. Unknown usage keeps its reserve.
    """
    policy = ToolPolicyMiddleware(trace, sql, analytics, sanitizer)
    return [
        SqlPlanningMiddleware(),
        ModelInputMiddleware(sanitizer),
        BudgetedModelRetry(config.model_max_retries),
        ModelBudgetMiddleware(config, trace),
        TraceMiddleware(trace, sanitizer),
        ResponseMiddleware(sanitizer),
        ToolErrorMiddleware(policy.on_error),
        policy,
    ]
