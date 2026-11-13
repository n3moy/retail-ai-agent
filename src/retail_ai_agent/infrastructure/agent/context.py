from dataclasses import dataclass, field

from retail_ai_agent.application.budget import Budget
from retail_ai_agent.application.errors import AppError
from retail_ai_agent.application.models import ChatRequest, QueryResult
from retail_ai_agent.config import Config
from retail_ai_agent.infrastructure.sql import SafeSql

BRAND_NOTICE = (
    "An automatic filter has been applied to your accessible brands. "
    "Data from other brands is excluded from the analysis."
)


@dataclass(frozen=True)
class InputTokenCount:
    fingerprint: bytes
    tokens: int


@dataclass
class RunContext:
    run_id: str
    budget: Budget
    request: ChatRequest
    schema: str = ""
    queries: list[QueryResult] = field(default_factory=list)
    empty_sql: set[str] = field(default_factory=set)
    intent: str = ""
    observed_queries: int = 0
    brand_count: int | None = None
    brand_filtered: bool = False
    stopped: AppError | None = None
    report_id: str = ""
    deletion_outcomes: list[str] = field(default_factory=list)
    # The runner executes model calls sequentially; retries reset this for each call.
    model_attempt: int = 0
    measured_input: InputTokenCount | None = None

    @property
    def may_publish(self) -> bool:
        return self.request.action == "report" or (
            self.request.action == "auto" and self.intent == "REPORT"
        )

    def with_warnings(self, text: str) -> str:
        if not self.brand_filtered:
            return text
        # This notice is rendered by the application in both chat and Markdown.
        # Keep a single copy even when the model quotes it from earlier history.
        body = text.replace(BRAND_NOTICE, "").strip()
        return "\n\n".join(filter(None, (body, BRAND_NOTICE)))


class RunContextFactory:
    """Create fresh mutable counters for each request, using application settings."""

    def __init__(self, config: Config, sql: SafeSql) -> None:
        self.config = config
        self.brand_count = None if sql.brands is None else len(set(sql.brands))
        # One process serves one actor. Startup has already loaded metadata.
        self.schema = "\n".join(
            f"{table}(" + ", ".join(f"{name} {kind}" for name, kind in columns.items()) + ")"
            for table, columns in sorted(sql.catalog.columns.items())
            if sql.table_access(table)
        )

    def create(self, run_id: str, request: ChatRequest) -> RunContext:
        return RunContext(
            run_id=run_id,
            request=request,
            schema=self.schema,
            brand_count=self.brand_count,
            budget=Budget(
                run_id=run_id,
                max_usd=self.config.request_budget_usd,
                timeout=self.config.report_timeout_seconds
                if request.action == "report"
                else self.config.question_timeout_seconds,
                max_model_calls=self.config.max_model_calls,
                max_sql_attempts=self.config.max_sql_attempts,
            ),
        )
