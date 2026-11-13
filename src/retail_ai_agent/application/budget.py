from dataclasses import dataclass, field
from time import monotonic

from retail_ai_agent.application.errors import BudgetExceeded


@dataclass
class Budget:
    max_usd: float = 1.0
    timeout: float = 120
    max_model_calls: int = 12
    max_sql_attempts: int = 4
    reserved_usd: float = 0
    model_usd: float = 0
    bigquery_usd: float = 0
    run_id: str = ""
    model_calls: int = 0
    sql_attempts: int = 0
    started: float = field(default_factory=monotonic)

    @property
    def spent_usd(self) -> float:
        return self.reserved_usd + self.model_usd + self.bigquery_usd

    def remaining(self) -> float:
        remaining = self.timeout - (monotonic() - self.started)
        if remaining <= 0:
            raise BudgetExceeded(
                "The request timed out. Try a smaller date range or a simpler question."
            )
        return remaining

    def reserve(self, amount: float) -> None:
        self.remaining()
        if self.spent_usd + amount > self.max_usd:
            raise BudgetExceeded("The request reached its budget. Try a narrower question.")
        self.reserved_usd += amount

    def check_model_call(self) -> None:
        self.remaining()
        if self.model_calls >= self.max_model_calls:
            raise BudgetExceeded(
                "The request reached its model call limit. Try a narrower question."
            )

    def model_call(self, reservation: float = 0.0) -> None:
        self.check_model_call()
        self.reserve(reservation)
        self.model_calls += 1

    def sql_attempt(self) -> None:
        self.remaining()
        if self.sql_attempts >= self.max_sql_attempts:
            raise BudgetExceeded(
                "Could not complete the task within allowed number of SQL queries."
            )
        self.sql_attempts += 1
