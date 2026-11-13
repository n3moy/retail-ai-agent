from typing import Protocol

from retail_ai_agent.application.budget import Budget
from retail_ai_agent.application.models import ChatRequest, ChatResponse, PreparedQuery, QueryResult
from retail_ai_agent.domain.models import PendingDeletion, Report


class AgentRunner(Protocol):
    def run(self, request: ChatRequest) -> ChatResponse: ...

    def resume(self, conversation_id: str, action_id: str, approved: bool) -> ChatResponse: ...

    def reselect(self, conversation_id: str, action_id: str, ids: list[str]) -> ChatResponse: ...


class Analytics(Protocol):
    def execute(self, query: PreparedQuery, budget: Budget) -> QueryResult: ...


class ReportStore(Protocol):
    def save(self, report: Report) -> Report: ...

    def get(self, report_id: str) -> Report | None: ...

    def list(self, owner_id: str) -> list[Report]: ...

    def prepare(self, action: PendingDeletion) -> PendingDeletion: ...

    def pending(self, action_id: str) -> PendingDeletion | None: ...

    def replace_deletion(self, current: PendingDeletion, updated: PendingDeletion) -> None:
        """Atomically invalidate the old confirmation ID and publish its replacement."""
        ...

    def finish_deletion(self, action: PendingDeletion, approved: bool) -> int:
        """Atomically check the saved snapshot, delete, and mark the action finished."""
        ...
