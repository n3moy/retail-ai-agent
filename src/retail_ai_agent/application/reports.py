from dataclasses import replace
from time import time

from retail_ai_agent.application.errors import AccessDenied, InvalidRequest
from retail_ai_agent.application.ports import ReportStore
from retail_ai_agent.domain.models import Actor, PendingDeletion, Report


class ReportService:
    def __init__(self, store: ReportStore, actor: Actor):
        self.store = store
        self.actor = actor

    def search(
        self, text: str = "", conversation_id: str = "", *, report_id: str = ""
    ) -> list[Report]:
        text = text.casefold()
        candidates = [self.get(report_id)] if report_id else self.store.list(self.actor.user_id)
        return [
            report
            for report in candidates
            if report.scope == self.actor.scope
            and (not conversation_id or report.conversation_id == conversation_id)
            and (text in report.question.casefold() or text in report.body.casefold())
        ]

    def get(self, report_id: str) -> Report:
        report = self.store.get(report_id)
        if (
            report is None
            or report.owner_id != self.actor.user_id
            or report.scope != self.actor.scope
        ):
            raise AccessDenied("Report not found or not accessible.")
        return report

    def save(
        self, run_id: str, conversation_id: str, question: str, sql: list[str], body: str
    ) -> Report:
        if not sql or not body.strip():
            raise InvalidRequest("A report cannot be saved without analysis results.")
        # A run ID is also an idempotency key when a graph node is replayed.
        return self.store.save(
            Report(
                id=run_id,
                owner_id=self.actor.user_id,
                conversation_id=conversation_id,
                scope=self.actor.scope,
                question=question,
                sql=tuple(sql),
                body=body,
                created_at=time(),
            )
        )

    def prepare(
        self, action_id: str, conversation_id: str, reports: list[Report]
    ) -> PendingDeletion:
        existing = self.store.pending(action_id)
        if existing:
            return self.pending(action_id, conversation_id)
        for report in reports:
            self.get(report.id)
        return self.store.prepare(
            PendingDeletion(
                id=action_id,
                owner_id=self.actor.user_id,
                conversation_id=conversation_id,
                scope=self.actor.scope,
                report_ids=tuple(report.id for report in reports),
                expires_at=time() + 600,
            )
        )

    def pending(self, action_id: str, conversation_id: str) -> PendingDeletion:
        action = self.store.pending(action_id)
        if (
            action is None
            or action.owner_id != self.actor.user_id
            or action.conversation_id != conversation_id
            or action.scope != self.actor.scope
        ):
            raise AccessDenied("Confirmation not found or not accessible.")
        return action

    def reselect(
        self, action_id: str, conversation_id: str, new_action_id: str, reports: list[Report]
    ) -> None:
        current = self.pending(action_id, conversation_id)
        if current.status != "pending" or current.expires_at <= time():
            raise InvalidRequest("This preview is no longer active. Request a new preview.")
        if not reports:
            raise InvalidRequest("Select at least one report or cancel the deletion.")
        updated = replace(current, id=new_action_id, report_ids=tuple(r.id for r in reports))
        self.store.replace_deletion(current, updated)

    def confirm(self, action_id: str, conversation_id: str, approved: bool) -> int:
        if type(approved) is not bool:
            raise InvalidRequest("Explicit confirmation or cancellation is required.")
        action = self.pending(action_id, conversation_id)
        if action.status != "pending":
            return action.deleted_count
        if approved and action.expires_at < time():
            raise InvalidRequest("The confirmation has expired. Cancel it and repeat the request.")
        if approved:
            for report_id in action.report_ids:
                self.get(report_id)
        return self.store.finish_deletion(action, approved)
