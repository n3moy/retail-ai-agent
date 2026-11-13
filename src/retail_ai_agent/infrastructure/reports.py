import os
import re
import tempfile
from dataclasses import replace
from pathlib import Path
from threading import RLock
from time import time
from typing import TypedDict

from pydantic import TypeAdapter

from retail_ai_agent.application.errors import InvalidRequest
from retail_ai_agent.domain.models import PendingDeletion, Report


class ReportMetadata(TypedDict):
    owner_id: str
    conversation_id: str
    scope: tuple[str, ...]
    question: str
    sql: tuple[str, ...]
    created_at: float


class ReportIndex(TypedDict):
    reports: dict[str, ReportMetadata]
    deletions: dict[str, PendingDeletion]


class FileReportStore:
    """Plain Markdown bodies and one JSON index, for a single CLI process."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.index_path = directory / "index.json"
        self.lock = RLock()
        self.adapter = TypeAdapter(ReportIndex)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self.index_path.exists():
            self._write_index({"reports": {}, "deletions": {}})
        self._cleanup_deleted(self._read_index())

    def _report_path(self, report_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", report_id):
            raise InvalidRequest("Invalid report ID.")
        path = self.directory / f"{report_id}.md"
        if path.is_symlink():
            raise InvalidRequest("The report file must not be a symbolic link.")
        return path

    def _read_index(self) -> ReportIndex:
        index = self.adapter.validate_json(self.index_path.read_bytes())
        # Validate paths before recovery can remove any files.
        for report_id in index["reports"]:
            self._report_path(report_id)
        for action_id, action in index["deletions"].items():
            if action.id != action_id or action.status not in {"pending", "deleted", "cancelled"}:
                raise InvalidRequest("The reports index.json file is corrupted.")
            for report_id in action.report_ids:
                self._report_path(report_id)
        return index

    def _write(self, path: Path, text: str) -> None:
        descriptor, temporary = tempfile.mkstemp(dir=self.directory, prefix=".write-")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                output.write(text)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _write_index(self, index: ReportIndex) -> None:
        self._write(self.index_path, self.adapter.dump_json(index, indent=2).decode() + "\n")

    def check(self) -> None:
        """Validate metadata and probe write access without changing library contents."""
        with self.lock:
            self._read_index()
            with tempfile.TemporaryFile(dir=self.directory) as probe:
                probe.write(b"ready")
                probe.flush()
                os.fsync(probe.fileno())

    def _report(self, report_id: str, metadata: ReportMetadata) -> Report | None:
        try:
            body = self._report_path(report_id).read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        return Report(id=report_id, body=body, **metadata)

    def save(self, report: Report) -> Report:
        with self.lock:
            path = self._report_path(report.id)
            index = self._read_index()
            if metadata := index["reports"].get(report.id):
                existing = self._report(report.id, metadata)
                if existing is None:
                    raise InvalidRequest("The saved report file is missing.")
                return existing
            if any(
                action.status == "deleted" and report.id in action.report_ids
                for action in index["deletions"].values()
            ):
                raise InvalidRequest("This report has already been deleted. Create a new report.")
            self._write(path, report.body)
            index["reports"][report.id] = {
                "owner_id": report.owner_id,
                "conversation_id": report.conversation_id,
                "scope": report.scope,
                "question": report.question,
                "sql": report.sql,
                "created_at": report.created_at,
            }
            # Publish only after the complete Markdown file is on disk.
            self._write_index(index)
            return report

    def get(self, report_id: str) -> Report | None:
        with self.lock:
            self._report_path(report_id)
            metadata = self._read_index()["reports"].get(report_id)
            return self._report(report_id, metadata) if metadata else None

    def list(self, owner_id: str) -> list[Report]:
        with self.lock:
            reports = [
                report
                for report_id, metadata in self._read_index()["reports"].items()
                if metadata["owner_id"] == owner_id
                and (report := self._report(report_id, metadata)) is not None
            ]
            return sorted(reports, key=lambda report: report.created_at, reverse=True)

    def prepare(self, action: PendingDeletion) -> PendingDeletion:
        with self.lock:
            index = self._read_index()
            if existing := index["deletions"].get(action.id):
                return existing
            for report_id in action.report_ids:
                self._report_path(report_id)
            index["deletions"][action.id] = action
            self._write_index(index)
            return action

    def pending(self, action_id: str) -> PendingDeletion | None:
        with self.lock:
            return self._read_index()["deletions"].get(action_id)

    def replace_deletion(self, current: PendingDeletion, updated: PendingDeletion) -> None:
        with self.lock:
            index = self._read_index()
            plans = index["deletions"]
            if plans.get(current.id) != current or current.status != "pending":
                raise InvalidRequest("The preview has changed. Check the current selection.")
            if current.expires_at <= time():
                raise InvalidRequest("This preview has expired. Request a new preview.")
            for report_id in updated.report_ids:
                report = self.get(report_id)
                if (
                    report is None
                    or report.owner_id != current.owner_id
                    or report.scope != current.scope
                ):
                    raise InvalidRequest("A selected report is unavailable. Check the selection.")
            plans[current.id] = replace(current, status="cancelled")
            plans[updated.id] = updated
            self._write_index(index)

    def _cleanup_deleted(self, index: ReportIndex) -> None:
        # If a process stopped after committing the index, finish removing the bodies on restart.
        for action in index["deletions"].values():
            if action.status == "deleted":
                for report_id in action.report_ids:
                    if report_id not in index["reports"]:
                        self._report_path(report_id).unlink(missing_ok=True)

    def finish_deletion(self, action: PendingDeletion, approved: bool) -> int:
        with self.lock:
            index = self._read_index()
            current = index["deletions"].get(action.id)
            if current is None:
                raise InvalidRequest("Confirmation not found.")
            if current.status != "pending":
                self._cleanup_deleted(index)
                return current.deleted_count
            if current != action:
                raise InvalidRequest("The confirmation has changed. Repeat the request.")
            if approved:
                if current.expires_at < time():
                    raise InvalidRequest(
                        "The confirmation has expired. Cancel it and repeat the request."
                    )
                for report_id in current.report_ids:
                    report = self.get(report_id)
                    if (
                        report is None
                        or report.owner_id != current.owner_id
                        or report.scope != current.scope
                    ):
                        raise InvalidRequest(
                            "The set of reports has changed. Repeat the deletion request."
                        )
                for report_id in current.report_ids:
                    del index["reports"][report_id]
            count = len(current.report_ids) if approved else 0
            index["deletions"][current.id] = replace(
                current, status="deleted" if approved else "cancelled", deleted_count=count
            )
            # One atomic replacement commits both library visibility and the approval outcome.
            self._write_index(index)
            self._cleanup_deleted(index)
            return count
