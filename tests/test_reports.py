import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from time import time
from unittest.mock import patch

from pydantic import ValidationError

from retail_ai_agent.application.errors import AccessDenied, InvalidRequest
from retail_ai_agent.application.reports import ReportService
from retail_ai_agent.domain.models import Actor
from retail_ai_agent.infrastructure.reports import FileReportStore


class ReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.store = FileReportStore(Path(temp.name) / "reports")
        self.actor = Actor("a", "manager", ("A",))
        self.reports = self.service(self.actor)

    def service(self, actor):
        return ReportService(self.store, actor)

    def test_search_preserves_distinct_contacts_in_original_questions(self):
        first = self.reports.save("first", "chat", "alice@example.com", ["SELECT 1"], "Total: 10")
        self.reports.save("second", "chat", "bob@example.com", ["SELECT 1"], "Total: 20")
        self.assertEqual(self.reports.search("ALICE@example.com"), [first])

    def save(self, run="run", conversation="chat"):
        return self.reports.save(run, conversation, "Sales", ["SELECT 1"], "Total: 10")

    def prepare(self):
        return self.reports.prepare("delete", "chat", self.reports.search())

    def test_saving_and_preparing_are_idempotent(self):
        self.assertEqual(self.save().id, self.save().id)
        first = self.prepare()
        self.save("later")
        self.assertEqual(self.prepare(), first)

    def test_ceo_cannot_read_or_delete_another_users_reports(self):
        report, action = self.save(), self.prepare()
        ceo = self.service(Actor("ceo", "ceo"))
        self.assertEqual(ceo.search(), [])
        with self.assertRaises(AccessDenied):
            ceo.get(report.id)
        with self.assertRaises(AccessDenied):
            ceo.confirm(action.id, "chat", True)

    def test_confirmation_binds_owner_conversation_scope_and_exact_snapshot(self):
        first = self.save()
        action = self.prepare()
        later = self.save("later")
        for service, conversation in [
            (self.reports, "other"),
            (self.service(Actor("a", "manager", ("B",))), "chat"),
        ]:
            with self.assertRaises(AccessDenied):
                service.confirm(action.id, conversation, True)
        self.assertEqual(self.reports.confirm(action.id, "chat", True), 1)
        self.assertIsNone(self.store.get(first.id))
        self.assertIsNotNone(self.store.get(later.id))
        self.assertEqual(self.reports.confirm(action.id, "chat", True), 1)

    def test_cancellation_cannot_later_be_approved(self):
        report = self.save()
        action = self.prepare()
        self.assertEqual(self.reports.confirm(action.id, "chat", False), 0)
        self.assertEqual(self.reports.confirm(action.id, "chat", True), 0)
        self.assertIsNotNone(self.store.get(report.id))

    def test_expired_confirmation_and_non_boolean_are_rejected(self):
        self.save()
        expired = replace(self.prepare(), id="expired", expires_at=time() - 1)
        self.store.prepare(expired)
        with self.assertRaises(InvalidRequest):
            self.reports.confirm(expired.id, "chat", True)
        self.assertEqual(self.reports.confirm(expired.id, "chat", False), 0)
        for value in (None, "yes", 1):
            with self.assertRaises(InvalidRequest):
                self.reports.confirm("delete", "chat", value)

    def test_failed_index_update_keeps_reports_and_approval_unchanged(self):
        report = self.save()
        action = self.prepare()
        with (
            patch("retail_ai_agent.infrastructure.reports.os.replace", side_effect=OSError("disk")),
            self.assertRaises(OSError),
        ):
            self.reports.confirm(action.id, "chat", True)
        self.assertIsNotNone(self.store.get(report.id))
        self.assertEqual(self.store.pending(action.id).status, "pending")
        self.assertEqual(list(self.store.directory.glob(".write-*")), [])

    def test_changed_snapshot_never_partially_deletes(self):
        first, second = self.save(), self.save("second")
        action = self.prepare()
        (self.store.directory / f"{second.id}.md").unlink()
        with self.assertRaises(AccessDenied):
            self.reports.confirm(action.id, "chat", True)
        self.assertIsNotNone(self.store.get(first.id))

    def test_markdown_is_arbitrary_utf8_and_metadata_is_separate(self):
        body = "# Sales\n\n| Period | Revenue (€) |\n| --- | --- |\n| January | 10 |\n\n**Finding:** sales rose.\n"
        report = self.reports.save("sales", "chat", "Sales", ["SELECT 1"], body)
        self.assertEqual((self.store.directory / "sales.md").read_text(), body)
        metadata = json.loads(self.store.index_path.read_text())["reports"]["sales"]
        self.assertEqual(metadata["owner_id"], "a")
        self.assertEqual(metadata["sql"], ["SELECT 1"])
        self.assertNotIn("body", metadata)
        self.assertEqual(FileReportStore(self.store.directory).get(report.id), report)
        self.assertEqual(
            {path.name for path in self.store.directory.iterdir()}, {"sales.md", "index.json"}
        )

    def test_manual_markdown_edit_is_visible_and_replay_does_not_overwrite_it(self):
        self.save()
        body = "# Updated finding\n\nFree-form text with no required sections."
        (self.store.directory / "run.md").write_text(body)
        self.assertEqual(self.reports.get("run").body, body)
        self.assertEqual([report.id for report in self.reports.search("Updated")], ["run"])
        self.assertEqual(self.save().body, body)

    def test_interrupted_file_cleanup_recovers_without_repeating_deletion(self):
        first, second = self.save(), self.save("second")
        action = self.prepare()
        unlink = Path.unlink

        def fail_last(path, *args, **kwargs):
            if path.name == "run.md":
                raise OSError("interrupted cleanup")
            return unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", fail_last), self.assertRaises(OSError):
            self.reports.confirm(action.id, "chat", True)
        self.assertIsNone(self.store.get(first.id))
        self.assertIsNone(self.store.get(second.id))
        self.assertFalse((self.store.directory / "second.md").exists())
        self.assertTrue((self.store.directory / "run.md").exists())
        self.assertEqual(self.store.pending(action.id).status, "deleted")
        self.store = FileReportStore(self.store.directory)
        self.reports = self.service(self.actor)
        self.assertEqual(list(self.store.directory.glob("*.md")), [])
        self.assertEqual(self.reports.confirm(action.id, "chat", False), 2)
        with self.assertRaises(InvalidRequest):
            self.save()

    def test_failed_save_is_not_published_and_can_be_retried(self):
        with (
            patch.object(self.store, "_write_index", side_effect=OSError("disk")),
            self.assertRaises(OSError),
        ):
            self.save()
        self.assertEqual(self.reports.search(), [])
        self.assertEqual(self.save().body, "Total: 10")

    def test_corrupt_index_is_not_overwritten(self):
        self.save()
        for content in ("{", '{"reports": [], "deletions": {}}'):
            with self.subTest(content=content):
                self.store.index_path.write_text(content)
                with self.assertRaises(ValidationError):
                    FileReportStore(self.store.directory)
                self.assertEqual(self.store.index_path.read_text(), content)
                self.assertEqual((self.store.directory / "run.md").read_text(), "Total: 10")

    def test_report_ids_cannot_escape_directory_or_read_symlinks(self):
        outside = self.store.directory.parent / "outside.md"
        outside.write_text("private")
        for report_id in ("../outside", str(outside), "a/b", ".", ""):
            with self.subTest(report_id=report_id):
                with self.assertRaises(InvalidRequest):
                    self.save(report_id)
                with self.assertRaises(InvalidRequest):
                    self.store.get(report_id)
        try:
            (self.store.directory / "link.md").symlink_to(outside)
        except OSError as error:
            if getattr(error, "winerror", None) == 1314:
                self.skipTest(
                    "Creating a Windows symlink requires Developer Mode or the symlink privilege"
                )
            raise
        with self.assertRaises(InvalidRequest):
            self.save("link")
        self.assertEqual(outside.read_text(), "private")
