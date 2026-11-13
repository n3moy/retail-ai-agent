import json
import os
import stat
import tempfile
import unittest
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Barrier
from unittest.mock import Mock, patch

from dishka import Provider, Scope, provide
from dishka.exceptions import GraphMissingFactoryError
from fakes import LocalAnalytics, gemini_config
from google.api_core.exceptions import Forbidden
from google.cloud.bigquery import Client as BigQueryClient
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from typer.testing import CliRunner

from retail_ai_agent.application.chat import ChatService
from retail_ai_agent.application.errors import InvalidRequest, ServiceUnavailable
from retail_ai_agent.application.ports import AgentRunner, Analytics, ReportStore
from retail_ai_agent.application.reports import ReportService
from retail_ai_agent.create_app import create_app
from retail_ai_agent.domain.models import Actor
from retail_ai_agent.infrastructure.bigquery import NO_RETRY, BigQueryAnalytics
from retail_ai_agent.infrastructure.local_state import LocalState
from retail_ai_agent.infrastructure.reports import FileReportStore
from retail_ai_agent.infrastructure.tracing import JsonlTrace
from retail_ai_agent.infrastructure.users import UserDirectory
from retail_ai_agent.ioc import make_app_container
from retail_ai_agent.presentation.cli import cli


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.users_file = self.root / "users.json"
        self.users_file.write_text(json.dumps({"a": {"role": "manager", "brands": ["A"]}}))
        self.config = gemini_config(
            google_cloud_project="unit-test",
            bigquery_dataset="unit-test.data",
            users_file=self.users_file,
            data_dir=self.root / "state",
            startup_timeout_seconds=3,
        )
        self.local = LocalAnalytics(Actor("a", "manager", ("A",)))
        self.addCleanup(self.local.connection.close)
        self.client = Mock()
        self.client.get_table.side_effect = self.local.bigquery_table
        self.client.list_tables.side_effect = self.local.bigquery_tables
        self.job = Mock()
        self.job.result.return_value = [{"ready": 1}]
        self.client.query.side_effect = [Mock(), self.job]
        self.model = Mock(model=self.config.gemini_model)
        self.client_factory = self.enterContext(
            patch(
                "retail_ai_agent.infrastructure.resources.bigquery.Client", return_value=self.client
            )
        )
        self.model_factory = self.enterContext(
            patch(
                "retail_ai_agent.infrastructure.resources.ChatGoogleGenerativeAI",
                return_value=self.model,
            )
        )
        self.runner = Mock()
        self.container = self.make_container()

    def make_container(self, user_id="a", *, real_runner=False, overrides=()):
        provider = Provider(scope=Scope.APP)
        if not real_runner:
            provider.provide(lambda: self.runner, provides=AgentRunner, override=True)
        container = make_app_container(self.config, user_id, provider, *overrides)
        self.addCleanup(container.close)
        return container

    def assert_closed(self, store):
        self.client.close.assert_called_once()
        # The file store holds no open handles and needs no close lifecycle.
        self.assertEqual(store.list("a"), [])

    def test_checks_run_before_serving_and_all_resources_close(self):
        store = self.container.get(FileReportStore)
        self.container.get(BaseChatModel)
        with create_app(self.container) as app:
            self.assertIs(app.runner, self.runner)
            self.assertIs(self.container.get(ReportService).store, store)
            self.assertIs(self.container.get(BigQueryAnalytics).client, self.client)
            self.assertIs(self.container.get(Analytics), self.container.get(BigQueryAnalytics))
            self.assertIs(self.container.get(ReportStore), store)
            self.assertEqual(self.client.query.call_count, 2)
            self.model.client.models.generate_content.assert_not_called()
            self.model.invoke.assert_not_called()
            self.assertTrue(store.index_path.is_file())
            saver = self.container.get(BaseCheckpointSaver)
            self.assertIsInstance(saver, InMemorySaver)
            self.assertIs(saver, self.container.get(BaseCheckpointSaver))
            self.assertTrue(self.container.get(JsonlTrace).path.exists())
        self.assert_closed(store)
        self.model.client.close.assert_called_once()

    def test_network_probes_use_configured_dataset_and_bounded_requests(self):
        pii_key = self.container.get(LocalState).pseudonym_key()
        with create_app(self.container):
            pass
        dry, execution = self.client.query.call_args_list
        self.assertTrue(dry.kwargs["job_config"].dry_run)
        for table in ("products", "orders", "order_items", "users"):
            self.assertIn(f"unit-test.data.{table}", dry.args[0])
        self.assertNotIn("@pii_salt", dry.args[0])
        self.assertNotIn(pii_key, dry.args[0])
        self.assertEqual(execution.args[0], "SELECT 1 AS ready")
        for call in (dry, execution, self.job.result.call_args):
            self.assertIs(call.kwargs["retry"], NO_RETRY)
            self.assertIsNone(call.kwargs["job_retry"])
            self.assertGreater(call.kwargs["timeout"], 0)
            self.assertLessEqual(call.kwargs["timeout"], 3)
        self.model_factory.assert_not_called()

    def test_real_container_wires_langgraph_and_sdk_without_network(self):
        self.container = self.make_container(real_runner=True)
        model = ChatGoogleGenerativeAI(
            model=self.config.gemini_model, api_key="unit-test", vertexai=False
        )
        self.model_factory.return_value = model
        with patch.object(
            model.client.models,
            "generate_content",
            side_effect=AssertionError("Startup must not call Gemini"),
        ) as generate:
            with create_app(self.container) as app:
                self.assertIs(app.runner.reports, self.container.get(ReportService))
                self.assertIs(
                    app.runner.graph.checkpointer, self.container.get(BaseCheckpointSaver)
                )
            generate.assert_not_called()
        self.assertEqual(list(self.config.data_dir.rglob("*.sqlite3")), [])

    def test_bigquery_permission_failure_prevents_serving_and_redacts_error(self):
        self.client.query.side_effect = Forbidden("secret credential")
        store = self.container.get(FileReportStore)
        with (
            self.assertRaisesRegex(ServiceUnavailable, "BigQuery") as error,
            create_app(self.container),
        ):
            self.fail("Must not serve before readiness")
        self.assertNotIn("secret", str(error.exception))
        self.model.client.models.generate_content.assert_not_called()
        self.assert_closed(store)
        self.model_factory.assert_not_called()

    def test_metadata_failure_prevents_queries_and_model_calls(self):
        self.client.get_table.side_effect = Forbidden("private metadata payload")
        with (
            self.assertRaisesRegex(ServiceUnavailable, "BigQuery") as error,
            create_app(self.container),
        ):
            self.fail("Must not serve with a missing schema")
        self.assertNotIn("private metadata payload", str(error.exception))
        self.client.query.assert_not_called()
        self.model.client.models.generate_content.assert_not_called()

    def test_unknown_user_does_not_read_metadata(self):
        self.container = self.make_container(user_id="missing")
        with self.assertRaises(InvalidRequest), create_app(self.container):
            self.fail("Unknown actor")
        self.client.get_table.assert_not_called()

    def test_bigquery_result_timeout_attempts_cancellation(self):
        self.job.result.side_effect = TimeoutError("secret")
        with (
            self.assertRaisesRegex(ServiceUnavailable, "BigQuery"),
            create_app(self.container),
        ):
            self.fail("Must not serve on a timeout")
        self.job.cancel.assert_called_once_with(retry=NO_RETRY, timeout=2)
        self.model.client.models.generate_content.assert_not_called()

    def test_partial_initialization_failure_closes_previously_opened_resources(self):
        self.container = self.make_container(real_runner=True)
        self.container.get(FileReportStore)
        self.model_factory.side_effect = RuntimeError("secret API key")
        with (
            self.assertRaisesRegex(ServiceUnavailable, "Gemini") as error,
            create_app(self.container),
        ):
            self.fail("Construction must fail")
        self.assertNotIn("secret", str(error.exception))
        self.client.close.assert_called_once()

    def test_trace_failure_prevents_remote_probes(self):
        self.container.get(LocalState).trace_path.mkdir()
        with (
            self.assertRaisesRegex(ServiceUnavailable, "Local storage"),
            create_app(self.container),
        ):
            self.fail("Trace must be writable")
        self.client.query.assert_not_called()
        self.model.client.models.generate_content.assert_not_called()

    def test_unknown_user_fails_before_creating_resources(self):
        self.container = self.make_container(user_id="unknown")
        with self.assertRaises(InvalidRequest), create_app(self.container):
            self.fail("Unknown user")
        self.client_factory.assert_not_called()
        self.model_factory.assert_not_called()
        self.assertFalse(self.config.data_dir.exists())

    def test_consumer_exception_is_preserved_and_resources_close(self):
        store = self.container.get(FileReportStore)
        self.container.get(BaseChatModel)
        with self.assertRaisesRegex(ValueError, "consumer"), create_app(self.container):
            raise ValueError("consumer")
        self.assert_closed(store)
        self.model.client.close.assert_called_once()

    def test_resource_provider_can_be_overridden_without_patching_composition(self):
        replacement = Mock()
        replacement.get_table.side_effect = self.local.bigquery_table
        replacement.list_tables.side_effect = self.local.bigquery_tables
        replacement.query.side_effect = [Mock(), self.job]

        class TestProvider(Provider):
            @provide(scope=Scope.APP, override=True)
            def client(self) -> Iterator[BigQueryClient]:
                with closing(replacement):
                    yield replacement

        self.container = self.make_container(overrides=(TestProvider(),))
        with create_app(self.container):
            self.assertIs(self.container.get(BigQueryAnalytics).client, replacement)
        self.client_factory.assert_not_called()
        replacement.close.assert_called_once()

    def test_invalid_graph_fails_before_opening_resources(self):
        class MissingDependency:
            pass

        class BrokenProvider(Provider):
            @provide(scope=Scope.APP, override=True)
            def chat(self, missing: MissingDependency) -> ChatService:
                raise AssertionError("A broken factory must never run")

        with self.assertRaises(GraphMissingFactoryError):
            self.make_container(overrides=(BrokenProvider(),))
        self.client_factory.assert_not_called()
        self.model_factory.assert_not_called()
        self.assertFalse(self.config.data_dir.exists())

    def test_restart_creates_fresh_memory_and_preserves_reports(self):
        with create_app(self.container):
            first = self.container.get(ReportService)
            memory = self.container.get(BaseCheckpointSaver)
            report = first.save("run", "chat", "Sales", ["SELECT 1"], "Total: 10")
        self.client.query.side_effect = [Mock(), self.job]
        self.container = self.make_container()
        with create_app(self.container):
            second = self.container.get(ReportService)
            self.assertIsNot(first, second)
            self.assertIsNot(memory, self.container.get(BaseCheckpointSaver))
            self.assertEqual(second.search(), [report])

    def test_cli_exits_before_prompt_when_dependency_is_unavailable(self):
        self.client.query.side_effect = Forbidden("secret credential")
        with (
            patch(
                "retail_ai_agent.presentation.cli.Config.model_validate", return_value=self.config
            ),
            patch(
                "retail_ai_agent.presentation.cli.make_app_container", return_value=self.container
            ),
            patch("builtins.input") as prompt,
        ):
            result = CliRunner().invoke(cli, [])
        self.assertEqual(result.exit_code, 1)
        prompt.assert_not_called()
        self.assertIn("BigQuery", result.stderr)
        self.assertNotIn("secret", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_corrupt_report_index_prevents_network_and_preserves_files(self):
        directory = self.container.get(LocalState).reports_path
        directory.mkdir()
        index = directory / "index.json"
        index.write_text("broken metadata")
        with (
            self.assertRaisesRegex(ServiceUnavailable, "report files"),
            create_app(self.container),
        ):
            self.fail("Invalid metadata")
        self.assertEqual(index.read_text(), "broken metadata")
        self.client_factory.assert_not_called()
        self.model_factory.assert_not_called()

    def test_report_write_failure_prevents_remote_probes(self):
        store = self.container.get(FileReportStore)
        with (
            patch.object(store, "check", side_effect=PermissionError("private path")),
            self.assertRaisesRegex(ServiceUnavailable, "Local storage"),
            create_app(self.container),
        ):
            self.fail("Reports must be writable")
        self.client.query.assert_not_called()
        self.model.client.models.generate_content.assert_not_called()


class LocalConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_pseudonym_key_is_private_stable_and_never_replaced_when_corrupt(self):
        state = LocalState(self.root)
        key = state.pseudonym_key()
        self.assertEqual(len(key), 64)
        self.assertEqual(LocalState(self.root).pseudonym_key(), key)
        path = self.root / "pseudonym.key"
        if os.name == "posix":  # Windows uses directory ACLs for user-private storage.
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        path.write_text("broken")
        with self.assertRaises(InvalidRequest):
            state.pseudonym_key()
        self.assertEqual(path.read_text(), "broken")

    def test_concurrent_startups_publish_one_complete_key(self):
        state = LocalState(self.root)
        barrier = Barrier(4)

        def load(_):
            barrier.wait(timeout=5)
            return state.pseudonym_key()

        with ThreadPoolExecutor(max_workers=4) as pool:
            keys = list(pool.map(load, range(4)))
        self.assertEqual(len(set(keys)), 1)
        self.assertEqual(len(keys[0]), 64)
        self.assertEqual(list(self.root.glob(".pseudonym-*")), [])

    def test_all_user_profiles_are_validated_before_serving(self):
        path = self.root / "users.json"
        for bad in (
            [],
            {"a": None},
            {"a": {"role": "admin"}},
            {"a": {"role": "ceo", "brands": [2]}},
        ):
            with self.subTest(bad=bad):
                path.write_text(json.dumps(bad))
                with self.assertRaises(InvalidRequest):
                    UserDirectory(path)
        path.write_text(
            json.dumps(
                {
                    "a": {"role": "manager", "brands": ["A"]},
                    "b": {"role": "ceo", "brands": ["B", "A"]},
                }
            )
        )
        directory = UserDirectory(path)
        self.assertEqual(directory.actor("a").brands, ("A",))
        self.assertEqual(directory.actor("b").role, "ceo")
