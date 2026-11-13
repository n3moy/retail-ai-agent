import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx2 as httpx
from dishka import Provider, Scope
from fakes import LocalAnalytics
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from retail_ai_agent.application.models import ChatRequest
from retail_ai_agent.application.ports import Analytics
from retail_ai_agent.config import Config
from retail_ai_agent.create_app import create_app
from retail_ai_agent.domain.models import Actor
from retail_ai_agent.ioc import make_app_container


def completion(text=None, *, tool=None, arguments=None):
    message = {"role": "assistant", "content": text}
    if tool:
        message["tool_calls"] = [
            {
                "id": f"call_{tool}",
                "type": "function",
                "function": {"name": tool, "arguments": json.dumps(arguments)},
            }
        ]
    return {
        "id": "test-completion",
        "object": "chat.completion",
        "created": 0,
        "model": "openai/gpt-4.1-mini",
        "choices": [
            {"index": 0, "message": message, "finish_reason": "tool_calls" if tool else "stop"}
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }


class ModelConfigTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {}, clear=True))

    def config(self, **kwargs):
        return Config(_env_file=None, google_cloud_project="test", **kwargs)

    def test_default_gemini_provider_reads_model_from_dotenv(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("GEMINI_MODEL=gemini-3.1-flash-lite\n")
            config = Config(
                _env_file=env_file, google_ai_studio_api_key="test", google_cloud_project="test"
            )
        self.assertEqual(config.model_provider, "gemini")
        self.assertEqual(config.model_name, "gemini-3.1-flash-lite")
        self.assertIsNone(config.openrouter_api_key)

    def test_gemini_requires_an_explicit_nonblank_model(self):
        for settings in ({}, {"gemini_model": ""}, {"gemini_model": "  "}):
            with self.subTest(settings=settings), self.assertRaises(ValidationError) as error:
                self.config(google_ai_studio_api_key="test", **settings)
            self.assertEqual(error.exception.errors()[0]["loc"], ("gemini_model",))

    def test_openrouter_env_configuration_does_not_require_google_model_credentials(self):
        with patch.dict(
            os.environ,
            {
                "MODEL_PROVIDER": "openrouter",
                "OPENROUTER_API_KEY": "private-openrouter-key",
                "OPENROUTER_MODEL": "vendor/test-model",
                "OPENROUTER_BASE_URL": "https://example.com/api/v1",
                "GOOGLE_CLOUD_PROJECT": "billing-project",
            },
        ):
            config = Config.model_validate({"_env_file": None})
        self.assertIsNone(config.google_ai_studio_api_key)
        self.assertEqual(config.model_name, "vendor/test-model")
        self.assertEqual(config.openrouter_base_url, "https://example.com/api/v1")
        self.assertEqual(config.google_cloud_project, "billing-project")
        self.assertNotIn("private-openrouter-key", repr(config))

    def test_missing_or_blank_selected_key_is_a_field_error(self):
        for provider, field in (
            ("gemini", "google_ai_studio_api_key"),
            ("openrouter", "openrouter_api_key"),
        ):
            for value in (None, "", "   "):
                with self.subTest(provider=provider, value=value):
                    with self.assertRaises(ValidationError) as error:
                        self.config(model_provider=provider, **{field: value})
                    self.assertEqual(error.exception.errors()[0]["loc"], (field,))

    def test_unknown_provider_does_not_silently_fall_back(self):
        with self.assertRaises(ValidationError) as error:
            self.config(model_provider="unknown")
        self.assertEqual(error.exception.errors()[0]["loc"], ("model_provider",))


class OpenRouterAdapterTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        users = self.root / "users.json"
        users.write_text(json.dumps({"a": {"role": "manager", "brands": ["A"]}}))
        self.config = Config(
            _env_file=None,
            model_provider="openrouter",
            google_ai_studio_api_key=None,
            openrouter_api_key="unit-test-openrouter",
            google_cloud_project="unit-test",
            users_file=users,
            data_dir=self.root / "state",
            startup_timeout_seconds=3,
        )
        self.analytics = LocalAnalytics(Actor("a", "manager", ("A",)))
        self.addCleanup(self.analytics.connection.close)
        self.bigquery = Mock()
        self.bigquery.get_table.side_effect = self.analytics.bigquery_table
        self.bigquery.list_tables.side_effect = self.analytics.bigquery_tables
        job = Mock()
        job.result.return_value = [{"ready": 1}]
        self.bigquery.query.return_value = job
        self.enterContext(
            patch(
                "retail_ai_agent.infrastructure.resources.bigquery.Client",
                return_value=self.bigquery,
            )
        )
        self.gemini = self.enterContext(
            patch("retail_ai_agent.infrastructure.resources.ChatGoogleGenerativeAI")
        )
        self.requests = []
        self.responses = []
        # Exercise real LangChain and OpenAI serialization down to the HTTP boundary.
        self.enterContext(patch.object(httpx.Client, "send", side_effect=self.send))

    def send(self, request, **kwargs):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        status, payload = response if isinstance(response, tuple) else (200, response)
        return httpx.Response(status, json=payload, request=request)

    def container(self):
        overrides = Provider(scope=Scope.APP)
        overrides.provide(lambda: self.analytics, provides=Analytics, override=True)
        container = make_app_container(self.config, "a", overrides)
        self.addCleanup(container.close)
        return container

    def events(self, name):
        records = [
            json.loads(line)
            for line in (self.config.data_dir / "trace.jsonl").read_text().splitlines()
        ]
        return [record for record in records if record["event"] == name]

    def assert_closed(self, model):
        self.assertTrue(model.root_client.is_closed())
        self.assertTrue(model.root_async_client.is_closed())
        self.gemini.assert_not_called()

    def test_startup_sql_report_usage_and_cleanup_through_actual_sdk(self):
        self.responses = [
            completion('{"intent":"REPORT"}'),
            completion(
                tool="sql_db_query",
                arguments={"query": "SELECT SUM(sale_price) AS total FROM order_items"},
            ),
            completion(tool="save_report", arguments={"body": "Sales: 10"}),
            completion("Sales: 10"),
        ]
        container = self.container()
        with create_app(container) as app:
            model = container.get(BaseChatModel)
            self.assertIsInstance(model, ChatOpenAI)
            self.assertEqual(self.requests, [])
            result = app.ask(
                ChatRequest("Create a sales report. Contact: private@example.com", "chat")
            )
        self.assertEqual(result.status, "complete", result.answer)
        self.assertTrue(result.report_id, result.answer)
        self.assertEqual(len(self.requests), 4)
        self.assert_closed(model)
        self.bigquery.close.assert_called_once()
        payloads = [json.loads(request.content) for request in self.requests]
        self.assertNotIn("private@example.com", json.dumps(payloads))
        self.assertFalse(payloads[0].get("tools"))
        for request, payload in zip(self.requests[1:], payloads[1:], strict=True):
            self.assertEqual(str(request.url), "https://openrouter.ai/api/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer unit-test-openrouter")
            self.assertEqual(payload["tools"], payloads[1]["tools"])
            self.assertEqual(payload["model"], self.config.openrouter_model)
            self.assertNotIn("max_retries", payload)
            self.assertEqual(
                payload["max_completion_tokens"],
                self.config.max_output_tokens,
            )
            for seconds in request.extensions["timeout"].values():
                self.assertGreater(seconds, 0)
                self.assertLessEqual(seconds, self.config.model_timeout_seconds)
        functions = {t["function"]["name"]: t["function"] for t in payloads[1]["tools"]}
        self.assertEqual(set(functions["sql_db_query"]["parameters"]["properties"]), {"query"})
        self.assertEqual(set(functions["write_todos"]["parameters"]["properties"]), {"todos"})
        self.assertEqual(
            set(functions),
            {
                "write_todos",
                "sql_db_query",
                "save_report",
                "list_reports",
                "show_report",
                "delete_reports",
            },
        )
        self.assertNotIn("runtime", str(payloads[1]["tools"]))
        self.assertTrue(
            any(
                m["role"] == "tool" and m["tool_call_id"] == "call_sql_db_query"
                for m in payloads[-1]["messages"]
            )
        )
        self.assertEqual(self.events("run_started")[0]["model"], self.config.openrouter_model)
        self.assertEqual(self.events("run_started")[0]["provider"], "openrouter")
        self.assertEqual(self.events("model_response")[0]["usage"]["total_tokens"], 30)
        self.assertAlmostEqual(self.events("run_finished")[0]["estimated_usd"], 0.0024)

    def test_transient_errors_are_retried_by_budgeted_middleware_without_sdk_retries(self):
        for failure in (
            (429, {"error": {"message": "private provider payload", "code": None}}),
            (503, {"error": {"message": "private provider payload", "code": None}}),
            httpx.ReadTimeout("private provider payload"),
            httpx.ConnectError("private provider payload"),
        ):
            with self.subTest(failure=failure):
                self.requests.clear()
                self.responses = [
                    failure,
                    completion('{"intent":"ANALYSIS"}'),
                    completion(tool="list_reports", arguments={}),
                    completion("No reports found."),
                ]
                container = self.container()
                with (
                    create_app(container) as app,
                    patch("langchain.agents.middleware.model_retry.time.sleep") as sleep,
                ):
                    result = app.ask(ChatRequest("Find my reports", "chat"))
                self.assertEqual(result.status, "complete", result.answer)
                self.assertEqual(len(self.requests), 4)
                sleep.assert_called_once_with(1)
                self.assertEqual(self.events("run_finished")[-1]["model_calls"], 4)
                self.assertNotIn(
                    "private provider payload", (self.config.data_dir / "trace.jsonl").read_text()
                )

    def test_report_command_uses_the_same_tool_loop_and_closes_sdk_clients(self):
        self.responses = [
            completion(tool="list_reports", arguments={}),
            completion("No reports found."),
        ]
        container = self.container()
        with create_app(container) as app:
            model = container.get(BaseChatModel)
            self.assertEqual(self.requests, [])
            result = app.ask(ChatRequest("/reports", "chat", action="list"))
            self.assertEqual(result.status, "complete", result.answer)
            self.assertEqual(len(self.requests), 2)
        self.assert_closed(model)
