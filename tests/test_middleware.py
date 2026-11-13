import json
import unittest
from unittest.mock import Mock, patch

import httpx
from fakes import LocalAnalytics, ScriptedModel, tool_call, toolkit_tools
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime

from retail_ai_agent.application.budget import Budget
from retail_ai_agent.application.errors import (
    AccessDenied,
    BudgetExceeded,
    QueryFailed,
    ServiceUnavailable,
    UnsafeQuery,
)
from retail_ai_agent.application.models import ChatRequest, QueryResult
from retail_ai_agent.config import Config
from retail_ai_agent.domain.models import Actor
from retail_ai_agent.infrastructure.agent.context import RunContext
from retail_ai_agent.infrastructure.agent.middleware import create_middleware
from retail_ai_agent.infrastructure.agent.middleware.budget import (
    ModelBudgetMiddleware,
    estimate_input,
)
from retail_ai_agent.infrastructure.privacy import LocalSanitizer

SLEEP = "langchain.agents.middleware.model_retry.time.sleep"
QUERY = "SELECT COUNT(*) FROM products"


class MiddlewareTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(
            _env_file=None,
            google_ai_studio_api_key="test",
            gemini_model="gemini-3.5-flash-lite",
            google_cloud_project="test",
            model_provider="openrouter",
            openrouter_api_key="test",
            input_usd_per_million=0,
            output_usd_per_million=1,
            max_output_tokens=100,
        )
        self.context = RunContext("test", Budget(), ChatRequest("Question", "chat"))
        # Capture the events emitted by middleware without writing JSONL files.
        self.trace = Mock()
        self.local = LocalAnalytics(Actor("a", "manager", ("A",)))
        self.addCleanup(self.local.connection.close)
        self.analytics = Mock()
        self.analytics.execute.return_value = QueryResult(QUERY, ({"count": 10},))

    def agent(self, responses, **kwargs):
        self.model = ScriptedModel(responses=list(responses))
        return create_agent(
            self.model,
            middleware=create_middleware(
                self.config, LocalSanitizer(), self.trace, self.local.sql, self.analytics
            ),
            context_schema=RunContext,
            **kwargs,
        )

    def invoke(self, agent, question="products"):
        return agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            context=self.context,
            config={"max_concurrency": 1},
        )

    def events(self, name):
        return [call.kwargs for call in self.trace.record.call_args_list if call.args[0] == name]

    def test_google_count_failure_does_not_reserve_or_generate_and_429_can_retry(self):
        self.config.model_provider = "gemini"
        raw = httpx.Request("POST", "https://example.com/countTokens")

        def http_failure(code):
            return httpx.HTTPStatusError(
                "private backend diagnostics",
                request=raw,
                response=httpx.Response(code, request=raw),
            )

        target = "retail_ai_agent.infrastructure.agent.middleware.budget.count_gemini_input"
        with (
            patch(target, side_effect=http_failure(401)),
            self.assertRaises(ServiceUnavailable) as err,
        ):
            self.invoke(self.agent([AIMessage(content="unused")]))
        self.assertNotIn("private", str(err.exception))
        self.assertEqual(self.context.budget.spent_usd, 0)
        self.assertEqual(self.model.seen, [])
        with (
            patch(target, side_effect=[http_failure(429), http_failure(429), 20]),
            patch(SLEEP) as sleep,
        ):
            self.invoke(self.agent([AIMessage(content="OK")]))
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1, 2])
        self.assertEqual(self.context.budget.model_calls, 1)
        self.assertEqual(self.context.budget.reserved_usd, 0)
        self.assertNotIn("private backend diagnostics", str(self.trace.record.call_args_list))

    def test_retry_keeps_failed_reservation_and_reconciles_successful_usage(self):
        agent = self.agent([TimeoutError("private@example.com"), AIMessage(content="done")])
        with patch(SLEEP) as sleep:
            self.invoke(agent)
        sleep.assert_called_once_with(1)
        self.assertEqual(self.context.budget.model_calls, 2)
        # Failed attempt keeps its 100-token reserve; success bills 10 output tokens.
        self.assertAlmostEqual(self.context.budget.spent_usd, 110 / 1_000_000)
        self.assertEqual([e["attempt"] for e in self.events("model_request")], [0, 1])
        self.assertEqual(len(self.events("model_response")), 1)
        self.assertNotIn("private@example.com", str(self.trace.record.call_args_list))

    def test_retry_exhaustion_is_bounded_and_does_not_release_reservations(self):
        request = httpx.Request("POST", "https://example.com/generate")
        failure = httpx.HTTPStatusError("secret", request=request, response=httpx.Response(503))
        agent = self.agent([failure] * 3)
        with patch(SLEEP) as sleep, self.assertRaises(ServiceUnavailable):
            self.invoke(agent)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])
        self.assertEqual(len(self.model.seen), 3)
        self.assertEqual(self.context.budget.model_calls, 3)
        self.assertAlmostEqual(self.context.budget.spent_usd, 300 / 1_000_000)
        self.assertEqual([e["attempt"] for e in self.events("model_error")], [0, 1, 2])

    def test_budget_error_between_retries_propagates_without_another_provider_call(self):
        self.context.budget.max_usd = 150 / 1_000_000
        agent = self.agent([TimeoutError("secret"), AIMessage(content="must not run")])
        with patch(SLEEP) as sleep, self.assertRaises(BudgetExceeded):
            self.invoke(agent)
        sleep.assert_called_once_with(1)
        self.assertEqual(len(self.model.seen), 1)
        self.assertEqual(len(self.events("model_request")), 1)
        self.assertAlmostEqual(self.context.budget.spent_usd, 100 / 1_000_000)
        self.assertEqual(self.context.budget.model_calls, 1)

    def test_retry_does_not_sleep_past_deadline(self):
        self.context.budget.timeout = 0.5
        agent = self.agent([TimeoutError("secret")])
        with patch(SLEEP) as sleep, self.assertRaises(ServiceUnavailable):
            self.invoke(agent)
        sleep.assert_not_called()
        self.assertEqual(len(self.model.seen), 1)

    def test_permanent_provider_error_is_safe_and_not_retried(self):
        agent = self.agent([ValueError("private@example.com")])
        with patch(SLEEP) as sleep, self.assertRaises(ServiceUnavailable) as error:
            self.invoke(agent)
        sleep.assert_not_called()
        self.assertEqual(len(self.model.seen), 1)
        self.assertNotIn("private@example.com", str(error.exception))

    def test_invalid_response_is_rejected_and_its_measured_usage_is_accounted(self):
        agent = self.agent(
            [
                AIMessage(
                    content="unverified",
                    invalid_tool_calls=[{"id": "bad", "name": "sql_db_query", "args": "{"}],
                )
            ]
        )
        with patch(SLEEP) as sleep, self.assertRaisesRegex(ServiceUnavailable, "invalid"):
            self.invoke(agent)
        sleep.assert_not_called()
        self.assertEqual(self.events("model_response"), [])
        self.assertAlmostEqual(self.context.budget.model_usd, 10 / 1_000_000)
        self.assertEqual(self.context.budget.reserved_usd, 0)

    def test_messages_are_clean_before_tracing(self):
        agent = self.agent([AIMessage(content="private@example.com")])
        result = self.invoke(agent, "private@example.com")
        self.assertEqual(result["messages"][-1].text, "[REDACTED_EMAIL]")
        self.assertNotIn("private@example.com", str(self.model.seen))
        self.assertNotIn("private@example.com", str(self.trace.record.call_args_list))

    def test_tool_output_is_clean_before_next_model_call_and_trace(self):
        self.analytics.execute.return_value = QueryResult(
            QUERY, ({"value": "private@example.com"},)
        )
        agent = self.agent(
            [tool_call(QUERY), AIMessage(content="done")],
            tools=toolkit_tools(self.local.catalog),
        )
        self.invoke(agent)
        observations = [m for m in self.model.seen[-1] if isinstance(m, ToolMessage)]
        self.assertIn("[REDACTED_EMAIL]", observations[0].content)
        self.assertNotIn("private@example.com", str(self.trace.record.call_args_list))

    def test_whole_row_reference_is_masked_before_reaching_the_model(self):
        query = "SELECT u AS record FROM users u"
        self.analytics.execute.return_value = QueryResult(
            query, ({"record": {"id": 1, "first_name": "Private", "email": "private@example.com"}},)
        )
        self.invoke(
            self.agent(
                [tool_call(query), AIMessage(content="done")],
                tools=toolkit_tools(self.local.catalog),
            )
        )
        self.assertEqual(self.context.queries[0].rows, ({"record": "[REDACTED]"},))
        for surface in (str(self.model.seen), str(self.trace.record.call_args_list)):
            self.assertNotIn("Private", surface)
            self.assertNotIn("private@example.com", surface)

    def test_tool_error_text_is_cleaned_on_the_next_model_request(self):
        self.analytics.execute.side_effect = QueryFailed("Contact private@example.com")
        agent = self.agent(
            [tool_call(QUERY), AIMessage(content="Cannot query")],
            tools=toolkit_tools(self.local.catalog),
        )
        self.invoke(agent)
        observation = next(m for m in self.model.seen[-1] if isinstance(m, ToolMessage))
        self.assertIn("[REDACTED_EMAIL]", observation.content)
        self.assertNotIn("private@example.com", str(self.model.seen))

    def test_invalid_tool_arguments_are_retryable_and_count_towards_sql_limit(self):
        malformed = tool_call(QUERY, "bad")
        malformed.tool_calls[0]["args"]["extra"] = "not allowed"
        agent = self.agent(
            [malformed, tool_call(QUERY, "good"), AIMessage(content="done")],
            tools=toolkit_tools(self.local.catalog),
        )
        self.invoke(agent)
        self.assertEqual(self.context.budget.sql_attempts, 2)
        self.analytics.execute.assert_called_once()
        self.assertIsNone(self.context.stopped)
        errors = [m for m in self.model.seen[1] if isinstance(m, ToolMessage)]
        self.assertEqual(errors[0].tool_call_id, "bad")
        self.assertIn('"retryable": true', errors[0].content)

    def test_missing_tool_id_still_returns_a_safe_error_observation(self):
        call = tool_call(QUERY)
        call.tool_calls[0]["id"] = None
        self.invoke(
            self.agent([call, AIMessage(content="done")], tools=toolkit_tools(self.local.catalog))
        )
        error = next(m for m in self.model.seen[-1] if isinstance(m, ToolMessage))
        self.assertEqual(error.status, "error")
        self.assertIn("tool call ID is required", error.content)
        self.analytics.execute.assert_not_called()

    def test_unknown_tool_stops_run_but_every_pending_call_gets_an_observation(self):
        calls = tool_call(QUERY, "denied")
        calls.tool_calls[0]["name"] = "delete_reports"
        calls.tool_calls.extend(tool_call(QUERY, "later").tool_calls)
        agent = self.agent([calls], tools=toolkit_tools(self.local.catalog))
        with self.assertRaises(AccessDenied):
            self.invoke(agent)
        self.analytics.execute.assert_not_called()
        self.assertEqual(self.context.budget.sql_attempts, 0)
        self.assertEqual([e["call_id"] for e in self.events("tool_result")], ["denied", "later"])
        self.assertEqual(len(self.model.seen), 1)

    def test_retry_attempt_number_resets_for_next_model_call(self):
        agent = self.agent(
            [TimeoutError(), tool_call(QUERY), AIMessage(content="done")],
            tools=toolkit_tools(self.local.catalog),
        )
        with patch(SLEEP):
            self.invoke(agent)
        self.assertEqual([e["attempt"] for e in self.events("model_request")], [0, 1, 0])

    def test_attempt_timeout_is_capped_and_missing_usage_keeps_reservation(self):
        self.context.budget.timeout = 5
        request = ModelRequest(
            model=Mock(),
            messages=[HumanMessage(content="products")],
            runtime=Runtime(context=self.context),
            model_settings={"temperature": 0},
        )
        handler = Mock(return_value=ModelResponse(result=[AIMessage(content="done")]))
        ModelBudgetMiddleware(self.config, self.trace).wrap_model_call(request, handler)
        settings = handler.call_args.args[0].model_settings
        self.assertGreater(settings["timeout"], 0)
        self.assertLessEqual(settings["timeout"], 5)
        self.assertEqual(settings["temperature"], 0)
        self.assertAlmostEqual(self.context.budget.spent_usd, 100 / 1_000_000)

    def test_gemini_skips_generation_when_sdk_would_round_timeout_to_zero(self):
        self.config.model_provider = "gemini"
        self.config.model_timeout_seconds = 0.0005
        target = "retail_ai_agent.infrastructure.agent.middleware.budget.count_gemini_input"
        with patch(target, return_value=20), self.assertRaisesRegex(BudgetExceeded, "timed out"):
            self.invoke(self.agent([AIMessage(content="unused")]))
        self.assertEqual(self.context.budget.model_calls, 0)
        self.assertEqual(self.context.budget.reserved_usd, 0)
        self.assertEqual(self.model.seen, [])

    def test_final_call_after_detailed_report_fits_only_within_available_budget(self):
        self.config.input_usd_per_million = 10
        self.config.output_usd_per_million = 40
        self.config.max_output_tokens = 2048
        self.context.budget.model_usd = 0.4
        middleware = ModelBudgetMiddleware(self.config, self.trace)
        body = "Quarterly revenue fell by €100 compared with last year. " * 300
        call = AIMessage(
            content=[],
            tool_calls=[{"id": "save", "name": "save_report", "args": {"body": body}}],
            additional_kwargs={
                "function_call": {"name": "save_report", "arguments": json.dumps({"body": body})}
            },
            usage_metadata={"input_tokens": 4868, "output_tokens": 1649, "total_tokens": 6517},
        )
        request = ModelRequest(
            model=ScriptedModel(),
            system_message=SystemMessage("Retail analysis"),
            messages=[
                HumanMessage(content="Prepare a report"),
                tool_call(QUERY),
                ToolMessage("SQL rows " * 1100, tool_call_id="query"),
            ],
            tools=toolkit_tools(self.local.catalog),
            runtime=Runtime(context=self.context),
        )
        middleware.wrap_model_call(request, Mock(return_value=ModelResponse(result=[call])))
        after_save = request.override(
            messages=[
                *request.messages,
                call,
                ToolMessage("Report saved: abc", tool_call_id="save"),
            ]
        )
        self.assertAlmostEqual(
            self.context.budget.spent_usd, 0.4 + (4868 * 10 + 1649 * 40) / 1_000_000
        )
        # The old byte count of LangChain reprs blocked this request.
        old_prompt = (
            str(after_save.system_message) + str(after_save.messages) + str(after_save.tools)
        )
        remaining = self.context.budget.max_usd - self.context.budget.spent_usd
        self.assertGreater((len(old_prompt.encode()) * 10 + 2048 * 40) / 1_000_000, remaining)
        # The normalized tool arguments still count, even with empty Gemini content blocks.
        estimated = estimate_input(after_save).tokens
        self.assertGreaterEqual(estimated, 4868 + len(body.encode()))
        without_compat_copy = call.model_copy(update={"additional_kwargs": {}})
        self.assertEqual(
            estimate_input(
                after_save.override(
                    messages=[*request.messages, without_compat_copy, after_save.messages[-1]]
                )
            ).tokens,
            estimated,
        )
        final = Mock(return_value=ModelResponse(result=[AIMessage(content="Done")]))
        middleware.wrap_model_call(after_save, final)
        final.assert_called_once()
        self.assertLessEqual(self.context.budget.spent_usd, self.context.budget.max_usd)

    def test_input_usage_does_not_carry_over_to_changed_prompts_tools_or_history(self):
        model = ScriptedModel()
        request = ModelRequest(
            model=model,
            system_message=SystemMessage("One"),
            messages=[HumanMessage(content="Data")],
            runtime=Runtime(context=self.context),
        )
        usage = {"input_tokens": 22, "output_tokens": 1, "total_tokens": 23}
        ModelBudgetMiddleware(self.config, self.trace).wrap_model_call(
            request,
            Mock(
                return_value=ModelResponse(result=[AIMessage(content="OK", usage_metadata=usage)])
            ),
        )
        self.assertEqual(estimate_input(request).tokens, 22)
        for changed in (
            request.override(messages=[HumanMessage(content="Edit")]),
            request.override(system_message=SystemMessage("Two")),
            request.override(tools=toolkit_tools(self.local.catalog)),
            request.override(model=ScriptedModel()),
        ):
            fresh = changed.override(
                runtime=Runtime(context=RunContext("new", Budget(), self.context.request))
            )
            self.assertEqual(estimate_input(changed).tokens, estimate_input(fresh).tokens)

    def test_schema_tools_show_all_columns_without_samples_or_sql_attempts(self):
        responses = [
            AIMessage(
                content="", tool_calls=[{"id": "tables", "name": "sql_db_list_tables", "args": {}}]
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "schema",
                        "name": "sql_db_schema",
                        "args": {"table_names": "users, products"},
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
        database = self.local.catalog.database
        with (
            patch.object(database, "get_table_info", wraps=database.get_table_info) as schema,
            patch.object(
                database, "get_usable_table_names", wraps=database.get_usable_table_names
            ) as tables,
        ):
            self.invoke(self.agent(responses, tools=toolkit_tools(self.local.catalog)))
        schema.assert_called_once_with(["users", "products"])
        tables.assert_called_with()
        self.assertEqual(self.context.budget.sql_attempts, 0)
        self.assertEqual(self.context.budget.model_calls, 3)
        self.analytics.execute.assert_not_called()
        results = "\n".join(str(event["content"]) for event in self.events("tool_result"))
        for field in ("email", "first_name", "id", "age"):
            self.assertIn(field, results)
        self.assertIn("CREATE TABLE", results)
        self.assertNotIn("private@example.com", results)

    def test_unknown_schema_is_returned_to_model_for_correction(self):
        calls = AIMessage(
            content="",
            tool_calls=[
                {"id": "schema", "name": "sql_db_schema", "args": {"table_names": "employees"}},
            ],
        )
        self.invoke(
            self.agent([calls, AIMessage(content="done")], tools=toolkit_tools(self.local.catalog))
        )
        self.analytics.execute.assert_not_called()
        self.assertIn("Error:", self.events("tool_result")[0]["content"])
        self.assertIsNone(self.context.stopped)

    def test_model_cannot_inject_a_prepared_statement_or_access_parameters(self):
        forged = tool_call(QUERY)
        forged.tool_calls[0]["args"]["prepared"] = {"sql": "DELETE FROM products", "parameters": {}}
        self.invoke(
            self.agent(
                [forged, AIMessage(content="done")],
                tools=toolkit_tools(self.local.catalog),
            )
        )
        self.analytics.execute.assert_not_called()

    def test_prepared_statement_is_scoped_and_never_saved_in_messages_or_trace(self):
        self.invoke(
            self.agent(
                [tool_call(QUERY), AIMessage(content="done")],
                tools=toolkit_tools(self.local.catalog),
            )
        )
        prepared = self.analytics.execute.call_args.args[0]
        self.assertIn("__ra_products", prepared.sql)
        self.assertEqual(prepared.parameters["brand_0"], "A")
        for surface in (str(self.model.seen), str(self.trace.record.call_args_list)):
            self.assertNotIn("__ra_products", surface)
        self.assertNotIn("pii_salt", prepared.parameters)

    def test_sql_policies_reject_unsafe_queries_before_the_executor(self):
        for sql in (
            "DELETE FROM products",
            "SELECT * FROM `other.data.products`",
            "SELECT * FROM EXTERNAL_QUERY('x', 'y')",
            "SELECT * FROM products WHERE brand = @brand_0",
        ):
            with self.subTest(sql=sql):
                self.context = RunContext("test", Budget(), ChatRequest("Question", "chat"))
                agent = self.agent([tool_call(sql)], tools=toolkit_tools(self.local.catalog))
                with self.assertRaises(UnsafeQuery):
                    self.invoke(agent)
                self.analytics.execute.assert_not_called()

    def test_query_tool_without_middleware_cannot_execute_raw_sql(self):
        tools = toolkit_tools(self.local.catalog)
        self.assertEqual(
            set(tools[0].tool_call_schema.model_json_schema()["properties"]), {"query"}
        )
        agent = create_agent(
            ScriptedModel(responses=[tool_call(QUERY)]),
            tools=tools,
            context_schema=RunContext,
        )
        with self.assertRaises(AccessDenied):
            self.invoke(agent)
        self.analytics.execute.assert_not_called()

    def test_toolkit_checker_cannot_bypass_model_budget_middleware(self):
        checker_model = ScriptedModel()
        tools = SQLDatabaseToolkit(db=self.local.catalog.database, llm=checker_model).get_tools()
        call = AIMessage(
            content="",
            tool_calls=[
                {"id": "checker", "name": "sql_db_query_checker", "args": {"query": QUERY}}
            ],
        )
        with self.assertRaises(AccessDenied):
            self.invoke(self.agent([call], tools=tools))
        self.assertEqual(checker_model.seen, [])
        self.assertEqual(self.context.budget.model_calls, 1)
        self.analytics.execute.assert_not_called()

    def test_schema_discovery_loop_is_bounded_by_the_request_budget(self):
        self.context.budget.max_model_calls = 2
        agent = self.agent(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"id": str(i), "name": "sql_db_list_tables", "args": {}}],
                )
                for i in range(3)
            ],
            tools=toolkit_tools(self.local.catalog),
        )
        with self.assertRaises(BudgetExceeded):
            self.invoke(agent)
        self.assertEqual(self.context.budget.model_calls, 2)
        self.assertEqual(self.context.budget.sql_attempts, 0)
        self.assertEqual(len(self.model.seen), 2)
        self.analytics.execute.assert_not_called()
