import json
import tempfile
import unittest
from pathlib import Path
from time import monotonic
from unittest.mock import Mock, patch

from fakes import LocalAnalytics, ScriptedModel, call, echo_result, gemini_config, tool_call
from langchain.agents.middleware import RedactionRule
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import ValidationError

from retail_ai_agent.application.chat import ChatService
from retail_ai_agent.application.errors import AccessDenied, InvalidRequest
from retail_ai_agent.application.models import ChatRequest, QueryResult
from retail_ai_agent.application.progress import observe_progress
from retail_ai_agent.application.reports import ReportService
from retail_ai_agent.domain.models import Actor
from retail_ai_agent.infrastructure.agent.context import BRAND_NOTICE, RunContextFactory
from retail_ai_agent.infrastructure.agent.graph import create_chat_graph
from retail_ai_agent.infrastructure.agent.middleware.privacy import clean_message
from retail_ai_agent.infrastructure.agent.prompts import REPORT_INSTRUCTION
from retail_ai_agent.infrastructure.agent.runner import LangGraphRunner
from retail_ai_agent.infrastructure.privacy import DEFAULT_PII_RULES, LocalSanitizer
from retail_ai_agent.infrastructure.reports import FileReportStore
from retail_ai_agent.infrastructure.tracing import JsonlTrace

QUERY = "SELECT SUM(sale_price) AS total FROM order_items"


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(
            patch(
                "retail_ai_agent.infrastructure.agent.middleware.budget.count_gemini_input",
                return_value=20,
            )
        )
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.actor = Actor("a", "manager", ("A",))
        self.sanitizer = LocalSanitizer()
        self.config = gemini_config(
            google_cloud_project="test",
            max_model_calls=8,
            max_sql_attempts=3,
        )
        self.open_storage()

    def open_storage(self):
        self.store = FileReportStore(self.path / "reports")
        self.checkpointer = InMemorySaver()

    def app(self, responses=(), actor=None, *, intent="ANALYSIS"):
        actor = actor or self.actor
        model = ScriptedModel(responses=list(responses), intent=intent)
        self.analytics = LocalAnalytics(actor)
        self.addCleanup(self.analytics.connection.close)
        self.reports = ReportService(self.store, actor)
        trace = JsonlTrace(self.path / "trace.jsonl")
        graph = create_chat_graph(
            model=model,
            analytics=self.analytics,
            reports=self.reports,
            config=self.config,
            sanitizer=self.sanitizer,
            trace=trace,
            checkpointer=self.checkpointer,
            sql=self.analytics.sql,
        )
        runner = LangGraphRunner(
            graph=graph,
            reports=self.reports,
            actor=actor,
            config=self.config,
            trace=trace,
            contexts=RunContextFactory(self.config, self.analytics.sql),
            sanitizer=self.sanitizer,
        )
        return ChatService(runner), model

    def create_report(self):
        app, _ = self.app(
            [
                tool_call(QUERY),
                call("save_report", body="Sales: 10"),
                AIMessage(content="Sales: 10"),
            ],
            intent="REPORT",
        )
        result = app.ask(ChatRequest("Create a sales report", "chat"))
        self.assertTrue(result.report_id, result.answer)
        return result.report_id

    def test_normal_answer_does_not_create_report(self):
        app, _ = self.app([tool_call(QUERY), AIMessage(content="Sales: 10")])
        result = app.ask(ChatRequest("What were total sales?", "chat"))
        self.assertIn("10", result.answer)
        self.assertEqual(self.store.list("a"), [])

    def test_unrequested_report_tool_after_valid_rows_cannot_publish(self):
        app, model = self.app(
            [
                tool_call(QUERY),
                call("save_report", body="January revenue: 10"),
                AIMessage(content="January revenue: 10"),
            ]
        )
        result = app.ask(ChatRequest("What were my sales in January?", "chat"))
        self.assertEqual(result.status, "complete")
        self.assertIn("10", result.answer)
        self.assertIn("not saved", model.seen[-1][-1].text)
        self.assertFalse(result.report_id)
        self.assertEqual(self.store.list("a"), [])

    def test_natural_report_uses_report_deadline_after_classification(self):
        self.config.question_timeout_seconds = 5
        self.config.report_timeout_seconds = 30
        app, _ = self.app([tool_call(QUERY), echo_result], intent="REPORT")
        execute = self.analytics.execute

        def check_deadline(prepared, budget):
            self.assertEqual(budget.timeout, 30)
            self.assertGreaterEqual(budget.model_calls, 2)
            self.assertLessEqual(budget.started, monotonic())
            return execute(prepared, budget)

        self.analytics.execute = check_deadline
        response = app.ask(ChatRequest("Prepare a January sales report", "chat"))
        self.assertEqual(response.status, "complete", response.answer)
        self.assertEqual(len(self.analytics.calls), 1)

    def test_unclassified_model_prose_is_never_displayed(self):
        app, model = self.app(intent="A lighthouse shines upon the sea")
        result = app.ask(ChatRequest("Write a poem about a lighthouse", "chat"))
        self.assertEqual(result.status, "failed")
        self.assertNotIn("lighthouse", result.answer)
        self.assertEqual(model.seen, [])
        self.assertEqual(self.analytics.calls, [])
        self.assertEqual(self.store.list("a"), [])

    def test_empty_scalar_report_and_real_zero(self):
        query = "SELECT SUM(sale_price) AS total FROM order_items WHERE status='Complete'"
        for matched in (False, True):
            with self.subTest(matched=matched):
                app, model = self.app(
                    [
                        tool_call(query),
                        call("save_report", body="Completed revenue: 0"),
                        AIMessage(content="No matching sales." if not matched else "Revenue: 0"),
                    ]
                )
                self.analytics.connection.execute(
                    "UPDATE order_items SET sale_price=0, status=? WHERE product_id=1",
                    ("Complete" if matched else "Cancelled",),
                )
                result = app.ask(ChatRequest("January completed revenue", "chat", action="report"))
                observation = json.loads(model.seen[1][-1].text)
                self.assertEqual(observation["rows"], [{"total": 0}] if matched else [])
                self.assertEqual("notice" in observation, not matched)
                self.assertEqual(bool(result.report_id), matched)
                self.assertEqual(len(self.store.list("a")), int(matched))

    def test_null_aggregate_observation_cannot_publish(self):
        app, model = self.app(
            [
                tool_call(QUERY),
                call("save_report", body="Revenue: 0"),
                echo_result,
            ]
        )
        self.analytics.execute = Mock(return_value=QueryResult(QUERY, ({"total": None},)))
        result = app.ask(ChatRequest("January revenue", "chat", action="report"))
        observation = json.loads(model.seen[1][-1].text)
        self.assertEqual(observation["rows"], [])
        self.assertIn("NULL", observation["notice"])
        self.assertEqual(result.report_id, "")
        self.assertEqual(self.store.list("a"), [])

    def test_stale_user_query_is_corrected_against_schema_and_scoped_rows(self):
        valid = (
            "SELECT SUM(sale_price) AS total FROM order_items WHERE status='Complete' "
            "AND created_at >= '2024-01-01' AND created_at < '2024-02-01'"
        )
        stale = valid.replace("sale_price", "item_revenue")
        app, model = self.app(
            [
                tool_call(stale, "old"),
                tool_call(valid, "fixed"),
                echo_result,
            ]
        )
        self.analytics.connection.executescript("""
            INSERT INTO order_items VALUES(3,100,1,1,'Cancelled',90,'2024-01-20',NULL,NULL,NULL);
            INSERT INTO order_items VALUES(4,100,1,1,'Complete',80,'2024-02-01',NULL,NULL,NULL);
        """)
        result = app.ask(
            ChatRequest("Validate and repair my January revenue SQL: " + stale, "chat")
        )
        observations = [m for m in model.seen[-1] if isinstance(m, ToolMessage)]
        self.assertTrue(json.loads(observations[0].text)["retryable"])
        self.assertIn("sale_price", model.seen[1][0].text)
        self.assertEqual(json.loads(observations[1].text)["rows"], [{"total": 10}])
        self.assertEqual(len(self.analytics.calls), 1)
        self.assertEqual(result.status, "complete")
        self.assertIn('"total": 10', result.answer)
        self.assertEqual(self.store.list("a"), [])

    def test_repeated_invalid_candidates_exhaust_shared_sql_limit(self):
        bad = "SELECT item_revenue FROM order_items"
        app, _ = self.app([tool_call(bad, f"bad{i}") for i in range(4)])
        result = app.ask(ChatRequest("Use item_revenue for January sales", "chat"))
        self.assertEqual(result.status, "failed")
        self.assertIn("allowed number", result.answer)
        self.assertEqual(self.analytics.calls, [])
        events = [json.loads(line) for line in (self.path / "trace.jsonl").read_text().splitlines()]
        self.assertEqual(events[-1]["sql_attempts"], self.config.max_sql_attempts)

    def test_sql_plan_updates_share_the_query_loop_and_reset_on_the_next_request(self):
        step = "Compare sales for private@example.com"
        planned = [{"content": step, "status": "in_progress"}]
        done = [{"content": step, "status": "completed"}]
        app, model = self.app(
            [
                call("write_todos", todos=planned),
                tool_call(QUERY),
                call("write_todos", todos=done),
                AIMessage(content="Sales: 10"),
                AIMessage(content="You're welcome!"),
            ]
        )
        result = app.ask(ChatRequest("Plan a sales comparison and calculate revenue", "chat"))
        self.assertEqual(result.status, "complete", result.answer)
        self.assertEqual(len(self.analytics.calls), 1)
        messages = [m for m in model.seen[-1] if isinstance(m, ToolMessage)]
        self.assertEqual([m.name for m in messages], ["write_todos", "sql_db_query", "write_todos"])
        self.assertIn("in_progress", messages[0].text)
        self.assertIn("completed", messages[-1].text)
        config = app.runner._graph_config("chat")
        self.assertEqual(
            app.runner.graph.get_state(config).values["todos"],
            [{"content": "Compare sales for [REDACTED_EMAIL]", "status": "completed"}],
        )
        trace = (self.path / "trace.jsonl").read_text()
        self.assertNotIn("private@example.com", trace)
        events = [json.loads(line) for line in trace.splitlines()]
        self.assertEqual(len([e for e in events if e["event"] == "tool_result"]), 3)
        self.assertEqual(events[-1]["sql_attempts"], 1)
        self.assertEqual(events[-1]["model_calls"], 5)

        self.assertEqual(app.ask(ChatRequest("Thanks", "chat")).answer, "You're welcome!")
        self.assertEqual(app.runner.graph.get_state(config).values["todos"], [])
        self.assertNotIn("Compare sales", str(model.seen[-1]))

    def test_conversation_replies_use_model_and_history_without_queries_or_reports(self):
        turns = [
            ("hello!!", "Hey! What would you like to explore?"),
            ("Can you answer?", "Yes, I'm here. What would you like to know?"),
            ("Can you keep it brief?", "Sure, I'll keep it brief."),
            ("thanks", "You're welcome!"),
        ]
        app, model = self.app([AIMessage(content=answer) for _, answer in turns])
        self.analytics.execute = Mock(side_effect=AssertionError("Chat must not query data"))
        for index, (question, answer) in enumerate(turns):
            with self.subTest(question=question):
                result = app.ask(ChatRequest(question, "chat"))
                self.assertEqual(result.status, "complete", result.answer)
                self.assertEqual(result.answer, answer)
                self.assertFalse(result.report_id)
                self.assertEqual(len(model.seen), index + 1)
                self.assertEqual(
                    [m.text for m in model.seen[-1] if not isinstance(m, SystemMessage)],
                    [text for turn in turns[:index] for text in turn] + [question],
                )
        self.analytics.execute.assert_not_called()
        self.assertEqual(self.store.list("a"), [])
        state = app.runner.graph.get_state(app.runner._graph_config("chat")).values
        self.assertEqual([m.text for m in state["messages"]], [t for turn in turns for t in turn])

    def test_chat_history_is_available_to_following_analysis(self):
        app, model = self.app(
            [
                AIMessage(content="I can compare sales across periods."),
                tool_call(QUERY),
                AIMessage(content="Sales: 10"),
                AIMessage(content="You're welcome!"),
            ]
        )
        app.ask(ChatRequest("What can you do?", "chat"))
        result = app.ask(ChatRequest("Hi, show me total sales", "chat"))
        self.assertEqual(result.status, "complete", result.answer)
        self.assertIn("I can compare sales across periods.", str(model.seen[2]))
        result = app.ask(ChatRequest("Thanks", "chat"))
        self.assertEqual(result.answer, "You're welcome!")
        self.assertEqual(len(self.analytics.calls), 1)
        self.assertEqual(self.store.list("a"), [])

    def test_empty_conversation_reply_is_retried(self):
        for empty in ("", " \n\t"):
            with self.subTest(empty=empty):
                app, model = self.app([AIMessage(content=empty), AIMessage(content="Answer.")])
                result = app.ask(ChatRequest("A message", "chat"))
                self.assertEqual(result.answer, "Answer.")
                self.assertEqual(len(model.seen), 2)
                self.assertEqual(self.analytics.calls, [])

    def test_conversation_reply_is_redacted_in_output_history_and_trace(self):
        app, _ = self.app([AIMessage(content="Contact: private@example.com")])
        result = app.ask(ChatRequest("Are you there?", "chat"))
        self.assertEqual(result.status, "complete", result.answer)
        self.assertEqual(result.answer, "Contact: [REDACTED_EMAIL]")
        state = app.runner.graph.get_state(app.runner._graph_config("chat")).values
        for surface in (str(state), (self.path / "trace.jsonl").read_text()):
            self.assertNotIn("private@example.com", surface)

    def test_legacy_function_arguments_are_cleaned_after_decoding_unicode(self):
        payload = r'{"body": "Email: user\u0040example.com"}'
        msg = AIMessage(
            content="",
            additional_kwargs={"function_call": {"name": "save_report", "arguments": payload}},
        )
        cleaned = clean_message(msg, self.sanitizer)
        args = json.loads(cleaned.additional_kwargs["function_call"]["arguments"])
        self.assertEqual(args["body"], "Email: [REDACTED_EMAIL]")

    def test_local_detector_failure_stops_before_count_or_generation(self):
        self.sanitizer = LocalSanitizer(
            rules=(RedactionRule("email", detector=Mock(side_effect=OSError("local error"))),)
        )
        app, model = self.app()
        with patch(
            "retail_ai_agent.infrastructure.agent.middleware.budget.count_gemini_input"
        ) as count:
            result = app.ask(ChatRequest("user@example.com", "chat"))
        self.assertEqual(result.status, "failed")
        self.assertNotIn("user@example.com", result.answer)
        self.assertNotIn("local error", result.answer)
        count.assert_not_called()
        self.assertEqual(model.seen, [])

    def test_report_command_with_only_greeting_does_not_create_report(self):
        app, model = self.app([AIMessage(content="Hi! What report should I prepare?")])
        result = app.ask(ChatRequest("Hi", "chat", action="report"))
        self.assertEqual(result.answer, "Hi! What report should I prepare?")
        self.assertEqual(len(model.seen), 1)
        self.assertEqual(self.analytics.calls, [])
        self.assertEqual(self.store.list("a"), [])

    def test_explicit_report_saves_only_executed_sql_and_answer(self):
        report = self.store.get(self.create_report())
        self.assertIn("__ra_order_items", report.sql[0])
        self.assertEqual(report.scope, self.actor.scope)
        self.assertIn("10", report.body)

    def test_report_command_publishes_a_completed_sanitized_draft_if_model_omits_save(self):
        app, _ = self.app([tool_call(QUERY), AIMessage(content="Sales: 10. Contact Alice Morgan.")])
        result = app.ask(ChatRequest("Report. Customer Alice Morgan.", "chat", action="report"))
        self.assertEqual(result.status, "complete", result.answer)
        self.assertIn(result.report_id, result.answer)
        report = self.store.get(result.report_id)
        self.assertIn("Sales: 10", report.body)
        self.assertNotIn("Morgan", report.body + report.question + result.answer)
        self.assertEqual(len(report.sql), 1)

    def test_context_schema_query_and_save_report(self):
        app, model = self.app(
            [
                tool_call(QUERY),
                call("save_report", body="Sales: 10"),
                AIMessage(content="Sales: 10"),
            ],
            intent="REPORT",
        )
        result = app.ask(ChatRequest("Create a sales report", "chat"))
        self.assertEqual(result.status, "complete", result.answer)
        self.assertTrue(result.report_id)
        self.assertEqual(len(self.analytics.calls), 1)
        observations = {
            message.tool_call_id: message.content
            for message in model.seen[-1]
            if isinstance(message, ToolMessage)
        }
        self.assertIn("order_items(", model.seen[0][0].text)
        self.assertIn("sale_price FLOAT64", model.seen[0][0].text)
        self.assertIn("user_id INT64", model.seen[0][0].text)
        self.assertEqual(json.loads(observations["query"])["rows"], [{"total": 10.0}])
        self.assertEqual(len(self.store.get(result.report_id).sql), 1)
        self.assertIn("automatic filter", result.answer)
        self.assertIn("automatic filter", self.store.get(result.report_id).body)
        state = app.runner.graph.get_state(app.runner._graph_config("chat")).values
        self.assertNotIn("prepared_queries", state)
        self.assertNotIn("pii_salt", str(state))

    def test_pii_is_masked_before_model_trace_history_and_report(self):
        query = "SELECT first_name AS contact, email AS address, id AS customer FROM users"
        app, model = self.app(
            [
                tool_call(query),
                call("save_report", body="Anonymized report"),
                AIMessage(content="Done"),
            ],
            intent="REPORT",
        )
        result = app.ask(ChatRequest("Create a customer report", "chat"))
        self.assertEqual(result.status, "complete", result.answer)
        observation = next(m for m in model.seen[-1] if isinstance(m, ToolMessage))
        row = json.loads(observation.content)["rows"][0]
        self.assertEqual(
            row, {"contact": "[REDACTED]", "address": "[REDACTED]", "customer": "[PII]"}
        )
        state = app.runner.graph.get_state(app.runner._graph_config("chat")).values
        surfaces = [
            str(model.seen),
            (self.path / "trace.jsonl").read_text(),
            str(state),
            self.store.get(result.report_id).body,
        ]
        for surface in surfaces:
            self.assertNotIn("Private", surface)
            self.assertNotIn("private@example.com", surface)

    def test_schema_turn_uses_context_and_rejects_unadvertised_tool_call(self):
        app, model = self.app(
            [
                AIMessage(content="orders, order_items, products, users"),
                tool_call("SELECT table_name FROM information_schema.tables"),
            ],
            intent="SCHEMA",
        )
        with patch.object(
            ScriptedModel, "bind_tools", side_effect=AssertionError("unexpected tools")
        ):
            result = app.ask(ChatRequest("Which tables are available?", "chat"))
            rejected = app.ask(ChatRequest("Which columns are in orders?", "chat"))
        self.assertEqual(result.status, "complete", result.answer)
        self.assertIn("order_items(", model.seen[0][0].text)
        self.assertEqual(rejected.status, "failed")
        self.assertEqual(self.analytics.calls, [])
        events = [json.loads(line) for line in (self.path / "trace.jsonl").read_text().splitlines()]
        self.assertFalse(any(e["event"] == "tool_call" for e in events))

    def test_grant_count_is_in_the_prompt_before_sql(self):
        for role, brands, expected in (
            ("manager", (), "exactly 0 distinct"),
            ("manager", ("A", "A"), "exactly 1 distinct"),
            ("manager", ("A", "B"), "exactly 2 distinct"),
            ("ceo", (), "access to all brands"),
        ):
            with self.subTest(role=role, brands=brands):
                app, model = self.app([AIMessage(content="Done")], Actor("test", role, brands))
                app.ask(ChatRequest("What comparisons are possible?", "chat"))
                self.assertIn(expected, model.seen[0][0].text)

    def test_scoped_report_and_answer_each_render_one_filter_notice(self):
        draft = f"Revenue: 10. Compare categories.\n\n{BRAND_NOTICE}\n\n{BRAND_NOTICE}"
        app, model = self.app(
            [tool_call(QUERY), call("save_report", body=draft), AIMessage(content=draft)],
            intent="REPORT",
        )
        response = app.ask(ChatRequest("Prepare a sales report", "chat"))
        self.assertEqual(response.status, "complete", response.answer)
        self.assertNotIn(BRAND_NOTICE, model.seen[1][-1].text)
        for text in (response.answer, self.store.get(response.report_id).body):
            self.assertEqual(text.count(BRAND_NOTICE), 1)
            self.assertIn("Compare categories.", text)

    def test_ceo_answer_has_no_brand_filter_warning_and_includes_all_brands(self):
        app, model = self.app(
            [tool_call(QUERY), AIMessage(content="Sales: 40")], actor=Actor("ceo", "ceo")
        )
        result = app.ask(ChatRequest("What were sales?", "chat"))
        self.assertEqual(result.answer, "Sales: 40")
        observation = next(m for m in model.seen[-1] if isinstance(m, ToolMessage))
        payload = json.loads(observation.content)
        self.assertEqual(payload["rows"], [{"total": 40}])
        self.assertNotIn("brand_filtered", payload)

    def test_empty_scoped_result_keeps_filter_warning(self):
        app, model = self.app(
            [
                tool_call("SELECT brand FROM products WHERE brand='B'"),
                AIMessage(content="No data."),
            ]
        )
        result = app.ask(ChatRequest("What were sales for brand B?", "chat"))
        self.assertEqual(result.status, "complete", result.answer)
        self.assertIn("automatic filter", result.answer)
        observation = next(m for m in model.seen[-1] if isinstance(m, ToolMessage))
        payload = json.loads(observation.content)
        self.assertEqual(payload["rows"], [])
        self.assertTrue(payload["brand_filtered"])

    def test_sql_error_is_returned_to_model_for_correction(self):
        app, model = self.app(
            [
                tool_call("SELECT typo FROM order_items", "bad"),
                tool_call(QUERY, "good"),
                AIMessage(content="10"),
            ]
        )
        result = app.ask(ChatRequest("What were sales?", "chat"))
        self.assertTrue(result.answer.startswith("10"))
        self.assertIn("automatic filter", result.answer)
        errors = [m for m in model.seen[2] if isinstance(m, ToolMessage)]
        self.assertEqual(errors[0].tool_call_id, "bad")
        self.assertIn("error", errors[0].content)

    def test_syntax_error_and_empty_result_can_be_corrected_within_one_budget(self):
        app, model = self.app(
            [
                tool_call("SELECT 'unterminated", "syntax"),
                tool_call("SELECT sale_price FROM order_items WHERE status='complete'", "empty"),
                tool_call("SELECT sale_price FROM order_items WHERE status='Complete'", "fixed"),
                AIMessage(content="Sales: 10"),
            ]
        )
        stages = []
        with observe_progress(stages.append):
            result = app.ask(ChatRequest("Show completed sales", "chat"))
        self.assertEqual(
            stages,
            [
                "classify",
                "model",
                "sql_db_query",
                "query_error",
                "model",
                "sql_db_query",
                "no_data",
                "model",
                "sql_db_query",
                "model",
            ],
        )
        self.assertEqual(result.status, "complete", result.answer)
        observations = [m.text for m in model.seen[-1] if isinstance(m, ToolMessage)]
        self.assertIn("Correct the GoogleSQL", observations[0])
        self.assertIn("No rows", observations[1])
        self.assertIn("10", observations[2])
        self.assertEqual(len(self.analytics.calls), 2)
        self.assertNotIn("error", result.answer)

    def test_empty_result_is_an_observation_and_is_not_saved_as_evidence(self):
        empty = "SELECT sale_price FROM order_items WHERE sale_price < 0"
        app, model = self.app(
            [
                tool_call(empty, "first"),
                tool_call(empty, "check"),
                call("save_report", body="No data."),
                echo_result,
            ],
            intent="REPORT",
        )
        result = app.ask(ChatRequest("Create a report on negative sales", "chat"))
        self.assertIn("not saved", result.answer)
        self.assertEqual(self.store.list("a"), [])
        self.assertIn("No rows", str(model.seen[-1]))
        self.assertEqual(len(self.analytics.calls), 1)

    def test_mixed_report_and_sql_tools_resume_with_the_same_evidence_and_budget(self):
        plan = [
            {
                "content": "Check the old report and refresh its sales figures",
                "status": "in_progress",
            }
        ]
        app, model = self.app(
            [
                call("write_todos", todos=plan),
                call("show_report", report_id="old"),
                tool_call(QUERY),
                call("delete_reports", report_id="old"),
                call("list_reports"),
                call("save_report", body="Updated sales: 10"),
                AIMessage(content="I updated the report."),
            ],
            intent="REPORT",
        )
        self.reports.save("old", "chat", "Old sales", [QUERY], "Sales: 9")
        preview = app.ask(ChatRequest("Check the old report, delete it and save a new one", "chat"))
        self.assertEqual(preview.status, "awaiting_confirmation", preview.answer)
        config = app.runner._graph_config("chat")
        self.assertEqual(app.runner.graph.get_state(config).values["todos"], plan)
        self.assertIsNotNone(self.store.get("old"))
        result = app.confirm("chat", preview.pending_action_id, True)
        self.assertEqual(result.status, "complete", result.answer)
        self.assertEqual(app.runner.graph.get_state(config).values["todos"], plan)
        self.assertIsNone(self.store.get("old"))
        saved = self.store.get(result.report_id)
        self.assertEqual(saved.sql, tuple(self.analytics.calls))
        self.assertIn("automatic filter", saved.body)
        self.assertEqual(len(self.analytics.calls), 1)
        observations = [m for m in model.seen[-1] if isinstance(m, ToolMessage)]
        self.assertEqual(
            [m.name for m in observations],
            [
                "write_todos",
                "show_report",
                "sql_db_query",
                "delete_reports",
                "list_reports",
                "save_report",
            ],
        )
        self.assertIn("No reports found", observations[4].text)
        events = [json.loads(line) for line in (self.path / "trace.jsonl").read_text().splitlines()]
        self.assertEqual(len({event["run_id"] for event in events}), 1)
        self.assertEqual(events[-1]["model_calls"], 8)
        self.assertEqual(events[-1]["sql_attempts"], 1)

    def test_failed_final_reply_reports_committed_changes(self):
        app, _ = self.app(
            [
                tool_call(QUERY),
                call("save_report", body="Sales: 10"),
                RuntimeError("offline"),
                call("delete_reports", conversation_only=True),
                RuntimeError("offline"),
            ],
            intent="REPORT",
        )
        saved = app.ask(ChatRequest("Create a report", "chat"))
        self.assertEqual(saved.status, "failed")
        self.assertIsNotNone(self.store.get(saved.report_id))
        self.assertIn(saved.report_id, saved.answer)

        preview = app.ask(ChatRequest("Delete the reports from this conversation", "chat"))
        deleted = app.confirm("chat", preview.pending_action_id, True)
        self.assertEqual(deleted.status, "failed")
        self.assertIsNone(self.store.get(saved.report_id))
        self.assertIn("Reports deleted: 1", deleted.answer)

    def test_save_requires_results_already_seen_in_this_turn(self):
        query_and_save = tool_call(QUERY)
        query_and_save.tool_calls.extend(call("save_report", body="Premature report").tool_calls)
        app, model = self.app(
            [
                query_and_save,
                echo_result,
                call("save_report", body="Previous turn"),
                echo_result,
            ],
            intent="REPORT",
        )
        for question in ("Create a report", "Save another report"):
            result = app.ask(ChatRequest(question, "chat"))
            self.assertFalse(result.report_id)
            self.assertIn("not saved", result.answer)
        self.assertEqual(len(model.seen), 4)
        self.assertEqual(self.store.list("a"), [])

    def test_delete_interrupt_can_be_confirmed_in_session_and_is_idempotent(self):
        report_id = self.create_report()
        app, model = self.app(
            [call("delete_reports"), echo_result, call("delete_reports"), echo_result]
        )
        preview = app.ask(
            ChatRequest("Delete reports", "chat", action="delete", conversation_only=True)
        )
        self.assertEqual(preview.status, "awaiting_confirmation")
        self.assertIn(report_id, preview.answer)
        self.assertIsNotNone(self.store.get(report_id))
        result = app.confirm("chat", preview.pending_action_id, True)
        self.assertEqual(result.status, "complete", result.answer)
        self.assertIn("Reports deleted: 1", result.answer)
        self.assertIsNone(self.store.get(report_id))
        self.assertIn(
            "already completed", app.confirm("chat", preview.pending_action_id, True).answer
        )
        empty = app.ask(ChatRequest("/delete", "chat", action="delete"))
        self.assertEqual(empty.status, "complete")
        self.assertFalse(empty.pending_action_id)
        self.assertEqual(len(model.seen), 4)

    def test_restart_discards_history_and_pending_pause_but_keeps_reports(self):
        report_id = self.create_report()
        app, _ = self.app(
            [call("delete_reports"), echo_result, call("delete_reports"), echo_result]
        )
        preview = app.ask(ChatRequest("Delete", "chat", action="delete"))
        self.open_storage()
        app, model = self.app(
            [AIMessage(content="New answer"), call("delete_reports"), echo_result]
        )
        self.assertEqual(app.runner.graph.get_state(app.runner._graph_config("chat")).values, {})
        for approved in (True, False):
            with self.assertRaisesRegex(InvalidRequest, "Repeat the deletion request"):
                app.confirm("chat", preview.pending_action_id, approved)
        self.assertIsNotNone(self.store.get(report_id))
        result = app.ask(ChatRequest("What data is available?", "chat"))
        self.assertEqual(result.answer, "New answer")
        self.assertNotIn("Sales: 10", str(model.seen))
        fresh = app.ask(ChatRequest("Delete", "chat", action="delete"))
        self.assertNotEqual(fresh.pending_action_id, preview.pending_action_id)
        self.assertIn(report_id, fresh.answer)
        self.assertEqual(app.confirm("chat", fresh.pending_action_id, True).status, "complete")
        self.assertIsNone(self.store.get(report_id))

    def test_cancellation_and_snapshot_exclude_new_reports(self):
        first = self.create_report()
        app, _ = self.app(
            [call("delete_reports"), echo_result, call("delete_reports"), echo_result]
        )
        preview = app.ask(ChatRequest("Delete", "chat", action="delete"))
        later = self.reports.save("later", "chat", "Later sales", [QUERY], "10")
        app.confirm("chat", preview.pending_action_id, True)
        self.assertIsNone(self.store.get(first))
        self.assertIsNotNone(self.store.get(later.id))
        preview = app.ask(ChatRequest("Delete", "chat", action="delete"))
        self.assertIn("cancelled", app.confirm("chat", preview.pending_action_id, False).answer)
        self.assertIsNotNone(self.store.get(later.id))

    def test_expired_graph_approval_can_be_cancelled_after_failure(self):
        report_id = self.create_report()
        app, _ = self.app([call("delete_reports"), echo_result])
        preview = app.ask(ChatRequest("Delete", "chat", action="delete"))
        action = self.store.pending(preview.pending_action_id)
        with patch("retail_ai_agent.application.reports.time", return_value=action.expires_at + 1):
            for _ in range(2):
                with self.assertRaises(InvalidRequest):
                    app.confirm("chat", action.id, True)
        cancelled = app.confirm("chat", action.id, False)
        self.assertEqual(cancelled.status, "complete", cancelled.answer)
        self.assertIn("cancelled", cancelled.answer)
        self.assertIsNotNone(self.store.get(report_id))

    def test_failure_before_hitl_resume_leaves_the_deletion_pending(self):
        report_id = self.create_report()
        app, _ = self.app([call("delete_reports"), echo_result])
        preview = app.ask(ChatRequest("Delete", "chat", action="delete"))
        with patch.object(app.runner.graph, "invoke", side_effect=RuntimeError("crash")):
            result = app.confirm("chat", preview.pending_action_id, True)
        self.assertEqual(result.status, "failed")
        self.assertIsNotNone(self.store.get(report_id))
        self.assertEqual(self.store.pending(preview.pending_action_id).status, "pending")
        self.assertIn("cancelled", app.confirm("chat", preview.pending_action_id, False).answer)
        self.assertIsNotNone(self.store.get(report_id))

    def test_failure_after_tool_commit_recovers_the_persisted_outcome(self):
        report_id = self.create_report()
        app, _ = self.app([call("delete_reports"), echo_result])
        preview = app.ask(ChatRequest("Delete", "chat", action="delete"))
        commit = self.store.finish_deletion

        def crash_after_commit(action, approved):
            commit(action, approved)
            raise OSError("crash")

        with patch.object(self.store, "finish_deletion", side_effect=crash_after_commit):
            result = app.confirm("chat", preview.pending_action_id, True)
        self.assertEqual(result.status, "failed")
        self.assertIsNone(self.store.get(report_id))
        app, model = self.app()
        result = app.confirm("chat", preview.pending_action_id, False)
        self.assertEqual(result.status, "complete", result.answer)
        self.assertIn("already completed: deleted", result.answer)
        self.assertEqual(model.seen, [])

    def test_failed_rejection_can_be_retried_with_a_fresh_approval(self):
        report_id = self.create_report()
        app, _ = self.app([call("delete_reports"), echo_result])
        preview = app.ask(ChatRequest("Delete", "chat", action="delete"))
        with patch.object(self.store, "finish_deletion", side_effect=OSError("disk full")):
            for _ in range(2):
                result = app.confirm("chat", preview.pending_action_id, False)
                self.assertEqual(result.status, "failed")
        self.assertIsNotNone(self.store.get(report_id))
        self.assertIn(
            "Reports deleted: 1", app.confirm("chat", preview.pending_action_id, True).answer
        )
        self.assertIsNone(self.store.get(report_id))

    def test_hitl_exposes_the_redacted_preview_and_rejects_tool_edits(self):
        app, model = self.app([call("delete_reports"), echo_result])
        self.reports.save("contact", "chat", "private@example.com", [QUERY], "Test")
        preview = app.ask(ChatRequest("Delete", "chat", action="delete"))
        config = app.runner._graph_config("chat")
        (pause,) = app.runner.graph.get_state(config).interrupts
        self.assertEqual(pause.value["type"], "tool_approval")
        self.assertEqual(pause.value["action_id"], preview.pending_action_id)
        self.assertEqual(pause.value["description"], preview.answer)
        self.assertIn("[REDACTED_EMAIL]", preview.answer)
        edit = {"type": "edit", "edited_action": {"name": "delete_reports", "args": {}}}
        with self.assertRaises(ValidationError):
            app.runner.graph.invoke(
                Command(resume={pause.id: edit}),
                config=config,
                context=app.runner._paused["chat"][0],
            )
        self.assertIsNotNone(self.store.get("contact"))
        with patch.object(self.reports, "confirm", wraps=self.reports.confirm) as confirm:
            cancelled = app.confirm("chat", preview.pending_action_id, False)
        self.assertIn("cancelled", cancelled.answer)
        confirm.assert_called_once_with(preview.pending_action_id, "chat", approved=False)
        self.assertEqual(len(model.seen), 2)

    def test_pending_action_cannot_be_confirmed_by_text_or_other_actor(self):
        self.create_report()
        app, model = self.app([call("delete_reports")])
        preview = app.ask(ChatRequest("Delete", "chat", action="delete"))
        self.assertEqual(
            app.ask(ChatRequest("yes, I confirm", "chat")).status, "awaiting_confirmation"
        )
        self.assertEqual(len(model.seen), 1)
        for actor in (Actor("b", "ceo"), Actor("a", "manager", ("B",))):
            other, _ = self.app(actor=actor)
            with self.assertRaises(AccessDenied):
                other.confirm("chat", preview.pending_action_id, True)
        with self.assertRaises(InvalidRequest):
            app.confirm("chat", preview.pending_action_id, "yes")

    def test_history_is_available_only_to_same_user_scope_and_conversation(self):
        app, _ = self.app([AIMessage(content="Answer ABC")])
        app.ask(ChatRequest("What is in the schema?", "chat"))
        app, model = self.app([AIMessage(content="Follow-up")])
        app.ask(ChatRequest("Tell me more", "chat"))
        self.assertIn("Answer ABC", str(model.seen))
        for actor in (Actor("b", "ceo"), Actor("a", "manager", ("B",))):
            other, model = self.app([AIMessage(content="New answer")], actor)
            other.ask(ChatRequest("Tell me more", "chat"))
            self.assertNotIn("Answer ABC", str(model.seen))

    def test_native_message_history_keeps_only_configured_number_of_complete_turns(self):
        self.config.history_max_turns = 2
        responses = [AIMessage(content=f"a{i}") for i in range(4)]
        app, model = self.app(responses)
        for i in range(4):
            result = app.ask(ChatRequest(f"q{i}", "chat"))
            self.assertEqual(result.status, "complete", result.answer)
        state = app.runner.graph.get_state(app.runner._graph_config("chat")).values
        self.assertNotIn("history", state)
        self.assertEqual([message.text for message in state["messages"]], ["q2", "a2", "q3", "a3"])
        self.assertEqual(
            [type(message) for message in state["messages"]],
            [HumanMessage, AIMessage, HumanMessage, AIMessage],
        )
        self.assertEqual(
            [m.text for m in model.seen[-1] if not isinstance(m, SystemMessage)],
            ["q1", "a1", "q2", "a2", "q3"],
        )

    def test_history_character_limit_preserves_latest_pair_even_when_oversized(self):
        self.config.history_max_chars = 12
        app, _ = self.app([AIMessage(content="a"), AIMessage(content="x" * 20)])
        app.ask(ChatRequest("q", "chat"))
        app.ask(ChatRequest("next", "chat"))
        state = app.runner.graph.get_state(app.runner._graph_config("chat")).values
        self.assertEqual([message.text for message in state["messages"]], ["next", "x" * 20])

    def test_failed_turn_does_not_pollute_history_and_next_request_has_fresh_budget(self):
        self.config.max_model_calls = 2
        app, model = self.app(
            [
                AIMessage(content="first answer"),
                RuntimeError("failed answer"),
                AIMessage(content="next answer"),
            ]
        )
        self.assertEqual(app.ask(ChatRequest("first", "chat")).status, "complete")
        self.assertEqual(app.ask(ChatRequest("broken turn", "chat")).status, "failed")
        self.assertEqual(app.ask(ChatRequest("next", "chat")).answer, "next answer")
        self.assertNotIn("broken turn", str(model.seen[-1:]))
        state = app.runner.graph.get_state(app.runner._graph_config("chat")).values
        self.assertEqual(
            [message.text for message in state["messages"]],
            ["first", "first answer", "next", "next answer"],
        )

    def test_report_prompt_is_system_instruction_and_does_not_leak_into_next_turn(self):
        question = "January sales"
        app, model = self.app(
            [
                AIMessage(content="Report body"),
                AIMessage(content="Normal answer"),
            ]
        )
        app.ask(ChatRequest(question, "chat", action="report"))
        analysis_call = model.seen[0]
        self.assertIn(REPORT_INSTRUCTION, analysis_call[0].text)
        self.assertEqual(analysis_call[-1].text, question)
        app.ask(ChatRequest("Continue", "chat"))
        self.assertNotIn(REPORT_INSTRUCTION, "\n".join(m.text for m in model.seen[-1]))
        state = app.runner.graph.get_state(app.runner._graph_config("chat")).values
        self.assertNotIn(
            REPORT_INSTRUCTION, "\n".join(message.text for message in state["messages"])
        )

    def test_pii_is_removed_before_models_including_history_and_final_output(self):
        app, model = self.app([AIMessage(content="Contact private@example.com"), AIMessage("Done")])
        result = app.ask(ChatRequest("private@example.com 4111 1111 1111 1111 192.168.1.1", "chat"))
        followup = app.ask(ChatRequest("Tell me more", "chat"))
        self.assertEqual(result.status, "complete", result.answer)
        self.assertEqual(followup.status, "complete", followup.answer)
        self.assertEqual(len(model.seen), 2)
        surfaces = str(model.seen) + result.answer + followup.answer
        for value in ("private@example.com", "4111 1111 1111 1111", "192.168.1.1"):
            self.assertNotIn(value, surfaces)

    def test_custom_pii_rule_applies_to_model_calls_tools_and_report_body(self):
        self.sanitizer = LocalSanitizer(
            rules=(*DEFAULT_PII_RULES, RedactionRule("member_id", detector=r"\bMEMBER-\d{6}\b"))
        )
        app, model = self.app(
            [
                tool_call(QUERY),
                call("save_report", body="Report for MEMBER-333333"),
                AIMessage(content="Report for MEMBER-333333"),
            ],
            intent="REPORT",
        )
        self.analytics.execute = Mock(return_value=QueryResult(QUERY, ({"note": "MEMBER-222222"},)))
        result = app.ask(ChatRequest("Report for MEMBER-111111", "chat"))
        self.assertEqual(result.status, "complete", result.answer)
        report = self.store.get(result.report_id)
        surfaces = (
            str(model.seen),
            result.answer,
            report.body,
        )
        for surface in surfaces:
            self.assertIn("[REDACTED_MEMBER_ID]", surface)
            for value in ("MEMBER-111111", "MEMBER-222222", "MEMBER-333333"):
                self.assertNotIn(value, surface)

    def test_private_mentions_require_an_employee_selected_id_in_legacy_library(self):
        app, model = self.app(
            [
                call("show_report", report_id="contact"),
                echo_result,
                call("list_reports"),
                echo_result,
                call("delete_reports"),
                echo_result,
                call("delete_reports"),
                echo_result,
            ]
        )
        secret = "private@example.com"
        self.reports.save("contact", "chat", secret, [QUERY], f"Contact: {secret}")
        self.reports.save("other", "chat", "other@example.com", [QUERY], "Other report")
        result = app.ask(ChatRequest("Show", "chat", action="show", report_id="contact"))
        self.assertIn("[REDACTED_EMAIL]", result.answer)
        result = app.ask(ChatRequest("List", "chat", action="list", search=secret))
        self.assertIn("other", result.answer)
        self.assertIn("contact", result.answer)
        result = app.ask(ChatRequest("Delete", "chat", action="delete", search=secret))
        self.assertFalse(result.pending_action_id)
        self.assertIn("/delete", result.answer)
        self.assertIn("other", result.answer)
        self.assertIn("contact", result.answer)
        result = app.ask(ChatRequest("Delete", "chat", action="delete", report_id="contact"))
        self.assertEqual(result.status, "awaiting_confirmation")
        action = self.store.pending(result.pending_action_id)
        self.assertEqual(action.report_ids, ("contact",))
        self.assertEqual(app.confirm("chat", action.id, False).status, "complete")
        self.assertEqual(len(model.seen), 8)
        self.assertNotIn(secret, str(model.seen))
        self.assertNotIn(secret, (self.path / "trace.jsonl").read_text())

    def test_graph_failure_message_is_clean_before_final_output(self):
        app, _ = self.app()
        with patch.object(
            app.runner.graph, "invoke", side_effect=InvalidRequest("private@example.com")
        ):
            result = app.ask(ChatRequest("Question", "chat"))
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.answer, "[REDACTED_EMAIL]")

    def test_shared_call_limit_includes_followup_after_tools(self):
        self.config.max_model_calls = 2
        app, model = self.app([tool_call(QUERY), AIMessage(content="Must not run")])
        result = app.ask(ChatRequest("What were sales?", "chat"))
        self.assertEqual(result.status, "failed")
        events = [json.loads(line) for line in (self.path / "trace.jsonl").read_text().splitlines()]
        failure = next(event for event in events if event["event"] == "run_failed")
        self.assertEqual(failure["error_type"], "BudgetExceeded")
        self.assertEqual(len(model.seen), 1)

    def test_retrying_classification_does_not_label_generation_as_a_retry(self):
        app, _ = self.app([AIMessage(content="Ready")])
        steps = []
        with (
            observe_progress(steps.append),
            patch(
                "retail_ai_agent.infrastructure.agent.middleware.budget.count_gemini_input",
                side_effect=[TimeoutError(), 20, 20],
            ),
            patch("langchain.agents.middleware.model_retry.time.sleep"),
        ):
            response = app.ask(ChatRequest("Are you available?", "chat"))
        self.assertEqual(response.answer, "Ready")
        self.assertEqual(steps, ["classify", "backoff", "retry", "model"])

    def test_transient_retry_is_bounded_and_shares_budget(self):
        app, model = self.app([TimeoutError("private@example.com"), AIMessage(content="Schema")])
        with patch("langchain.agents.middleware.model_retry.time.sleep"):
            result = app.ask(ChatRequest("What data is there?", "chat"))
        self.assertEqual(result.answer, "Schema")
        events = [json.loads(line) for line in (self.path / "trace.jsonl").read_text().splitlines()]
        self.assertEqual(events[-1]["model_calls"], 3)
        self.assertEqual(len(model.seen), 2)
        self.assertNotIn("private@example.com", str(events))

    def test_access_denial_returns_every_tool_result_and_stops_later_queries(self):
        calls = AIMessage(
            content="",
            tool_calls=[
                {"id": "bad", "name": "sql_db_query", "args": {"query": "DELETE FROM users"}},
                {"id": "later", "name": "sql_db_query", "args": {"query": QUERY}},
            ],
        )
        app, model = self.app([calls, AIMessage(content="Must not run")])
        result = app.ask(ChatRequest("Show PII", "chat"))
        self.assertEqual(result.status, "failed")
        self.assertEqual(self.analytics.calls, [])
        self.assertEqual(len(model.seen), 1)
        events = [json.loads(line) for line in (self.path / "trace.jsonl").read_text().splitlines()]
        self.assertEqual(
            [e["call_id"] for e in events if e["event"] == "tool_result"], ["bad", "later"]
        )

    def test_malformed_tool_call_is_not_a_final_answer(self):
        malformed = AIMessage(
            content="Unverified answer",
            invalid_tool_calls=[
                {"name": "sql_db_query", "args": "{", "id": "bad", "error": "invalid JSON"}
            ],
        )
        app, _ = self.app([malformed])
        result = app.ask(ChatRequest("What were sales?", "chat"))
        self.assertEqual(result.status, "failed")
        self.assertNotIn("Unverified", result.answer)

    def test_report_command_cannot_bypass_off_topic_refusal(self):
        answer = "I can help with retail analytics, sales questions and saved reports."
        app, model = self.app([AIMessage(content="A poem about a lighthouse")], intent="UNRELATED")
        self.analytics.execute = Mock()
        result = app.ask(ChatRequest("Write a poem", "chat", action="report"))
        self.assertEqual(result.answer, answer)
        self.assertEqual(model.seen, [])
        self.analytics.execute.assert_not_called()

    def test_natural_language_report_library_uses_tools_without_sql(self):
        report_id = self.create_report()
        app, model = self.app(
            [
                call("list_reports", search="sales"),
                echo_result,
                call("delete_reports", conversation_only=True),
            ]
        )
        result = app.ask(ChatRequest("Find my sales reports", "chat"))
        self.assertIn(report_id, result.answer)
        preview = app.ask(ChatRequest("Delete this conversation's reports", "chat"))
        self.assertEqual(preview.status, "awaiting_confirmation")
        self.assertEqual(len(model.seen), 3)
        self.assertEqual(self.analytics.calls, [])

    def test_delete_by_literal_phrase_checks_filters_before_preparing(self):
        question = "Delete all reports mentioning Client X"
        app, _ = self.app(
            [
                call("delete_reports", search=question),
                call("delete_reports"),
                call("delete_reports", search="Client Y"),
                call("delete_reports", search="Client X"),
                echo_result,
            ]
        )
        self.reports.save("question", "chat", "CLIENT X", [QUERY], "January")
        self.reports.save("body", "chat", "Sales", [QUERY], "Q1 for Client X.")
        self.reports.save("outside", "other", "Client X", [QUERY], "February")
        self.reports.save("unrelated", "chat", "Client Y", [QUERY], "March")
        with patch.object(self.reports, "search", wraps=self.reports.search) as search:
            preview = app.ask(ChatRequest(question, "chat"))
        self.assertEqual(preview.status, "awaiting_confirmation", preview.answer)
        search.assert_called_once_with("Client X", "", report_id="")
        action = self.reports.pending(preview.pending_action_id, "chat")
        self.assertEqual(set(action.report_ids), {"question", "body", "outside"})
        self.assertIsNotNone(self.store.get("question"))
        app.confirm("chat", action.id, False)

    def test_employee_can_replace_selection_and_only_confirm_the_new_preview(self):
        app, model = self.app(
            [call("delete_reports", search="50%_off", conversation_only=True), echo_result]
        )
        for report_id, body in [("kept", "50%_OFF"), ("removed", "50%_off"), ("other", "50xxoff")]:
            self.reports.save(report_id, "chat", "Sale", [QUERY], body)
        self.reports.save("different-chat", "elsewhere", "Sale", [QUERY], "50%_off")
        ReportService(self.store, Actor("b", "ceo")).save(
            "private", "chat", "Sale", [QUERY], "50%_off"
        )
        preview = app.ask(
            ChatRequest("Delete reports mentioning 50%_off from this conversation", "chat")
        )
        first = preview.pending_action_id
        self.assertEqual(set(self.reports.pending(first, "chat").report_ids), {"kept", "removed"})
        with self.assertRaises(InvalidRequest):
            app.reselect("chat", first, ["private"])
        self.assertEqual(self.reports.pending(first, "chat").status, "pending")
        options = app.reselect("chat", first, [])
        self.assertEqual(options.pending_action_id, first)
        self.assertIn("other", options.answer)
        self.assertNotIn("private", options.answer)
        with patch.object(self.store, "_write_index", side_effect=OSError("disk full")):
            failed = app.reselect("chat", first, ["other"])
        self.assertEqual(failed.pending_action_id, first)
        self.assertEqual(self.reports.pending(first, "chat").status, "pending")
        updated = app.reselect("chat", first, ["kept", "other", "kept"])
        self.assertNotEqual(updated.pending_action_id, first)
        self.assertEqual(updated.status, "awaiting_confirmation", updated.answer)
        self.assertIn("other", updated.answer)
        self.assertNotIn("removed", updated.answer)
        self.assertEqual(len(model.seen), 1)
        app.confirm("chat", first, True)
        self.assertEqual(len(self.store.list("a")), 4)
        app.confirm("chat", updated.pending_action_id, True)
        self.assertEqual({r.id for r in self.store.list("a")}, {"removed", "different-chat"})
        self.assertIsNotNone(self.store.get("private"))

    def test_delete_a_discussed_report_by_id_skips_reports_that_mention_it(self):
        app, model = self.app(
            [
                call("show_report", report_id="october"),
                echo_result,
                call("delete_reports", report_id="october"),
                echo_result,
            ]
        )
        self.reports.save("october", "original", "Revenue drop", [QUERY], "Sales fell by 10.")
        self.reports.save("notes", "other", "Notes", [QUERY], "See report october.")
        discussion = app.ask(ChatRequest("What does report october say?", "new-chat"))
        self.assertIn("Sales fell by 10", discussion.answer)
        preview = app.ask(ChatRequest("Delete that report", "new-chat"))
        self.assertEqual(preview.status, "awaiting_confirmation", preview.answer)
        self.assertEqual(
            self.reports.pending(preview.pending_action_id, "new-chat").report_ids, ("october",)
        )
        self.assertIsNotNone(self.store.get("october"))
        result = app.confirm("new-chat", preview.pending_action_id, True)
        self.assertEqual(result.status, "complete", result.answer)
        self.assertEqual(result.review_outcome, "Reports deleted: 1.")
        self.assertIn(result.review_outcome, model.seen[-1][0].text)
        self.assertIsNone(self.store.get("october"))
        self.assertIsNotNone(self.store.get("notes"))
        self.assertEqual(self.analytics.calls, [])
