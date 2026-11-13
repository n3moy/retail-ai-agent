import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import ValidationError
from typer.testing import CliRunner

from retail_ai_agent.application.models import ChatRequest, ChatResponse
from retail_ai_agent.application.progress import progress
from retail_ai_agent.presentation.cli import cli


class CliTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.config = SimpleNamespace(default_user="default-user")
        self.config_factory = self.enterContext(
            patch(
                "retail_ai_agent.presentation.cli.Config.model_validate", return_value=self.config
            )
        )
        self.container_factory = self.enterContext(
            patch("retail_ai_agent.presentation.cli.make_app_container")
        )
        self.service = Mock()
        self.service.ask.return_value = ChatResponse("Answer", "chat")
        self.app_factory = self.enterContext(
            patch(
                "retail_ai_agent.presentation.cli.create_app",
                return_value=nullcontext(self.service),
            )
        )

    def test_report_commands_preserve_free_text_arguments(self):
        cases = [
            ("/report January sales", "report", "", "", False),
            ("/reports January sales", "list", "January sales", "", False),
            ("/show report123", "show", "", "report123", False),
            ("/delete January sales", "delete", "January sales", "", False),
            ("/delete " + "a1" * 16, "delete", "", "a1" * 16, False),
            ("/delete-conversation January", "delete", "January", "", True),
        ]
        for text, action, search, report_id, conversation_only in cases:
            with self.subTest(command=text):
                self.service.reset_mock()
                result = self.runner.invoke(cli, ["--conversation", "chat", text])
                self.assertEqual(result.exit_code, 0, result.output)
                self.service.ask.assert_called_once_with(
                    ChatRequest(
                        "January sales" if action == "report" else text,
                        "chat",
                        action=action,
                        search=search,
                        report_id=report_id,
                        conversation_only=conversation_only,
                    )
                )
                self.service.confirm.assert_not_called()

    def test_terminal_spinner_follows_operations_and_stops_on_failure(self):
        def failing_request(request):
            progress("list_reports")
            progress("sql_db_query")
            progress("model")
            raise OSError("private backend diagnostics")

        self.service.ask.side_effect = failing_request
        with patch("retail_ai_agent.presentation.progress.Console") as console:
            term = console.return_value
            term.is_terminal = True
            term.is_dumb_terminal = False
            status = term.status.return_value
            result = self.runner.invoke(cli, ["test question"])
        self.assertEqual(result.exit_code, 1)
        self.assertNotIn("private backend diagnostics", result.output)
        updates = status.__enter__.return_value.update.call_args_list
        self.assertEqual(
            [c.args[0] for c in updates],
            ["Looking for reports", "Checking SQL and access", "Preparing answer"],
        )
        self.assertEqual(status.__exit__.call_count, 2)  # startup and failed request

    def test_steps_show_operations_on_redirected_stderr_only(self):
        def answer(request):
            for event in (
                "model",
                "sql_db_query",
                "query_error",
                "private unknown tool",
                "backoff",
                "retry",
                "sql_db_query",
                "dry_run",
                "bigquery",
                "report_saved",
            ):
                progress(event)
            return ChatResponse("Revenue: 10", request.conversation_id)

        self.service.ask.side_effect = answer
        expected = [
            "Preparing answer",
            "Checking SQL and access",
            "SQL needs correction",
            "Waiting before retrying the model",
            "Retrying the model",
            "Checking SQL and access",
            "Estimating the BigQuery scan",
            "Running the query in BigQuery",
            "Report saved",
        ]
        for args, stdin in [(["--steps", "Sales?"], None), (["--steps"], "Sales?\n/exit\n")]:
            with self.subTest(args=args):
                result = self.runner.invoke(cli, args, input=stdin)
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(
                    [line.strip().removeprefix("· ") for line in result.stderr.splitlines()],
                    expected,
                )
                self.assertIn("Revenue: 10", result.stdout)
                self.assertNotIn("Checking SQL", result.stdout)

    def test_missing_command_arguments_and_unknown_commands_do_not_call_service(self):
        for text in ("/report", "/show", "/confirm", "/cancel", "/unknown", "/exit extra", "   "):
            with self.subTest(command=text):
                result = self.runner.invoke(cli, [text])
                self.assertEqual(result.exit_code, 1, result.output)
                self.assertTrue(result.stderr)
        self.service.ask.assert_not_called()
        self.service.confirm.assert_not_called()

    def test_confirmation_and_cancellation_have_explicit_handlers(self):
        self.service.confirm.return_value = ChatResponse("Done", "chat")
        for command, approved in (("/confirm", True), ("/cancel", False)):
            with self.subTest(command=command):
                self.service.confirm.reset_mock()
                result = self.runner.invoke(cli, ["--conversation", "chat", f"{command} action123"])
                self.assertEqual(result.exit_code, 0, result.output)
                self.service.confirm.assert_called_once_with("chat", "action123", approved)
        self.service.ask.assert_not_called()

    def test_select_command_lists_or_replaces_without_confirming(self):
        self.service.reselect.return_value = ChatResponse("Preview", "chat")
        for args, ids in [("action", []), ("action one two", ["one", "two"])]:
            with self.subTest(args=args):
                result = self.runner.invoke(cli, ["--conversation", "chat", f"/select {args}"])
                self.assertEqual(result.exit_code, 0, result.output)
                self.service.reselect.assert_called_with("chat", "action", ids)
        self.service.ask.assert_not_called()
        self.service.confirm.assert_not_called()

    def test_review_displays_the_recorded_outcome_even_if_model_prose_is_wrong(self):
        self.service.confirm.return_value = ChatResponse(
            "Please confirm deletion.", "chat", review_outcome="Deletion cancelled."
        )
        result = self.runner.invoke(cli, ["--conversation", "chat", "/cancel action123"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Review result: Deletion cancelled.", result.stdout)

    def test_session_commands_do_not_call_service_or_wait_for_input_in_one_shot_mode(self):
        for text in ("/help", "/new", "/exit"):
            with self.subTest(command=text), patch("builtins.input") as prompt:
                result = self.runner.invoke(cli, [text])
                self.assertEqual(result.exit_code, 0, result.output)
                prompt.assert_not_called()
                if text == "/help":
                    for command in ("/report", "/show", "/delete", "/confirm", "/cancel", "/new"):
                        self.assertIn(command, result.stdout)
        self.service.ask.assert_not_called()
        self.service.confirm.assert_not_called()

    def test_interactive_chat_recovers_from_invalid_commands(self):
        result = self.runner.invoke(cli, [], input="/show\n/unknown\nQuestion\n/exit\n")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Usage: /show", result.stderr)
        self.assertIn("Unknown command", result.stderr)
        self.assertIn("Answer", result.stdout)
        self.service.ask.assert_called_once()

    def test_help_does_not_load_settings_or_start_services(self):
        for option in ("--help", "-h"):
            with self.subTest(option=option):
                result = self.runner.invoke(cli, [option])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("question", result.stdout.lower())
                for text in (
                    "Retail analytics chat",
                    "--user",
                    "--conversation",
                    "--install-completion",
                ):
                    self.assertIn(text, result.stdout)
        self.config_factory.assert_not_called()
        self.app_factory.assert_not_called()

    def test_invalid_arguments_exit_before_loading_settings(self):
        for args in (["--unknown"], ["--user"], ["first", "second"]):
            with self.subTest(args=args):
                result = self.runner.invoke(cli, args)
                self.assertEqual(result.exit_code, 2, result.output)
                self.assertTrue(result.stderr)
        self.config_factory.assert_not_called()
        self.app_factory.assert_not_called()

    def test_no_arguments_start_interactive_chat_with_default_user(self):
        result = self.runner.invoke(cli, [], input="Question\n/exit\n")
        self.assertEqual(result.exit_code, 0, result.output)
        self.container_factory.assert_called_once_with(self.config, "default-user")
        request = self.service.ask.call_args.args[0]
        self.assertEqual(request.question, "Question")
        self.assertTrue(request.conversation_id)
        self.assertIn("Answer", result.stdout)

    def test_question_and_options_work_in_either_order(self):
        options = ["--user", "manager_a", "--conversation", "chat"]
        for args in (["Question", *options], [*options, "Question"]):
            with self.subTest(args=args):
                self.service.ask.reset_mock()
                self.container_factory.reset_mock()
                result = self.runner.invoke(cli, args)
                self.assertEqual(result.exit_code, 0, result.output)
                self.container_factory.assert_called_once_with(self.config, "manager_a")
                self.service.ask.assert_called_once_with(ChatRequest("Question", "chat"))
                self.assertIn("Answer", result.stdout)
                self.assertEqual(result.stderr, "")

    def test_failed_response_sets_process_exit_code(self):
        self.service.ask.return_value = ChatResponse("Request failed", "chat", status="failed")
        result = self.runner.invoke(cli, ["Question"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Request failed", result.stderr)
        self.assertNotIn("Request failed", result.stdout)

    def test_invalid_configuration_reports_field_and_exits(self):
        self.config_factory.side_effect = ValidationError.from_exception_data(
            "Config", [{"type": "missing", "loc": ("google_cloud_project",), "input": {}}]
        )
        result = self.runner.invoke(cli, [])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Check settings: google_cloud_project", result.stderr)
        self.assertEqual(result.stdout, "")
        self.app_factory.assert_not_called()
