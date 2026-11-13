import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import Mock, patch

from retail_ai_agent.application.chat import ChatService
from retail_ai_agent.application.errors import InvalidRequest, ServiceUnavailable
from retail_ai_agent.application.models import ChatRequest, ChatResponse
from retail_ai_agent.presentation.commands import CommandContext, commands
from retail_ai_agent.presentation.session import ChatSession


class ChatTests(unittest.TestCase):
    def test_invalid_request_does_not_call_graph(self):
        runner = Mock()
        app = ChatService(runner)
        for question, conversation in [(" ", "chat"), ("Question", " "), ("x" * 8001, "chat")]:
            with self.assertRaises(InvalidRequest):
                app.ask(ChatRequest(question, conversation))
        runner.run.assert_not_called()

    def test_cli_keeps_session_after_failure_and_can_start_a_new_one(self):
        app = Mock()
        app.ask.side_effect = [
            ServiceUnavailable("Temporarily unavailable"),
            ChatResponse("First answer", "chat"),
            ChatResponse("Second answer", "chat"),
        ]
        output, errors = io.StringIO(), io.StringIO()
        with (
            patch(
                "builtins.input", side_effect=["Question", "Follow-up", "/new", "One more", "/exit"]
            ),
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            self.assertEqual(ChatSession(app, commands).run(), 0)
        calls = [call.args[0] for call in app.ask.call_args_list]
        self.assertEqual(calls[0].conversation_id, calls[1].conversation_id)
        self.assertNotEqual(calls[1].conversation_id, calls[2].conversation_id)
        self.assertIn("Temporarily unavailable", errors.getvalue())
        self.assertIn("First answer", output.getvalue())

    def test_one_shot_failed_status_returns_nonzero_exit_code(self):
        app = Mock()
        app.ask.return_value = ChatResponse("Request failed", "chat", status="failed")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(ChatSession(app, commands).run("Question"), 1)

    def test_confirmation_uses_separate_application_entry_point(self):
        app = Mock()
        commands.dispatch(CommandContext(app, "chat"), "/confirm action123")
        app.confirm.assert_called_once_with("chat", "action123", True)
        app.ask.assert_not_called()

    def test_one_shot_deletion_keeps_session_open_for_confirmation(self):
        app = Mock()
        app.ask.return_value = ChatResponse(
            "The reports will be deleted",
            "chat",
            status="awaiting_confirmation",
            pending_action_id="id",
        )
        app.confirm.return_value = ChatResponse("Deleted 1 report", "chat")
        output = io.StringIO()
        with (
            patch("builtins.input", side_effect=["/confirm id", "/exit"]) as prompt,
            redirect_stdout(output),
        ):
            self.assertEqual(ChatSession(app, commands, "chat").run("/delete"), 0)
        app.ask.assert_called_once()
        app.confirm.assert_called_once_with("chat", "id", True)
        self.assertEqual(prompt.call_count, 2)
        self.assertIn("Deleted 1 report", output.getvalue())
