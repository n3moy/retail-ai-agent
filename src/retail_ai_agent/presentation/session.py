"""Terminal I/O and session lifetime. Command behavior lives in commands.py."""

from uuid import uuid4

import typer

from retail_ai_agent.application.chat import ChatService
from retail_ai_agent.application.errors import AppError, InvalidRequest
from retail_ai_agent.presentation.commands import CommandContext, CommandResult, CommandRouter
from retail_ai_agent.presentation.progress import cli_status


class ChatSession:
    def __init__(
        self,
        service: ChatService,
        commands: CommandRouter,
        conversation_id: str = "",
        *,
        steps: bool = False,
    ) -> None:
        self.context = CommandContext(service, conversation_id or uuid4().hex)
        self.commands = commands
        self.steps = steps

    def run(self, question: str = "") -> int:
        if question and not question.strip():
            raise InvalidRequest("Enter a non-empty question.")
        typer.echo(f"Conversation: {self.context.conversation_id}")
        if not question:
            typer.echo(self.commands.help())
        while True:
            try:
                text = (question or input("> ")).strip()
                if not text:
                    continue
                with cli_status("Preparing answer", steps=self.steps):
                    result = self.commands.dispatch(self.context, text)
                self._display(result)
                if result.exit_requested:
                    return 0
                if question:
                    if result.response and result.response.pending_action_id:
                        # The in-memory interrupt needs this process to stay alive.
                        question = ""
                    else:
                        return int(
                            result.response is not None and result.response.status == "failed"
                        )
            except AppError as error:
                typer.echo(str(error), err=True)
                if question:
                    return 1
            except EOFError, KeyboardInterrupt:
                typer.echo()
                return 0
            except Exception:  # noqa: BLE001 — keep the session usable after storage/trace failures.
                typer.echo(
                    "Internal error. Check access to local storage and retry the request.",
                    err=True,
                )
                if question:
                    return 1

    @staticmethod
    def _display(result: CommandResult) -> None:
        if result.message:
            typer.echo(result.message)
        if result.response:
            response = result.response
            typer.echo(response.answer, err=response.status == "failed")
            if response.review_outcome and response.review_outcome != response.answer:
                typer.echo(f"Review result: {response.review_outcome}")
            if response.pending_action_id:
                typer.echo(
                    f"List choices or change selection: /select {response.pending_action_id} [report IDs...]"
                )
                typer.echo(f"Confirm: /confirm {response.pending_action_id}")
                typer.echo(f"Cancel: /cancel {response.pending_action_id}")
