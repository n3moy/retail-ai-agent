"""Interactive commands: registration, help, and thin application handlers."""

from collections.abc import Callable
from dataclasses import dataclass
from re import fullmatch
from uuid import uuid4

from retail_ai_agent.application.chat import ChatService
from retail_ai_agent.application.errors import InvalidRequest
from retail_ai_agent.application.models import ChatRequest, ChatResponse


@dataclass
class CommandContext:
    service: ChatService
    conversation_id: str


@dataclass(frozen=True)
class CommandResult:
    response: ChatResponse | None = None
    message: str = ""
    exit_requested: bool = False


type CommandHandler = Callable[[CommandContext, str], CommandResult]


@dataclass(frozen=True)
class ChatCommand:
    name: str
    help: str
    argument: str
    required: bool
    handler: CommandHandler

    @property
    def usage(self) -> str:
        return f"{self.name} {self.argument}".rstrip()


class CommandRouter:
    def __init__(self) -> None:
        self._commands: dict[str, ChatCommand] = {}

    def command(
        self, name: str, *, help: str, argument: str = "", required: bool = False
    ) -> Callable[[CommandHandler], CommandHandler]:
        def register(handler: CommandHandler) -> CommandHandler:
            if name in self._commands:
                raise ValueError(f"Command already registered: {name}")
            self._commands[name] = ChatCommand(name, help, argument, required, handler)
            return handler

        return register

    def help(self) -> str:
        lines = (f"  {command.usage:<30} {command.help}" for command in self._commands.values())
        return "Enter a question or command:\n" + "\n".join(lines)

    def dispatch(self, context: CommandContext, text: str) -> CommandResult:
        parts = text.strip().split(maxsplit=1)
        if not parts:
            raise InvalidRequest("Enter a non-empty question.")
        name, argument = parts[0], parts[1] if len(parts) == 2 else ""
        if not name.startswith("/"):
            return CommandResult(context.service.ask(ChatRequest(text, context.conversation_id)))
        command = self._commands.get(name)
        if command is None:
            raise InvalidRequest("Unknown command. Use /help.")
        if (command.required and not argument) or (argument and not command.argument):
            raise InvalidRequest(f"Usage: {command.usage}")
        return command.handler(context, argument)


commands = CommandRouter()


@commands.command("/report", help="create and save a report", argument="<question>", required=True)
def report(context: CommandContext, argument: str) -> CommandResult:
    request = ChatRequest(argument, context.conversation_id, action="report")
    return CommandResult(context.service.ask(request))


@commands.command("/reports", help="search your reports", argument="[text]")
def reports(context: CommandContext, argument: str) -> CommandResult:
    request = ChatRequest(
        f"/reports {argument}".rstrip(), context.conversation_id, action="list", search=argument
    )
    return CommandResult(context.service.ask(request))


@commands.command("/show", help="read a report", argument="<report ID>", required=True)
def show(context: CommandContext, argument: str) -> CommandResult:
    request = ChatRequest(
        f"/show {argument}", context.conversation_id, action="show", report_id=argument
    )
    return CommandResult(context.service.ask(request))


@commands.command("/delete", help="prepare to delete your reports", argument="[report ID or text]")
def delete(context: CommandContext, argument: str) -> CommandResult:
    report_id = argument.lower() if fullmatch(r"[0-9a-fA-F]{32}", argument) else ""
    request = ChatRequest(
        f"/delete {argument}".rstrip(),
        context.conversation_id,
        action="delete",
        search="" if report_id else argument,
        report_id=report_id,
    )
    return CommandResult(context.service.ask(request))


@commands.command(
    "/delete-conversation", help="prepare to delete reports in this conversation", argument="[text]"
)
def delete_conversation(context: CommandContext, argument: str) -> CommandResult:
    request = ChatRequest(
        f"/delete-conversation {argument}".rstrip(),
        context.conversation_id,
        action="delete",
        search=argument,
        conversation_only=True,
    )
    return CommandResult(context.service.ask(request))


@commands.command(
    "/select",
    help="list available reports, or preview a replacement selection",
    argument="<action ID> [report IDs...]",
    required=True,
)
def select_reports(context: CommandContext, argument: str) -> CommandResult:
    action_id, *report_ids = argument.split()
    return CommandResult(context.service.reselect(context.conversation_id, action_id, report_ids))


@commands.command(
    "/confirm", help="confirm the previewed deletion", argument="<action ID>", required=True
)
def confirm(context: CommandContext, argument: str) -> CommandResult:
    return CommandResult(context.service.confirm(context.conversation_id, argument, True))


@commands.command("/cancel", help="cancel deletion", argument="<action ID>", required=True)
def cancel(context: CommandContext, argument: str) -> CommandResult:
    return CommandResult(context.service.confirm(context.conversation_id, argument, False))


@commands.command("/new", help="start a new conversation")
def new_conversation(context: CommandContext, argument: str) -> CommandResult:
    context.conversation_id = uuid4().hex
    return CommandResult(message=f"Conversation: {context.conversation_id}")


@commands.command("/help", help="show help")
def help_command(context: CommandContext, argument: str) -> CommandResult:
    return CommandResult(message=commands.help())


@commands.command("/exit", help="exit")
def exit_command(context: CommandContext, argument: str) -> CommandResult:
    return CommandResult(exit_requested=True)
