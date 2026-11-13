from contextlib import ExitStack
from typing import Annotated

import typer
from pydantic import ValidationError

from retail_ai_agent.application.errors import AppError
from retail_ai_agent.config import Config
from retail_ai_agent.create_app import create_app
from retail_ai_agent.ioc import make_app_container
from retail_ai_agent.presentation.commands import commands
from retail_ai_agent.presentation.progress import cli_status
from retail_ai_agent.presentation.session import ChatSession

cli = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_show_locals=False,
)


@cli.command(help="Retail analytics chat (Gemini / OpenRouter + BigQuery)")
def chat(
    question: Annotated[
        str,
        typer.Argument(help="Question or chat command; omit to start an interactive conversation."),
    ] = "",
    user: Annotated[
        str | None, typer.Option(help="Employee ID; defaults to DEFAULT_USER in settings.")
    ] = None,
    conversation: Annotated[
        str, typer.Option(help="ID for grouping reports; history lasts until the CLI closes.")
    ] = "",
    steps: Annotated[
        bool, typer.Option("--steps", help="Print the operations for each turn to stderr.")
    ] = False,
) -> None:
    try:
        config = Config.model_validate({})
        container = make_app_container(config, user or config.default_user)
        with ExitStack() as resources:
            with cli_status("Connecting to BigQuery"):
                app = resources.enter_context(create_app(container))
            session = ChatSession(app, commands, conversation, steps=steps)
            exit_code = session.run(question)
    except ValidationError as error:
        fields = ", ".join(str(item["loc"][0]) for item in error.errors())
        typer.echo(f"Check settings: {fields}", err=True)
        exit_code = 1
    except AppError as error:
        typer.echo(str(error), err=True)
        exit_code = 1
    except Exception:  # noqa: BLE001 — contain resource setup and teardown failures at the CLI boundary.
        typer.echo(
            "Could not open or close the session. Check local storage and try again.", err=True
        )
        exit_code = 1
    raise typer.Exit(exit_code)


def main() -> None:
    cli()
