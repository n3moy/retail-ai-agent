from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console

from retail_ai_agent.application.progress import observe_progress

_LABELS = {
    "model": "Preparing answer",
    "retry": "Retrying the model",
    "backoff": "Waiting before retrying the model",
    "sql_db_query": "Checking SQL and access",
    "dry_run": "Estimating the BigQuery scan",
    "bigquery": "Running the query in BigQuery",
    "no_data": "Query returned no usable data",
    "query_error": "SQL needs correction",
    "write_todos": "Updating the analysis plan",
    "list_reports": "Looking for reports",
    "show_report": "Reading the report",
    "save_report": "Saving the report",
    "report_saved": "Report saved",
    "delete_reports": "Checking report deletion",
}


@contextmanager
def cli_status(initial: str, *, steps: bool = False) -> Iterator[None]:
    console = Console(stderr=True)
    if steps:

        def print_step(operation: str) -> None:
            if label := _LABELS.get(operation):
                console.print(f"  · {label}", markup=False, highlight=False, soft_wrap=True)

        with observe_progress(print_step):
            yield
    elif not console.is_terminal or console.is_dumb_terminal:
        yield
    else:
        with (
            console.status(initial, spinner="dots") as status,
            observe_progress(lambda op: status.update(_LABELS.get(op, "Working"))),
        ):
            yield
