from collections.abc import Iterator
from contextlib import contextmanager, suppress
from time import monotonic

from google.cloud import bigquery

from retail_ai_agent.application.errors import ServiceUnavailable
from retail_ai_agent.infrastructure.bigquery import NO_RETRY
from retail_ai_agent.infrastructure.reports import FileReportStore
from retail_ai_agent.infrastructure.schema import BigQuerySchemaLoader
from retail_ai_agent.infrastructure.sql import SafeSql
from retail_ai_agent.infrastructure.tracing import JsonlTrace


@contextmanager
def startup_step(message: str) -> Iterator[None]:
    """Provider errors can contain credentials and query parameters."""
    try:
        yield
    except Exception:  # noqa: BLE001 — expose only the named dependency and recovery hint.
        raise ServiceUnavailable(message) from None


class StartupChecks:
    def __init__(
        self,
        client: bigquery.Client,
        sql: SafeSql,
        store: FileReportStore,
        trace: JsonlTrace,
        timeout: float,
        schema_loader: BigQuerySchemaLoader,
    ):
        self.client = client
        self.sql = sql
        self.store = store
        self.trace = trace
        self.timeout = timeout
        self.schema_loader = schema_loader

    def run(self) -> None:
        with startup_step(
            "Local storage is unavailable. Check the files and DATA_DIR permissions."
        ):
            self._storage()
        with startup_step(
            "BigQuery failed its readiness check. Check ADC, GOOGLE_CLOUD_PROJECT, "
            "BIGQUERY_DATASET, table access and job creation permissions, network, and timeout."
        ):
            self._bigquery()

    def _storage(self) -> None:
        self.store.check()
        with self.trace.path.open("a", encoding="utf-8"):
            pass

    def _bigquery(self) -> None:
        deadline = monotonic() + self.timeout

        def remaining() -> float:
            seconds = deadline - monotonic()
            if seconds <= 0:
                raise TimeoutError
            return seconds

        self.schema_loader.load(remaining())
        # Check the four relations required by the assignment, using the normal brand filter.
        query = self.sql.compile(
            " UNION ALL ".join(
                f"SELECT 1 FROM {table}" for table in ("products", "order_items", "orders", "users")
            )
        )
        self.client.query(
            query,
            job_config=bigquery.QueryJobConfig(
                dry_run=True,
                use_query_cache=False,
                query_parameters=[
                    bigquery.ScalarQueryParameter(name, "STRING", value)
                    for name, value in self.sql.parameters.items()
                ],
            ),
            retry=NO_RETRY,
            job_retry=None,
            timeout=remaining(),
        )
        # A real job also checks execution/result access; it scans no business tables.
        job = self.client.query(
            "SELECT 1 AS ready",
            job_config=bigquery.QueryJobConfig(use_query_cache=False, maximum_bytes_billed=1),
            retry=NO_RETRY,
            job_retry=None,
            timeout=remaining(),
        )
        try:
            rows = list(job.result(timeout=remaining(), retry=NO_RETRY, job_retry=None))
            if len(rows) != 1 or rows[0]["ready"] != 1:
                raise ValueError("Unexpected readiness result")
        except Exception:
            with suppress(Exception):
                job.cancel(retry=NO_RETRY, timeout=2)
            raise
