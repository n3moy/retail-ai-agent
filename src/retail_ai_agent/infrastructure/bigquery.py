from contextlib import suppress

from google.api_core import exceptions
from google.api_core.retry import Retry
from google.auth.exceptions import GoogleAuthError
from google.cloud import bigquery

from retail_ai_agent.application.budget import Budget
from retail_ai_agent.application.errors import (
    AccessDenied,
    BudgetExceeded,
    QueryFailed,
    ServiceUnavailable,
)
from retail_ai_agent.application.models import PreparedQuery, QueryResult
from retail_ai_agent.application.progress import progress
from retail_ai_agent.config import Config
from retail_ai_agent.infrastructure.pricing import bigquery_on_demand_usd
from retail_ai_agent.infrastructure.tracing import JsonlTrace

NO_RETRY = Retry(predicate=lambda _: False)


class BigQueryAnalytics:
    def __init__(self, client: bigquery.Client, config: Config, trace: JsonlTrace):
        self.client = client
        self.config = config
        self.trace = trace

    def execute(self, prepared: PreparedQuery, budget: Budget) -> QueryResult:
        query = f"SELECT * FROM ({prepared.sql}) AS result LIMIT {self.config.max_result_rows + 1}"
        parameters = [
            bigquery.ScalarQueryParameter(name, "STRING", value)
            for name, value in prepared.parameters.items()
        ]
        job = None
        reservation = 0.0
        try:
            progress("dry_run")
            dry = self.client.query(
                query,
                job_config=bigquery.QueryJobConfig(
                    dry_run=True, use_query_cache=False, query_parameters=parameters
                ),
                timeout=min(15, budget.remaining()),
                retry=NO_RETRY,
                job_retry=None,
            )
            if (dry.total_bytes_processed or 0) > self.config.bigquery_max_bytes:
                raise QueryFailed(
                    "The query reads too much data. Narrow the date range or selection."
                )
            if dry.reservation_id:
                raise ServiceUnavailable(
                    "This dataset uses capacity billing. An on-demand tariff cannot price this query."
                )
            reservation = bigquery_on_demand_usd(dry.location, self.config.bigquery_max_bytes)
            budget.reserve(reservation)
            progress("bigquery")
            job = self.client.query(
                query,
                job_config=bigquery.QueryJobConfig(
                    maximum_bytes_billed=self.config.bigquery_max_bytes,
                    query_parameters=parameters,
                ),
                timeout=min(15, budget.remaining()),
                retry=NO_RETRY,
                job_retry=None,
            )
            self.trace.record(
                "bigquery_submitted", run_id=budget.run_id, job_id=job.job_id, location=job.location
            )
            rows = list(job.result(timeout=budget.remaining(), retry=NO_RETRY, job_retry=None))
            actual = None
            if job.cache_hit:
                actual = 0.0
            elif not job.reservation_id and job.total_bytes_billed is not None:
                actual = bigquery_on_demand_usd(job.location, job.total_bytes_billed)
            if actual is not None:
                budget.reserved_usd -= reservation
                budget.bigquery_usd += actual
            self.trace.record(
                "bigquery_usage",
                run_id=budget.run_id,
                job_id=job.job_id,
                location=job.location,
                cache_hit=job.cache_hit,
                reservation_id=job.reservation_id,
                billed_bytes=job.total_bytes_billed,
                cost_usd=actual,
            )
            budget.remaining()
            return QueryResult(
                sql=query,
                rows=tuple(dict(row) for row in rows[: self.config.max_result_rows]),
                truncated=len(rows) > self.config.max_result_rows,
            )
        except exceptions.BadRequest:
            if job is not None and not job.error_result:
                raise ServiceUnavailable(
                    "Could not read the query result. Its billing status is still unknown."
                ) from None
            budget.reserved_usd -= reservation
            raise QueryFailed(
                "BigQuery rejected the SQL query. Check the syntax, types, and bytes billed limit."
            ) from None
        except exceptions.Forbidden:
            if job is None or job.error_result:
                budget.reserved_usd -= reservation
            raise AccessDenied(
                "BigQuery denied the request. Check dataset access, quotas and billing."
            ) from None
        except (
            TimeoutError,
            exceptions.GoogleAPICallError,
            GoogleAuthError,
            BudgetExceeded,
        ) as error:
            if job is not None:
                # Cancellation can fail independently; the original failure remains actionable.
                with suppress(TimeoutError, exceptions.GoogleAPICallError, GoogleAuthError):
                    job.cancel(retry=NO_RETRY, timeout=2)
            if isinstance(error, BudgetExceeded):
                raise
            raise ServiceUnavailable(
                "BigQuery is unavailable or timed out. Try again later."
            ) from None
