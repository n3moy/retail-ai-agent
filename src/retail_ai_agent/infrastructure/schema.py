"""Load the connected dataset's actual schema through LangChain, without sample rows."""

import warnings
from time import monotonic
from typing import Any, Literal, NoReturn

import sqlglot
from google.cloud import bigquery
from google.cloud.bigquery.dataset import DatasetListItem
from google.cloud.bigquery.table import TableListItem
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.exc import SAWarning
from sqlalchemy.sql import Executable
from sqlglot import exp

from retail_ai_agent.application.errors import (
    AccessDenied,
    InvalidRequest,
    ServiceUnavailable,
)
from retail_ai_agent.config import Config
from retail_ai_agent.infrastructure.bigquery import NO_RETRY


class SchemaCatalog:
    def __init__(self) -> None:
        self._columns: dict[str, dict[str, str]] | None = None
        self._database: SQLDatabase | None = None

    @property
    def database(self) -> SQLDatabase:
        if self._database is None:
            raise ServiceUnavailable("The schema has not been loaded yet. Restart the application.")
        return self._database

    @property
    def columns(self) -> dict[str, dict[str, str]]:
        if self._columns is None:
            raise ServiceUnavailable("The schema has not been loaded yet. Restart the application.")
        return self._columns

    def load(self, database: SQLDatabase) -> None:
        """Keep every reflected column; SQL validation and PII lineage use the same schema."""
        self._columns = None
        self._database = None
        columns = {}
        for name in database.get_usable_table_names():
            ddl = sqlglot.parse_one(database.get_table_info([name]), read=database.dialect)
            if not isinstance(ddl, exp.Create) or not isinstance(ddl.this, exp.Schema):
                raise InvalidRequest("Could not read the table structure.")
            fields = {}
            for field in ddl.this.expressions:
                if isinstance(field, exp.ColumnDef):
                    if not isinstance(field.kind, exp.DataType):
                        raise InvalidRequest("A column type is missing from the metadata.")
                    fields[field.name] = field.kind.sql(dialect="bigquery")
            columns[name] = fields
        self._columns = columns
        self._database = database


class MetadataSQLDatabase(SQLDatabase):
    """Native schema tools; query execution must pass through budget/access middleware."""

    def _execute(
        self,
        command: str | Executable,
        fetch: Literal["all", "one", "cursor"] = "all",
        *,
        parameters: dict[str, Any] | None = None,
        execution_options: dict[str, Any] | None = None,
    ) -> NoReturn:
        # Also prevents accidental use of a bare toolkit without our middleware.
        raise AccessDenied("The query must go through SQL middleware.")


class DatasetMetadataClient:
    """Reflect the configured dataset without enumerating the whole public project.

    LangChain reflects each listed table through get_table. All remote calls share one
    startup deadline. This adapter has no query method, so it cannot fetch sample data.
    """

    def __init__(self, client: bigquery.Client, dataset: str, timeout: float):
        self.client = client
        self.project, self.dataset_id = dataset.split(".")
        self.deadline = monotonic() + timeout
        self._cache: dict[str, bigquery.Table] = {}
        self._tables: list[TableListItem] | None = None

    def remaining(self) -> float:
        remaining = self.deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError
        return remaining

    def list_datasets(self, project: str) -> list[DatasetListItem]:
        if project != self.project:
            raise AccessDenied("The project is not accessible.")
        return [
            DatasetListItem(
                {"datasetReference": {"projectId": self.project, "datasetId": self.dataset_id}}
            )
        ]

    def list_tables(
        self, dataset: bigquery.DatasetReference, **kwargs: object
    ) -> list[TableListItem]:
        if dataset.project != self.project or dataset.dataset_id != self.dataset_id:
            raise AccessDenied("The dataset is not accessible.")
        if self._tables is None:
            self._tables = list(
                self.client.list_tables(dataset, retry=NO_RETRY, timeout=self.remaining())
            )
        return self._tables

    def get_table(self, table: bigquery.TableReference) -> bigquery.Table:
        if table.project != self.project or table.dataset_id != self.dataset_id:
            raise AccessDenied("The table is not accessible.")
        if table.table_id not in self._cache:
            self._cache[table.table_id] = self.client.get_table(
                table, retry=NO_RETRY, timeout=self.remaining()
            )
        return self._cache[table.table_id]


class BigQuerySchemaLoader:
    def __init__(self, client: bigquery.Client, config: Config, catalog: SchemaCatalog) -> None:
        self.client = client
        self.config = config
        self.catalog = catalog

    def load(self, timeout: float) -> None:
        dataset = self.config.bigquery_dataset
        metadata = DatasetMetadataClient(self.client, dataset, timeout)
        engine = create_engine(
            f"bigquery://{dataset.replace('.', '/', 1)}?user_supplied_client=true",
            connect_args={"client": metadata, "prefer_bqstorage_client": False},
        )
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Did not recognize type 'GEOGRAPHY' of column .*",
                    category=SAWarning,
                    module=r"sqlalchemy_bigquery\._types",
                )
                self.catalog.load(
                    MetadataSQLDatabase(
                        engine,
                        sample_rows_in_table_info=0,
                        indexes_in_table_info=False,
                        lazy_table_reflection=True,
                    )
                )
        finally:
            engine.dispose()
