import sqlite3
from dataclasses import replace

import sqlglot
from google.cloud import bigquery
from google.cloud.bigquery.table import TableListItem
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from sqlalchemy import create_engine

from retail_ai_agent.application.access import allowed_brands
from retail_ai_agent.application.models import QueryResult
from retail_ai_agent.config import Config
from retail_ai_agent.infrastructure.privacy import LocalSanitizer
from retail_ai_agent.infrastructure.schema import MetadataSQLDatabase, SchemaCatalog
from retail_ai_agent.infrastructure.sql import SafeSql


def gemini_config(**settings):
    return Config(
        _env_file=None,
        model_provider="gemini",
        google_ai_studio_api_key="unit-test",
        gemini_model="gemini-3.5-flash-lite",
        **settings,
    )


class ScriptedModel(BaseChatModel):
    responses: list = Field(default_factory=list)
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            response = response(messages)
        response = response.model_copy(deep=True)
        response.usage_metadata = {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
        return ChatResult(generations=[ChatGeneration(message=response)])


def tool_call(sql, call_id="query", *, for_report=False):
    args = {"query": sql}
    if for_report:
        args["for_report"] = True
    return AIMessage(content="", tool_calls=[{"id": call_id, "name": "sql_db_query", "args": args}])


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"id": name, "name": name, "args": args}])


def echo_result(messages):
    """Script a final reply using the real tool observation, including generated IDs."""
    return AIMessage(content=messages[-1].text)


def toolkit_tools(catalog):
    return [
        tool
        for tool in SQLDatabaseToolkit(db=catalog.database, llm=ScriptedModel()).get_tools()
        if tool.name != "sql_db_query_checker"
    ]


class LocalAnalytics:
    """Execute the production SQL projections against a mixed-brand SQLite fixture."""

    def __init__(self, actor):
        self.calls = []
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            CREATE TABLE products(id INT, category TEXT, brand TEXT, name TEXT,
                                  retail_price REAL, department TEXT, cost REAL);
            INSERT INTO products VALUES(1,'Jeans','A','jeans',10,'Men',5),
                                       (2,'Jeans','B','jeans',30,'Men',15);
            CREATE TABLE orders(order_id INT,user_id INT,status TEXT,created_at TEXT,
                                num_of_item INT, shipped_at TEXT,delivered_at TEXT,returned_at TEXT);
            INSERT INTO orders VALUES(100,1,'Complete','2024-01-15',2,NULL,NULL,NULL);
            CREATE TABLE users(id INT,age INT,gender TEXT,state TEXT,country TEXT,
                               email TEXT,first_name TEXT);
            INSERT INTO users VALUES(1,35,'M','California','USA','private@example.com','Private');
            CREATE TABLE order_items(id INT,order_id INT,user_id INT,product_id INT,status TEXT,
              sale_price REAL,created_at TEXT,shipped_at TEXT,delivered_at TEXT,returned_at TEXT);
            INSERT INTO order_items VALUES(1,100,1,1,'Complete',10,'2024-01-15',NULL,NULL,NULL),
                                          (2,100,1,2,'Complete',30,'2024-01-15',NULL,NULL,NULL);
        """)

        self.engine = create_engine("sqlite://", creator=lambda: self.connection)
        self.catalog = SchemaCatalog()
        self.catalog.load(MetadataSQLDatabase(self.engine, sample_rows_in_table_info=0))
        self.sql = SafeSql("test.data", allowed_brands(actor), self.catalog)
        self.sanitizer = LocalSanitizer(pii_key="test-key-that-is-not-a-real-secret")

    def query(self, sql):
        prepared = self.sql.prepare(sql)
        result = self.execute(prepared)
        return replace(
            result,
            rows=tuple(
                self.sanitizer.mask_row(row, prepared.pii_columns, prepared.customer_ids)
                for row in result.rows
            ),
        )

    def bigquery_tables(self, dataset, **kwargs):
        return [
            TableListItem({"tableReference": dataset.table(name[0]).to_api_repr(), "type": "TABLE"})
            for name in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]

    def bigquery_table(self, reference, **kwargs):
        types = {"INT": "INTEGER", "TEXT": "STRING", "REAL": "FLOAT"}
        columns = self.connection.execute(f'PRAGMA table_info("{reference.table_id}")').fetchall()
        fields = [bigquery.SchemaField(row["name"], types[row["type"]]) for row in columns]
        return bigquery.Table(reference, schema=fields)

    def execute(self, prepared, budget=None):
        tree = sqlglot.parse_one(prepared.sql, read="bigquery")
        for table in tree.find_all(sqlglot.exp.Table):
            if table.catalog:
                table.set("db", None)
                table.set("catalog", None)
        self.calls.append(prepared.sql)
        cursor = self.connection.execute(tree.sql(dialect="sqlite"), prepared.parameters)
        return QueryResult(prepared.sql, tuple(dict(row) for row in cursor.fetchall()))
