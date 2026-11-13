"""Read-only SQL, PII lineage, and brand predicates applied before user aggregates."""

import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, SqlglotError, TokenError
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

from retail_ai_agent.application.errors import InvalidRequest, QueryFailed, UnsafeQuery
from retail_ai_agent.application.models import PreparedQuery
from retail_ai_agent.infrastructure.privacy import pii_outputs
from retail_ai_agent.infrastructure.schema import SchemaCatalog

SAFE_FUNCTIONS = {
    "COUNT",
    "COUNT_IF",
    "SUM",
    "AVG",
    "MIN",
    "MAX",
    "ROUND",
    "ABS",
    "CEIL",
    "FLOOR",
    "COALESCE",
    "NULLIF",
    "IF",
    "CASE",
    "AND",
    "OR",
    "CAST",
    "TRY_CAST",
    "EXTRACT",
    "DATE",
    "TIMESTAMP",
    "DATETIME",
    "DATE_TRUNC",
    "TIMESTAMP_TRUNC",
    "DATETIME_TRUNC",
    "DATE_DIFF",
    "TIMESTAMP_DIFF",
    "DATETIME_DIFF",
    "DATE_ADD",
    "DATE_SUB",
    "TIMESTAMP_ADD",
    "TIMESTAMP_SUB",
    "CURRENT_DATE",
    "CURRENT_TIMESTAMP",
    "CURRENT_DATETIME",
    "TIME_TO_STR",
    "TS_OR_DS_TO_TIMESTAMP",  # SQLGlot inserts this cast for BigQuery FORMAT_TIMESTAMP.
    "STR_TO_DATE",
    "STR_TO_TIME",
    "UPPER",
    "LOWER",
    "CONCAT",
    "LENGTH",
    "SUBSTRING",
    "TRIM",
    "REPLACE",
    "SAFE_DIVIDE",
    "NULLIFZERO",
    "ROW_NUMBER",
    "RANK",
    "DENSE_RANK",
    "LAG",
    "LEAD",
    "FIRST_VALUE",
    "LAST_VALUE",
    "STDDEV",
    "STDDEV_POP",
    "STDDEV_SAMP",
    "VARIANCE",
    "VAR_POP",
    "VAR_SAMP",
}


class SafeSql:
    def __init__(self, dataset: str, brands: tuple[str, ...] | None, catalog: SchemaCatalog):
        self.catalog = catalog
        if not re.fullmatch(r"[\w-]+\.[\w]+", dataset):
            raise InvalidRequest("Invalid BIGQUERY_DATASET.")
        self.dataset = dataset
        self.brands = brands
        self.parameters = {f"brand_{i}": brand for i, brand in enumerate(brands or ())}

    def table_access(self, name: str) -> str | None:
        if self.brands is None:
            return "all"
        if name in {"orders", "users"}:
            return "order_items"
        columns = self.catalog.columns[name]
        if "brand" in columns:
            return "brand"
        if "product_id" in columns:
            return "product_id"
        return None

    def compile(self, sql: str) -> str:
        return self.prepare(sql).sql

    def prepare(self, sql: str) -> PreparedQuery:
        if len(sql) > 20000:
            raise UnsafeQuery("The SQL query is too long.")
        try:
            statements = sqlglot.parse(sql, read="bigquery")
        except ParseError, TokenError:
            raise QueryFailed("Syntax error. Correct the GoogleSQL query.") from None
        if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.Union)):
            raise UnsafeQuery("Only one SELECT query is allowed.")
        tree = normalize_identifiers(statements[0], dialect="bigquery")
        for node in tree.walk():
            if isinstance(
                node, (exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.DDL, exp.Command)
            ):
                raise UnsafeQuery("Only read-only queries are allowed.")
            if isinstance(node, exp.Select) and node.args.get("kind"):
                raise UnsafeQuery("Select regular columns without SELECT AS STRUCT/VALUE.")
            if isinstance(
                node,
                (
                    exp.Parameter,
                    exp.Placeholder,
                    exp.Dot,
                    exp.Into,
                    exp.Unnest,
                    exp.Pivot,
                    exp.TableSample,
                    exp.Lock,
                ),
            ):
                raise UnsafeQuery("This SQL construct is not supported.")
            if isinstance(node, exp.Identifier) and node.name.startswith("__"):
                raise UnsafeQuery("Reserved names are not accessible.")
            if isinstance(node, exp.With) and node.args.get("recursive"):
                raise UnsafeQuery("Recursive queries are not supported.")
            if isinstance(node, exp.Func) and node.sql_name() not in SAFE_FUNCTIONS:
                raise UnsafeQuery("This function is not available in the analytics schema.")
            if isinstance(node, exp.Table) and (
                not isinstance(node.this, exp.Identifier)
                or any(
                    value
                    for key, value in node.args.items()
                    if key not in {"this", "alias", "db", "catalog"}
                )
            ):
                raise UnsafeQuery("This SQL source construct is not supported.")
            if isinstance(node, exp.Table) and (node.db or node.catalog):
                if f"{node.catalog}.{node.db}" != self.dataset:
                    raise UnsafeQuery("Use tables from the connected dataset.")
                node.set("db", None)
                node.set("catalog", None)
        for scope in traverse_scope(tree):
            for source in scope.sources.values():
                if isinstance(source, exp.Table) and source.name not in self.catalog.columns:
                    raise QueryFailed(
                        "The table does not exist in the connected database. Check the schema."
                    )
        try:
            schema: dict[str, object] = dict(self.catalog.columns)
            tree = qualify(tree, dialect="bigquery", schema=schema, identify=False)
            names = [name.casefold() for name in tree.named_selects]
            if len(names) != len(set(names)):
                raise QueryFailed("Use unique aliases for output columns.")
            masked, customer_ids = pii_outputs(tree, schema)
        except SqlglotError:
            raise QueryFailed("Unknown or ambiguous field. Check the schema and aliases.") from None

        # Resolve CTEs first: only physical sources are rewritten, never user CTE references.
        sources = set()
        for scope in traverse_scope(tree):
            select = scope.expression
            # A scalar SUM normally emits one synthetic row even on empty input.
            # Apply at its own SELECT level so CTEs and UNION arms keep the same rule.
            if (
                isinstance(select, exp.Select)
                and select.args.get("from_")
                and not select.args.get("group")
                and any(
                    not agg.find_ancestor(exp.Window)
                    for agg in scope.find_all(exp.Sum, exp.Avg, exp.Min, exp.Max)
                )
            ):
                select.having("COUNT(*) > 0", dialect="bigquery", copy=False)
            for source in scope.sources.values():
                if isinstance(source, exp.Table):
                    sources.add(source.name)
                    if self.brands is not None:
                        source.set("this", exp.to_identifier("__ra_" + source.name))
                    else:
                        project, dataset = self.dataset.split(".")
                        source.set("catalog", exp.to_identifier(project, quoted=True))
                        source.set("db", exp.to_identifier(dataset, quoted=True))
        filtered = self.brands is not None and bool(sources)
        if filtered:
            projections = sqlglot.parse_one(self.brand_projections(sources), read="bigquery").args[
                "with_"
            ]
            existing = tree.args.get("with_")
            if existing:
                projections.set("expressions", projections.expressions + existing.expressions)
            tree.set("with_", projections)
        return PreparedQuery(
            tree.sql(dialect="bigquery"), dict(self.parameters), masked, filtered, customer_ids
        )

    def brand_projections(self, sources: set[str]) -> str:
        """Keep original columns; scope the four retail relations before joins/aggregation."""
        parameters = ", ".join(f"@brand_{i}" for i in range(len(self.brands or ()))) or "NULL"
        dataset = self.dataset
        order_columns = ", ".join(
            "items.num_of_item AS num_of_item"
            if column == "num_of_item"
            else exp.column(column, table="o", quoted=True).sql(dialect="bigquery")
            for column in self.catalog.columns["orders"]
        )
        projections = {
            "products": f"SELECT p.* FROM `{dataset}.products` p WHERE p.brand IN ({parameters})",
            "order_items": f"SELECT i.* FROM `{dataset}.order_items` i JOIN __ra_products p ON p.id = i.product_id",
            "orders": f"""SELECT {order_columns} FROM `{dataset}.orders` o
                JOIN (SELECT order_id, COUNT(*) AS num_of_item FROM __ra_order_items GROUP BY order_id)
                items ON items.order_id = o.order_id""",
            "users": f"""SELECT u.* FROM `{dataset}.users` u WHERE EXISTS
                (SELECT 1 FROM __ra_order_items i WHERE i.user_id = u.id)""",
        }
        for name in sorted(sources - projections.keys()):
            access = self.table_access(name)
            table = exp.table_(
                name, db=dataset.split(".")[1], catalog=dataset.split(".")[0], quoted=True
            ).sql(dialect="bigquery")
            if access == "brand":
                projections[name] = f"SELECT t.* FROM {table} t WHERE t.brand IN ({parameters})"
            elif access == "product_id":
                projections[name] = (
                    f"SELECT t.* FROM {table} t JOIN __ra_products p ON p.id = t.product_id"
                )
            else:
                raise UnsafeQuery(
                    "This table has no known relationship to brands. "
                    "The analysis cannot be safely scoped."
                )
        ctes = ", ".join(f"__ra_{name} AS ({query})" for name, query in projections.items())
        return f"WITH {ctes} SELECT 1"
