import unittest
from unittest.mock import Mock, patch

from fakes import LocalAnalytics, gemini_config
from google.cloud import bigquery

from retail_ai_agent.application.errors import AccessDenied, ServiceUnavailable, UnsafeQuery
from retail_ai_agent.application.models import ChatRequest
from retail_ai_agent.domain.models import Actor
from retail_ai_agent.infrastructure.agent.context import RunContextFactory
from retail_ai_agent.infrastructure.bigquery import NO_RETRY
from retail_ai_agent.infrastructure.schema import (
    BigQuerySchemaLoader,
    DatasetMetadataClient,
    MetadataSQLDatabase,
    SchemaCatalog,
)
from retail_ai_agent.infrastructure.sql import SafeSql


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.local = LocalAnalytics(Actor("a", "manager", ("A",)))
        self.addCleanup(self.local.engine.dispose)

    def reload(self):
        self.local.catalog.load(MetadataSQLDatabase(self.local.engine, sample_rows_in_table_info=0))

    def test_context_exposes_reflected_schema_for_current_role_without_sample_rows(self):
        self.local.connection.executescript("""
            ALTER TABLE products ADD COLUMN promo_score INT;
            CREATE TABLE campaigns(brand TEXT, amount REAL);
            CREATE TABLE labels(product_id INT, label TEXT);
            CREATE TABLE employees(email TEXT);
            INSERT INTO employees VALUES('employee@example.com');
        """)
        self.reload()
        config = gemini_config(google_cloud_project="test")
        for brands in (("A",), None):
            with self.subTest(brands=brands):
                sql = SafeSql("test.data", brands, self.local.catalog)
                factory = RunContextFactory(config, sql)
                schema = factory.create("run", ChatRequest("Which tables?", "chat")).schema
                for field in ("campaigns(", "labels(", "promo_score INT64", "email STRING"):
                    self.assertIn(field, schema)
                self.assertEqual("employees(" in schema, brands is None)
                self.assertNotIn("employee@example.com", schema)
                self.assertNotIn("private@example.com", schema)

    def test_new_columns_are_queryable_and_stay_brand_scoped(self):
        self.local.connection.executescript("""
            ALTER TABLE products ADD COLUMN promo_score INT;
            UPDATE products SET promo_score = 7 WHERE brand = 'A';
            UPDATE products SET promo_score = 99 WHERE brand = 'B';
        """)
        self.reload()
        self.assertEqual(self.local.catalog.columns["products"]["promo_score"], "INT64")
        self.assertEqual(
            self.local.query("SELECT brand, promo_score FROM products").rows,
            ({"brand": "A", "promo_score": 7},),
        )

    def test_type_changes_are_reflected_without_python_or_policy_edits(self):
        self.local.connection.executescript("""
            ALTER TABLE products RENAME COLUMN cost TO previous_cost;
            ALTER TABLE products ADD COLUMN cost INT;
        """)
        self.reload()
        self.assertEqual(self.local.catalog.columns["products"]["cost"], "INT64")
        self.assertIn("previous_cost", self.local.catalog.columns["products"])

    def test_contact_columns_are_masked_through_aliases_without_text_recognition(self):
        self.local.connection.executescript("""
            ALTER TABLE users ADD COLUMN phone TEXT;
            ALTER TABLE users ADD COLUMN street_address TEXT;
            UPDATE users SET phone = '+1 202-555-0147', street_address = '123 Main Street';
        """)
        self.reload()
        (row,) = self.local.query(
            "SELECT first_name AS a, phone AS b, street_address AS c FROM users"
        ).rows
        for value in row.values():
            self.assertEqual(value, "[REDACTED]")

    def test_new_table_with_brand_is_automatically_filtered_before_aggregation(self):
        self.local.connection.executescript("""
            CREATE TABLE promotions(brand TEXT, amount INT);
            INSERT INTO promotions VALUES('A', 7), ('B', 99);
        """)
        self.reload()
        self.assertEqual(
            self.local.query("SELECT SUM(amount) AS total FROM promotions").rows, ({"total": 7},)
        )

    def test_unknown_brand_relationship_does_not_hide_schema_or_restrict_ceo(self):
        self.local.connection.executescript("""
            CREATE TABLE employees(email TEXT);
            INSERT INTO employees VALUES('private@example.com');
        """)
        self.reload()
        self.assertIn("email", self.local.catalog.database.get_table_info(["employees"]))
        with self.assertRaisesRegex(UnsafeQuery, "relationship to brands"):
            self.local.query("SELECT email FROM employees")
        self.local.sql = SafeSql("test.data", None, self.local.catalog)
        row = self.local.query("SELECT email FROM employees").rows[0]
        self.assertEqual(row["email"], "[REDACTED]")

    def test_failed_reflection_does_not_leave_a_usable_stale_catalog(self):
        database = Mock()
        database.get_usable_table_names.return_value = ["users"]
        database.get_table_info.side_effect = ValueError("bad metadata")
        with self.assertRaises(ValueError):
            self.local.catalog.load(database)
        with self.assertRaises(ServiceUnavailable):
            _ = self.local.catalog.database

    def test_bigquery_reflection_uses_real_langchain_without_querying_rows(self):
        client = Mock()
        client.get_table.side_effect = self.local.bigquery_table
        client.list_tables.side_effect = self.local.bigquery_tables
        config = gemini_config(
            google_cloud_project="billing",
            bigquery_dataset="source.data",
        )
        catalog = SchemaCatalog()
        BigQuerySchemaLoader(client, config, catalog).load(timeout=3)
        self.assertEqual(catalog.columns, self.local.catalog.columns)
        self.assertIn("email", catalog.database.get_table_info(["users"]))
        self.assertEqual(client.get_table.call_count, 4)
        client.list_tables.assert_called_once()
        for call in [*client.get_table.call_args_list, client.list_tables.call_args]:
            self.assertEqual(call.args[0].project, "source")
            self.assertEqual(call.args[0].dataset_id, "data")
            self.assertIs(call.kwargs["retry"], NO_RETRY)
            self.assertGreater(call.kwargs["timeout"], 0)
            self.assertLessEqual(call.kwargs["timeout"], 3)
        client.query.assert_not_called()
        client.list_datasets.assert_not_called()

    def test_metadata_client_uses_connected_dataset_and_startup_deadline(self):
        client = Mock()
        metadata = DatasetMetadataClient(client, "source.data", 3)
        for table in ("other.data.products", "source.other.products"):
            with self.subTest(table=table), self.assertRaises(AccessDenied):
                metadata.get_table(bigquery.TableReference.from_string(table))
        with patch(
            "retail_ai_agent.infrastructure.schema.monotonic", return_value=metadata.deadline + 1
        ):
            with self.assertRaises(TimeoutError):
                metadata.get_table(bigquery.TableReference.from_string("source.data.products"))
            with self.assertRaises(TimeoutError):
                metadata.list_tables(bigquery.DatasetReference("source", "data"))
        client.get_table.assert_not_called()
        client.list_tables.assert_not_called()
