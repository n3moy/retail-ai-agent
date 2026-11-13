import unittest

from fakes import LocalAnalytics
from langchain.agents.middleware import PIIMatch, RedactionRule

from retail_ai_agent.application.errors import QueryFailed, UnsafeQuery
from retail_ai_agent.domain.models import Actor
from retail_ai_agent.infrastructure.privacy import DEFAULT_PII_RULES, LocalSanitizer


class SecurityTests(unittest.TestCase):
    def analytics(self, role="manager", brands=("A",)):
        db = LocalAnalytics(Actor("test", role, brands))
        self.addCleanup(db.connection.close)
        return db

    def test_brand_scope_is_applied_before_aggregation(self):
        for role, brands, expected in [
            ("manager", ("A",), 10),
            ("manager", ("B",), 30),
            ("manager", ("A", "B"), 40),
            ("ceo", (), 40),
            ("manager", (), None),
        ]:
            with self.subTest(role=role, brands=brands):
                rows = (
                    self.analytics(role, brands)
                    .query("SELECT SUM(sale_price) AS total FROM order_items")
                    .rows
                )
                self.assertEqual(rows, () if expected is None else ({"total": expected},))

    def test_mixed_order_count_does_not_reveal_other_brands(self):
        self.assertEqual(
            self.analytics().query("SELECT num_of_item FROM orders").rows[0]["num_of_item"], 1
        )

    def test_monthly_revenue_accepts_bigquery_format_timestamp_and_keeps_brand_scope(self):
        db = self.analytics()
        result = db.query(
            "SELECT FORMAT_TIMESTAMP('%Y-%m', created_at) AS month, SUM(sale_price) AS revenue "
            "FROM order_items WHERE status='Complete' GROUP BY month"
        )
        self.assertEqual(result.rows, ({"month": "2024-01", "revenue": 10.0},))

    def test_ctes_joins_union_and_subqueries_stay_scoped(self):
        db = self.analytics()
        queries = [
            "WITH x AS (SELECT sale_price FROM order_items) SELECT SUM(sale_price) AS total FROM x",
            "SELECT SUM(sale_price) AS total FROM order_items WHERE product_id IN (SELECT id FROM products)",
            "SELECT SUM(sale_price) AS total FROM order_items UNION ALL SELECT SUM(sale_price) FROM order_items",
            "SELECT SUM(i.sale_price) AS total FROM order_items i JOIN products p ON i.product_id=p.id",
        ]
        for sql in queries:
            with self.subTest(sql=sql):
                self.assertTrue(all(row["total"] == 10 for row in db.query(sql).rows))

    def test_customer_fields_are_masked_for_managers_and_ceo(self):
        for role in ("manager", "ceo"):
            row = self.analytics(role).query("SELECT * FROM users").rows[0]
            self.assertEqual(
                set(row), {"id", "age", "gender", "state", "country", "email", "first_name"}
            )
            self.assertTrue(row["id"].startswith("[PII:"))
            self.assertEqual(row["email"], "[REDACTED]")
            self.assertEqual(row["first_name"], "[REDACTED]")
            self.assertEqual(row["age"], 35)
            self.assertNotIn("private", str(row).lower())

    def test_unsafe_sql_is_rejected_before_execution(self):
        db = self.analytics()
        for sql in [
            "SELECT * FROM INFORMATION_SCHEMA.TABLES",
            "SELECT * FROM EXTERNAL_QUERY('x','y')",
            "SELECT * FROM products; DELETE FROM products",
            "DELETE FROM products",
            "SELECT AS STRUCT * FROM users",
            "SELECT AS VALUE email FROM users",
            "SELECT @pii_salt FROM products",
            "SELECT * FROM __ra_users",
            "SELECT remote_function(brand) FROM products",
            "SELECT * FROM orders FOR SYSTEM_TIME AS OF '2024-01-01'",
            "SELECT * FROM products WHERE id IN (SELECT id FROM `other.ds.products`)",
            "WITH __ra_users AS (SELECT 1) SELECT * FROM __ra_users",
        ]:
            with self.subTest(sql=sql), self.assertRaises((UnsafeQuery, QueryFailed)):
                db.query(sql)
        self.assertEqual(db.calls, [])

    def test_pii_aliases_expressions_subqueries_and_union_are_masked(self):
        db = self.analytics()
        for query in (
            "SELECT email AS contact FROM users",
            "SELECT CONCAT(email, 'x') AS contact FROM users",
            "SELECT SUBSTRING(email, 1, 3) AS contact FROM users",
            "WITH t AS (SELECT email AS x FROM users) SELECT x AS contact FROM t",
            "SELECT (SELECT email FROM users LIMIT 1) AS contact FROM products",
            "SELECT MAX(id) AS contact FROM users",
            "SELECT COUNT(id) + id AS contact FROM users GROUP BY id",
            "SELECT COUNT(*) AS contact FROM users UNION ALL SELECT id FROM users",
            "SELECT CAST(user_id AS STRING) AS contact FROM order_items",
        ):
            with self.subTest(query=query):
                for row in db.query(query).rows:
                    if "email" in query:
                        self.assertEqual(row["contact"], "[REDACTED]")
                    else:
                        self.assertRegex(row["contact"], r"^\[PII:[0-9a-f]{24}\]$")

    def test_counts_and_customer_joins_keep_correct_values_and_stable_pseudonyms(self):
        db = self.analytics()
        rows = db.query(
            "SELECT u.id AS customer, i.user_id AS buyer, SUM(i.sale_price) AS total FROM users u JOIN order_items i ON u.id=i.user_id GROUP BY u.id, i.user_id"
        ).rows
        self.assertEqual(rows[0]["customer"], rows[0]["buyer"])
        self.assertEqual(rows[0]["total"], 10)
        self.assertEqual(
            db.query("SELECT COUNT(DISTINCT user_id) AS n FROM order_items").rows, ({"n": 1},)
        )
        self.assertEqual(
            db.query(
                "SELECT COUNT(DISTINCT user_id) * 100.0 / COUNT(*) AS rate FROM order_items"
            ).rows,
            ({"rate": 100.0},),
        )

    def test_fully_qualified_tables_in_connected_dataset_stay_scoped(self):
        rows = (
            self.analytics()
            .query("SELECT SUM(sale_price) AS total FROM `test.data.order_items`")
            .rows
        )
        self.assertEqual(rows, ({"total": 10},))

    def test_duplicate_aliases_cannot_hide_sensitive_result_columns(self):
        with self.assertRaises(QueryFailed):
            self.analytics().query("SELECT age AS x, email AS x FROM users")

    def test_predicate_cannot_override_bound_brand(self):
        rows = self.analytics().query("SELECT brand FROM products WHERE brand='B' OR 1=1").rows
        self.assertEqual(rows, ({"brand": "A"},))

    def test_builtin_pii_is_redacted_without_changing_brand_names(self):
        sanitizer = LocalSanitizer(brands=("Calvin Klein",))
        for value, kind in (
            ("private@example.com", "EMAIL"),
            ("4111 1111 1111 1111", "CREDIT_CARD"),
            ("192.168.1.1", "IP"),
            ("AA:BB:CC:DD:EE:FF", "MAC_ADDRESS"),
            ("https://example.com/customer/123", "URL"),
        ):
            with self.subTest(kind=kind):
                cleaned = sanitizer.clean(f"Calvin Klein: {value}")
                self.assertEqual(cleaned, f"Calvin Klein: [REDACTED_{kind}]")
                self.assertEqual(sanitizer.clean(cleaned), cleaned)

    def test_urls_are_redacted_before_embedded_pii(self):
        sanitizer = LocalSanitizer()
        for url in (
            "https://192.168.1.1/customer/secret",
            "https://example.com/customer?email=private@example.com",
        ):
            with self.subTest(url=url):
                self.assertEqual(sanitizer.clean(url), "[REDACTED_URL]")

    def test_analytics_text_is_preserved(self):
        sanitizer = LocalSanitizer(brands=("Calvin Klein",))
        for text in (
            "Sales of Calvin Klein in New York",
            "**Brand Scope**: Calvin Klein. Top-performing Calvin Klein items.",
            "# Calvin Klein Completed Sales - January 2024 (UTC)",
            "# Calvin Klein January 2024 Completed Sales Report",
            "Report ID: `df08eb77d8624d088353d8a315ba0353`",
            '{"brand":"Calvin Klein"}',
            "SELECT p.brand FROM products p WHERE p.brand = 'Calvin Klein'\nAND p.id = 1",
            "Revenue: 15731234567.89",
            "SELECT SUM(sale_price) FROM order_items WHERE status = 'Complete'",
            "SELECT COUNT(DISTINCT order_id) FROM order_items",
        ):
            with self.subTest(text=text):
                self.assertEqual(sanitizer.clean(text), text)

    def test_free_text_names_addresses_and_brand_context(self):
        sanitizer = LocalSanitizer(brands=("Calvin Klein",))
        for question, secrets in (
            (
                (
                    "Customer Calvin Klein, +1 (202) 555-0148, 42 Cedar Lane. "
                    "Compare Calvin Klein sales in New York for January 2024 at $123.45."
                ),
                ["555-0148", "Cedar"],
            ),
            (
                (
                    "Buyer Calvin Klein, phone 020 7946 0958, 7 Cedar Road, Unit 8. "
                    "Sales for Calvin Klein in New York, January 2024, $123.45."
                ),
                ["0958", "Cedar", "Unit 8"],
            ),
            (
                (
                    "Buyer John Smith. Contact Alice Morgan. Calvin Klein sales in New York, "
                    "January 2024, $123.45."
                ),
                ["John", "Smith", "Alice", "Morgan"],
            ),
        ):
            with self.subTest(question=question):
                safe = sanitizer.clean(question)
                self.assertEqual(safe.count("Calvin Klein"), 1)
                for secret in secrets:
                    self.assertNotIn(secret, safe)
                for value in ("New York", "2024", "$123.45"):
                    self.assertIn(value, safe)
                self.assertEqual(safe, sanitizer.clean(safe))

    def test_regex_rule_extends_defaults(self):
        sanitizer = LocalSanitizer(
            rules=(*DEFAULT_PII_RULES, RedactionRule("member_id", detector=r"\bMEMBER-\d{6}\b"))
        )
        self.assertEqual(
            sanitizer.clean("MEMBER-123456 private@example.com"),
            "[REDACTED_MEMBER_ID] [REDACTED_EMAIL]",
        )

    def test_callable_detector_extends_nested_value_cleaning(self):
        def detect_person(text: str) -> list[PIIMatch]:
            name = "John Smith"
            start = text.find(name)
            return (
                [PIIMatch(type="person", value=name, start=start, end=start + len(name))]
                if start >= 0
                else []
            )

        sanitizer = LocalSanitizer(
            rules=(*DEFAULT_PII_RULES, RedactionRule("person", detector=detect_person))
        )
        self.assertEqual(
            sanitizer.value({"rows": [{"contact": "John Smith private@example.com", "count": 2}]}),
            {"rows": [{"contact": "[REDACTED_PERSON] [REDACTED_EMAIL]", "count": 2}]},
        )

    def test_timestamps_are_preserved_during_text_redaction(self):
        text = "2024-01-15 12:00:00+00:00 / 2024-01-15T12:00:00Z / 2024-01-15"
        self.assertEqual(LocalSanitizer().clean(text), text)
