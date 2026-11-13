import unittest
from unittest.mock import Mock

from fakes import LocalAnalytics, gemini_config
from google.api_core.exceptions import BadRequest, Forbidden
from google.api_core.exceptions import ServiceUnavailable as BigQueryDown

from retail_ai_agent.application.budget import Budget
from retail_ai_agent.application.errors import (
    AccessDenied,
    BudgetExceeded,
    QueryFailed,
    ServiceUnavailable,
)
from retail_ai_agent.domain.models import Actor
from retail_ai_agent.infrastructure.bigquery import BigQueryAnalytics
from retail_ai_agent.infrastructure.pricing import bigquery_on_demand_usd, gemini_rates
from retail_ai_agent.infrastructure.sql import SafeSql


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.local = LocalAnalytics(Actor("a", "manager", ("A",)))
        self.addCleanup(self.local.connection.close)
        self.sql = SafeSql("test.data", ("A",), self.local.catalog)

    def gateway(self, client):
        self.trace = Mock()
        return BigQueryAnalytics(
            client,
            gemini_config(google_cloud_project="test", bigquery_max_bytes=10_000_000),
            self.trace,
        )

    def test_dry_run_scan_limit_prevents_execution(self):
        client = Mock()
        client.query.return_value.total_bytes_processed = 20_000_000
        with self.assertRaises(QueryFailed):
            self.gateway(client).execute(
                self.sql.prepare("SELECT COUNT(*) FROM products"), Budget()
            )
        self.assertEqual(client.query.call_count, 1)
        self.assertTrue(client.query.call_args.kwargs["job_config"].dry_run)

    def test_bigquery_uses_parameters_and_server_side_billing_limit(self):
        client = Mock()
        dry = Mock(total_bytes_processed=100, location="US", reservation_id=None)
        job = Mock(total_bytes_billed=100, location="US", cache_hit=False, reservation_id=None)
        job.result.return_value = [{"total": 10}]
        client.query.side_effect = [dry, job]
        result = self.gateway(client).execute(
            self.sql.prepare("SELECT SUM(sale_price) AS total FROM order_items"), Budget()
        )
        self.assertEqual(result.rows, ({"total": 10},))
        configuration = client.query.call_args.kwargs["job_config"]
        self.assertEqual(configuration.maximum_bytes_billed, 10_000_000)
        self.assertIn("brand_0", [p.name for p in configuration.query_parameters])
        self.assertIn("__ra_order_items", result.sql)
        self.assertNotIn("x" * 32, result.sql)

    def test_query_error_never_returns_raw_provider_payload(self):
        client = Mock()
        client.query.side_effect = BadRequest("private@example.com")
        with self.assertRaises(QueryFailed) as error:
            self.gateway(client).execute(
                self.sql.prepare("SELECT COUNT(*) FROM products"), Budget()
            )
        self.assertNotIn("private", str(error.exception))

    def test_timeout_attempts_cancellation_and_keeps_cost_reservation(self):
        client = Mock()
        job = Mock()
        job.result.side_effect = TimeoutError("secret")
        client.query.side_effect = [
            Mock(total_bytes_processed=100, location="US", reservation_id=None),
            job,
        ]
        budget = Budget()
        with self.assertRaises(ServiceUnavailable):
            self.gateway(client).execute(self.sql.prepare("SELECT COUNT(*) FROM products"), budget)
        job.cancel.assert_called_once()
        self.assertGreater(budget.spent_usd, 0)

    def test_permission_failure_and_outage_do_not_replay_a_paid_job_or_leak_details(self):
        for failure, expected, unknown_cost in (
            (Forbidden("private backend detail"), AccessDenied, False),
            (BigQueryDown("private backend detail"), ServiceUnavailable, True),
        ):
            with self.subTest(failure=type(failure).__name__):
                client = Mock()
                dry = Mock(total_bytes_processed=100, location="US", reservation_id=None)
                client.query.side_effect = [dry, failure]
                budget = Budget()
                with self.assertRaises(expected) as err:
                    self.gateway(client).execute(
                        self.sql.prepare("SELECT id FROM products"), budget
                    )
                self.assertNotIn("private", str(err.exception))
                self.assertEqual(client.query.call_count, 2)
                self.assertEqual(budget.bigquery_usd, 0)
                self.assertEqual(budget.reserved_usd > 0, unknown_cost)

    def test_historical_run_at_standard_list_prices(self):
        rates = gemini_rates("models/gemini-3.5-flash-lite", "standard")
        self.assertEqual(rates.cost(41247, 12018, 2914), 0.01641424)
        self.assertEqual(bigquery_on_demand_usd("US", 3 * 41943040), 0.0007152557373046875)
        self.assertNotEqual(gemini_rates("gemini-3.5-flash", "standard"), rates)
        self.assertEqual(gemini_rates("gemini-3.1-flash-lite", "free").cost(41247, 0, 2914), 0)
        with self.assertRaises(ServiceUnavailable):
            gemini_rates("unpriced-model", "standard")
        with self.assertRaises(ServiceUnavailable):
            rates.cost(1, 2, 10)

    def test_cache_and_unknown_billing_metadata_have_distinct_outcomes(self):
        for cached, billed, amount, held in (
            (True, None, 0, False),
            (False, 41943040, 0.0002384185791015625, False),
            (False, None, None, True),
        ):
            with self.subTest(cached=cached, billed=billed):
                client = Mock()
                job = Mock(
                    location="US",
                    reservation_id=None,
                    total_bytes_billed=billed,
                    cache_hit=cached,
                    job_id="job-1",
                )
                job.result.return_value = []
                client.query.side_effect = [
                    Mock(total_bytes_processed=100, location="US", reservation_id=None),
                    job,
                ]
                budget = Budget(run_id="run-1")
                self.gateway(client).execute(self.sql.prepare("SELECT id FROM products"), budget)
                event = self.trace.record.call_args.kwargs
                self.assertEqual((event["job_id"], event["run_id"]), ("job-1", "run-1"))
                self.assertEqual(event["cost_usd"], amount)
                self.assertEqual(budget.reserved_usd > 0, held)
                self.assertEqual(budget.bigquery_usd, amount or 0)

    def test_unknown_region_and_capacity_pricing_stop_before_paid_query(self):
        for location, reservation in (("another-region", None), ("US", "reserved-slots")):
            client = Mock()
            client.query.return_value = Mock(
                total_bytes_processed=100, location=location, reservation_id=reservation
            )
            with self.assertRaises(ServiceUnavailable):
                self.gateway(client).execute(self.sql.prepare("SELECT id FROM products"), Budget())
            self.assertEqual(client.query.call_count, 1)

    def test_budget_stops_before_spending_or_starting_an_extra_attempt(self):
        budget = Budget(max_usd=0.1, max_model_calls=1, max_sql_attempts=1)
        with self.assertRaises(BudgetExceeded):
            budget.reserve(0.2)
        self.assertEqual(budget.spent_usd, 0)
        budget.model_call()
        budget.sql_attempt()
        with self.assertRaises(BudgetExceeded):
            budget.model_call()
        with self.assertRaises(BudgetExceeded):
            budget.sql_attempt()
        budget.started -= budget.timeout
        with self.assertRaises(BudgetExceeded):
            budget.remaining()
