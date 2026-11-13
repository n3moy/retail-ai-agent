"""Mask result fields by SQL lineage, and redact free text before it reaches the model."""

import hmac
import json
from collections.abc import Sequence
from enum import IntEnum

import sqlglot
from langchain.agents.middleware import RedactionRule
from sqlglot import exp
from sqlglot.lineage import Node, lineage

from retail_ai_agent.application.errors import ServiceUnavailable
from retail_ai_agent.infrastructure.text_privacy import analyzer, redact_entities

# Redact complete URLs before processing any email or IP embedded in them.
DEFAULT_PII_RULES = tuple(
    RedactionRule(kind) for kind in ("url", "email", "credit_card", "ip", "mac_address")
)

PII_COLUMNS = {
    "email",
    "first_name",
    "last_name",
    "full_name",
    "phone",
    "phone_number",
    "street_address",
    "ip_address",
    "postal_code",
    "latitude",
    "longitude",
    "user_id",
    "customer_id",
    "date_of_birth",
    "birth_date",
}


class ColumnPrivacy(IntEnum):
    PUBLIC = 0
    CUSTOMER_ID = 1
    REDACT = 2


def pii_outputs(
    query: exp.Query, schema: dict[str, object]
) -> tuple[frozenset[str], frozenset[str]]:
    """Aliases, CTEs and expressions cannot hide a sensitive source column."""

    def policy(node: Node) -> ColumnPrivacy:
        # Direct PII overrides customer pseudonyms in mixed expressions.
        if isinstance(node.expression, exp.Table):
            column = sqlglot.parse_one(node.name, read="bigquery", into=exp.Column).name.lower()
            if column in {"user_id", "customer_id"} or (
                node.expression.name == "users" and column == "id"
            ):
                return ColumnPrivacy.CUSTOMER_ID
            return ColumnPrivacy.REDACT if column in PII_COLUMNS else ColumnPrivacy.PUBLIC
        # BigQuery can return a whole row via SELECT u FROM users u. Such row references
        # have no scalar-column lineage; mask the entire value rather than leak a STRUCT.
        if (
            isinstance(node.expression, exp.Placeholder)
            or node.expression.find(exp.TableColumn) is not None
            or any(not column.table for column in node.expression.find_all(exp.Column))
        ):
            return ColumnPrivacy.REDACT
        return max((policy(child) for child in node.downstream), default=ColumnPrivacy.PUBLIC)

    # COUNT and arithmetic over counts expose statistics, not identifiers. Replace them
    # only in the lineage copy; the executable SQL keeps the original expressions.
    analysis = query.transform(
        lambda node: exp.Literal.number(0) if isinstance(node, exp.Count) else node
    )
    outputs = {
        name.casefold(): policy(node)
        for name, node in lineage(None, analysis, schema=schema, dialect="bigquery").items()
    }
    return (
        frozenset(name for name, kind in outputs.items() if kind == ColumnPrivacy.REDACT),
        frozenset(name for name, kind in outputs.items() if kind == ColumnPrivacy.CUSTOMER_ID),
    )


class LocalSanitizer:
    """Use LangChain's PII rules for text and keyed pseudonyms for SQL fields."""

    def __init__(
        self,
        *,
        pii_key: str = "",
        rules: Sequence[RedactionRule] = DEFAULT_PII_RULES,
        brands: Sequence[str] = (),
    ):
        self.pii_key = pii_key
        self.brands = frozenset(brand.casefold() for brand in brands)
        self._rules = tuple(rule.resolve() for rule in rules)
        try:
            analyzer()
        except Exception:  # noqa: BLE001 — model loader diagnostics can include local paths.
            raise ServiceUnavailable(
                "Could not load the local privacy model. Run uv sync --locked."
            ) from None

    def mask_row(
        self, row: dict, pii_columns: frozenset[str], customer_ids: frozenset[str]
    ) -> dict:
        def release(key, value):
            if value is None:
                return None
            if key.casefold() in pii_columns:
                return "[REDACTED]"
            if key.casefold() not in customer_ids:
                return self._field(key, value)
            if not self.pii_key:
                return "[PII]"
            token = hmac.new(self.pii_key.encode(), str(value).encode(), "sha256").hexdigest()[:24]
            return f"[PII:{token}]"

        return {key: release(key, value) for key, value in row.items()}

    def _field(self, key, item):
        if key == "brand" and isinstance(item, str) and item.casefold() in self.brands:
            return item
        return self.value(item)

    def clean(self, text: str) -> str:
        """Apply the same redaction rules as PIIMiddleware at every application boundary."""
        if text.lstrip().startswith(("{", "[")):
            # Tool observations and legacy function-call arguments can contain JSON
            # Unicode escapes. Inspect decoded strings just as Gemini will.
            try:
                payload = json.loads(text)
            except ValueError:
                pass
            else:
                cleaned = self.value(payload)
                return text if payload == cleaned else json.dumps(cleaned, ensure_ascii=False)
        try:
            for rule in self._rules:
                text, _ = rule.apply(text)
            return redact_entities(text, self.brands)
        except Exception:  # noqa: BLE001 — detector errors can contain the raw input.
            raise ServiceUnavailable(
                "Local privacy check failed. Please retry the request."
            ) from None

    def value(self, value):
        if isinstance(value, str):
            return self.clean(value)
        if isinstance(value, dict):
            return {
                self.clean(key) if isinstance(key, str) else key: self._field(key, item)
                for key, item in value.items()
            }
        if isinstance(value, (tuple, list)):
            return [self.value(item) for item in value]
        return value
