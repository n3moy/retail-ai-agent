"""Published USD list rates for text requests, verified 2026-11-12."""

from dataclasses import dataclass
from decimal import Decimal

from retail_ai_agent.application.errors import ServiceUnavailable

GEMINI_PRICING = "https://ai.google.dev/gemini-api/docs/pricing"
BIGQUERY_PRICING = "https://cloud.google.com/bigquery/pricing#on_demand_pricing"


@dataclass(frozen=True)
class TokenRates:
    """USD per million tokens on the Gemini Developer API Standard tier."""

    input: float
    cached: float
    output: float

    def cost(self, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
        if not (0 <= cached_tokens <= input_tokens and output_tokens >= 0):
            raise ServiceUnavailable("The model returned inconsistent token usage.")
        return float(
            (
                (input_tokens - cached_tokens) * Decimal(str(self.input))
                + cached_tokens * Decimal(str(self.cached))
                + output_tokens * Decimal(str(self.output))
            )
            / 1_000_000
        )


GEMINI_TEXT_RATES = {
    "gemini-3.5-flash-lite": TokenRates(0.30, 0.03, 2.50),
    "gemini-3.5-flash": TokenRates(1.50, 0.15, 9.00),
    "gemini-3.1-flash-lite": TokenRates(0.25, 0.025, 1.50),
}


def gemini_rates(model: str, tier: str) -> TokenRates:
    rates = GEMINI_TEXT_RATES.get(model.removeprefix("models/"))
    if rates is None:
        raise ServiceUnavailable(
            "No verified Gemini text tariff for this model. Select a supported GEMINI_MODEL: "
            + ", ".join(GEMINI_TEXT_RATES)
        )
    return TokenRates(0, 0, 0) if tier == "free" else rates


def bigquery_on_demand_usd(location: str, billed_bytes: int) -> float:
    # The prototype dataset is in the US multi-region. Other SKUs require their own rate.
    if location != "US":
        raise ServiceUnavailable("No verified BigQuery on-demand tariff for this job location.")
    if billed_bytes < 0:
        raise ServiceUnavailable("BigQuery returned inconsistent billed bytes.")
    return billed_bytes * 6.25 / 2**40
