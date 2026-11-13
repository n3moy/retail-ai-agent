"""Local entity detection for prose crossing the agent boundary."""

import re
from functools import cache
from importlib.util import find_spec

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NerModelConfiguration, SpacyNlpEngine
from presidio_analyzer.predefined_recognizers import PhoneRecognizer, SpacyRecognizer

from retail_ai_agent.application.errors import ServiceUnavailable

# Ignore already-released placeholders during repeated scans of history and tool results.
REDACTION = re.compile(r"\[(?:REDACTED(?:_[A-Z_0-9]+)?|PII(?::[a-f0-9]{24})?)\]", re.IGNORECASE)
_REPORT_ID = re.compile(r"\b[0-9a-f]{32}\b", re.IGNORECASE)
_ADDRESS = re.compile(
    r"\b\d{1,6}[A-Za-z]?\s+(?:[\w.'’-]+[ \t]+){1,5}"
    r"(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr|boulevard|blvd|court|ct|way)\b"
    r"(?:\.?[, ]+(?:(?:apt\.?|suite|unit)\s*#?|#)\s*[\w-]+)?",
    re.IGNORECASE,
)
# NER can label the same name as a company or a person. A named customer or
# contact takes precedence over a catalog match at that particular occurrence.
_NAMED_PERSON = re.compile(
    r"(?i:\b(?:customer|buyer|client|contact|name|mr\.?|mrs\.?|ms\.?))"
    r"\s*[:=]?\s+([A-Z][a-z'’-]+(?:[ \t]+[A-Z][a-z'’-]+){0,2})\b"
)
_BRAND_PREFIX = re.compile(
    r"(?:\b(?:brand(?:\s+scope)?|sales\s+(?:of|for))|\bbrand\s*=)"
    r"[\s:*_\"'«]*$",
    re.IGNORECASE,
)
_BRAND_SUFFIX = re.compile(
    r"^[*_]*\s+(?:(?:completed|monthly|total|net|gross)\s+)?(?:sales|revenue|products?|items?|brand)\b",
    re.IGNORECASE,
)
_FIELD_LABELS = {"customer", "client", "contact", "buyer", "email"}
_SALES_CONTEXT = re.compile(r"\b(?:sales|revenue|brand|product|promotional)\b", re.IGNORECASE)
# The model can tag retail terms and the sample account label as people.
# The explicit customer-name rule above still wins in personal context.
_DOMAIN_TERMS = {
    "client x",
    "complete",
    "shipped",
    "processing",
    "cancelled",
    "returned",
    "report_id",
    "order_id",
}


@cache
def analyzer() -> AnalyzerEngine:
    # Presidio otherwise attempts to download missing models at runtime.
    model = "en_core_web_sm"
    if find_spec(model) is None:
        raise ServiceUnavailable("The local privacy model is missing. Run uv sync --locked.")
    config = NerModelConfiguration(
        labels_to_ignore=[
            "CARDINAL",
            "FAC",
            "LANGUAGE",
            "LAW",
            "MONEY",
            "ORDINAL",
            "PRODUCT",
            "QUANTITY",
            "PERCENT",
            "EVENT",
            "WORK_OF_ART",
            "MISC",
        ]
    )
    nlp = SpacyNlpEngine(
        models=[{"lang_code": "en", "model_name": model}],
        ner_model_configuration=config,
    )
    registry = RecognizerRegistry(supported_languages=["en"])
    registry.add_recognizer(SpacyRecognizer(supported_language="en"))
    registry.add_recognizer(
        PhoneRecognizer(supported_language="en", supported_regions=("US", "GB"))
    )
    return AnalyzerEngine(nlp_engine=nlp, registry=registry, supported_languages=["en"])


def redact_entities(text: str, brands: frozenset[str]) -> str:
    # Preserve character offsets. Never feed PII to an external detection service.
    def spaces(match: re.Match) -> str:
        return " " * len(match.group())

    visible = _REPORT_ID.sub(spaces, REDACTION.sub(spaces, text))
    if not visible.strip():
        return text
    spans = [(m.start(), m.end(), "ADDRESS") for m in _ADDRESS.finditer(visible)]
    spans.extend((m.start(1), m.end(1), "PERSON") for m in _NAMED_PERSON.finditer(visible))

    # Remove known business mentions from the NER input before tokenization.
    # An entity span from a prose model can otherwise swallow the closing SQL quote.
    for brand in brands:
        for m in re.finditer(r"(?<!\w)" + re.escape(brand) + r"(?!\w)", visible, re.IGNORECASE):
            prefix, suffix = visible[: m.start()], visible[m.end() :]
            # Report headings often put the month or quarter between brand and metric.
            heading_start = re.fullmatch(r"#{1,6}\s+[*_]*", prefix.rsplit("\n", 1)[-1])
            sales_heading = heading_start and _SALES_CONTEXT.search(suffix.partition("\n")[0])
            if (
                _BRAND_PREFIX.search(prefix)
                or _BRAND_SUFFIX.match(suffix)
                or sales_heading
                or (
                    _SALES_CONTEXT.search(prefix) and re.search(r"\bfor\s+$", prefix, re.IGNORECASE)
                )
                or (not prefix.strip() and suffix.startswith(":"))
            ):
                visible = prefix + " " * len(m.group()) + suffix

    for hit in analyzer().analyze(text=visible, language="en", entities=["PERSON", "PHONE_NUMBER"]):
        word = visible[hit.start : hit.end]
        if word.strip().casefold() in _FIELD_LABELS and REDACTION.match(
            text[hit.start + len(word.rstrip()) :].lstrip(" :")
        ):
            continue
        if hit.entity_type == "PERSON" and word.casefold() in _DOMAIN_TERMS:
            continue
        spans.append((hit.start, hit.end, hit.entity_type))

    # Merge overlapping matches to redact the entire address or name.
    merged: list[tuple[int, int, str]] = []
    for start, end, kind in sorted(spans, key=lambda s: (s[0], -s[1])):
        if merged and start < merged[-1][1]:
            left, right, first_kind = merged[-1]
            merged[-1] = (left, max(end, right), first_kind)
        else:
            merged.append((start, end, kind))
    for start, end, kind in reversed(merged):
        text = text[:start] + f"[REDACTED_{kind}]" + text[end:]
    return text
