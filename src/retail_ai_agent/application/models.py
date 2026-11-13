from dataclasses import dataclass, field
from typing import Literal

Action = Literal["auto", "chat", "analysis", "report", "list", "show", "delete", "refuse"]


@dataclass(frozen=True)
class ChatRequest:
    question: str
    conversation_id: str
    action: Action = "auto"
    search: str = ""
    report_id: str = ""
    conversation_only: bool = False


@dataclass(frozen=True)
class ChatResponse:
    answer: str
    conversation_id: str
    status: str = "complete"
    pending_action_id: str = ""
    report_id: str = ""
    review_outcome: str = ""


@dataclass(frozen=True)
class PreparedQuery:
    sql: str
    parameters: dict[str, str] = field(repr=False)
    pii_columns: frozenset[str] = frozenset()
    brand_filtered: bool = False
    customer_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class QueryResult:
    sql: str
    rows: tuple[dict, ...]
    truncated: bool = False
