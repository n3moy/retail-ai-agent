from dataclasses import dataclass


@dataclass(frozen=True)
class Actor:
    user_id: str
    role: str
    brands: tuple[str, ...] = ()

    @property
    def scope(self) -> tuple[str, ...]:
        return (self.role, *sorted(self.brands))


@dataclass(frozen=True)
class Report:
    id: str
    owner_id: str
    conversation_id: str
    scope: tuple[str, ...]
    question: str
    sql: tuple[str, ...]
    body: str
    created_at: float


@dataclass(frozen=True)
class PendingDeletion:
    id: str
    owner_id: str
    conversation_id: str
    scope: tuple[str, ...]
    report_ids: tuple[str, ...]
    expires_at: float
    status: str = "pending"
    deleted_count: int = 0
