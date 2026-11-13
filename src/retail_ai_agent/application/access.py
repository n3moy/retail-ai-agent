from retail_ai_agent.application.errors import AccessDenied
from retail_ai_agent.domain.models import Actor


def allowed_brands(actor: Actor) -> tuple[str, ...] | None:
    """None means all brands; an empty tuple grants no data access."""
    if actor.role == "ceo":
        return None
    if actor.role == "manager":
        return actor.brands
    raise AccessDenied("Unknown employee role.")
