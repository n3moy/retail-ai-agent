from collections.abc import Iterator
from contextlib import contextmanager

from dishka import Container

from retail_ai_agent.application.chat import ChatService
from retail_ai_agent.application.errors import AppError, ServiceUnavailable
from retail_ai_agent.domain.models import Actor
from retail_ai_agent.infrastructure.startup import StartupChecks


@contextmanager
def create_app(container: Container) -> Iterator[ChatService]:
    """Check readiness before serving; release resources even if startup fails."""
    try:
        try:
            container.get(Actor)  # Reject an invalid user before opening resources.
            container.get(StartupChecks).run()
            app = container.get(ChatService)
        except AppError:
            raise
        except Exception:  # noqa: BLE001 — never expose credentials from SDK errors.
            raise ServiceUnavailable(
                "Could not initialize the application. Check the configuration."
            ) from None
        yield app
    finally:
        container.close()
