"""Per-invocation progress notifications. Subscribers receive only operation names."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_observer: ContextVar[Callable[[str], None] | None] = ContextVar("progress_observer", default=None)


@contextmanager
def observe_progress(observer: Callable[[str], None]) -> Iterator[None]:
    token = _observer.set(observer)
    try:
        yield
    finally:
        _observer.reset(token)


def progress(operation: str) -> None:
    if observer := _observer.get():
        observer(operation)
