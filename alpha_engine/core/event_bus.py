"""Simple in-process publish-subscribe event bus."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from alpha_engine.core.logger import get_logger

logger = get_logger(__name__)


class EventBus:
    """Event bus supporting multiple handlers per event type."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable[[Any], None]) -> None:
        """Register a handler for a given event type."""
        self._subscribers[event_type].append(handler)

    def publish(self, event_type: str, data: Any) -> None:
        """Publish data to all handlers; continue on handler errors."""
        logger.info("[TRACE] EVENT OUT %s: %r", event_type, data)
        for handler in self._subscribers.get(event_type, []):
            try:
                logger.info("[TRACE] EVENT IN %s: %r", event_type, data)
                handler(data)
            except Exception as exc:  # pragma: no cover - error path is behaviorally tested
                logger.exception(
                    "event handler failed for event_type=%s error=%s",
                    event_type,
                    exc,
                )

