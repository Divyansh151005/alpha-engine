"""Tests for event bus behavior."""

from __future__ import annotations

from typing import Any

from alpha_engine.core.event_bus import EventBus


def test_publish_calls_multiple_subscribers() -> None:
    bus = EventBus()
    received: list[tuple[str, Any]] = []

    def handler_a(payload: Any) -> None:
        received.append(("a", payload))

    def handler_b(payload: Any) -> None:
        received.append(("b", payload))

    bus.subscribe("bar", handler_a)
    bus.subscribe("bar", handler_b)

    bus.publish("bar", {"price": 100.0})

    assert len(received) == 2
    assert received[0][0] == "a"
    assert received[1][0] == "b"


def test_publish_continues_if_one_subscriber_fails() -> None:
    bus = EventBus()
    received: list[str] = []

    def failing_handler(_: Any) -> None:
        raise RuntimeError("simulated handler failure")

    def healthy_handler(_: Any) -> None:
        received.append("ok")

    bus.subscribe("foo", failing_handler)
    bus.subscribe("foo", healthy_handler)

    bus.publish("foo", {"close": 10.0})

    assert received == ["ok"]

