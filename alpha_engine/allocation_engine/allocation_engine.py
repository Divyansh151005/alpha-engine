"""Dynamic capital allocator based on alpha signal strength and confidence."""

from __future__ import annotations

import math
from collections import deque

from alpha_engine.allocation_engine.allocation_models import PositionAllocation
from alpha_engine.alpha.alpha_signal import AlphaSignal
from alpha_engine.core.event_bus import EventBus
from alpha_engine.core.logger import get_logger

_EPS = 1e-9
_CORRELATION_WINDOW = 50
_logger = get_logger(__name__)


def _direction_sign(direction: str) -> float:
    if direction == "long":
        return 1.0
    if direction == "short":
        return -1.0
    return 0.0


def _corr(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((y - mb) ** 2 for y in b) / (len(a) - 1)
    denom = math.sqrt(va * vb)
    if denom <= 1e-12:
        if va <= 1e-12 and vb <= 1e-12:
            return 1.0 if (ma * mb >= 0) else -1.0
        return 0.0
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (len(a) - 1)
    return cov / denom


class CapitalAllocator:
    """Allocate capital across strategies with correlation-aware weight adjustment."""

    def __init__(self, event_bus: EventBus, correlation_window: int = _CORRELATION_WINDOW) -> None:
        self._event_bus = event_bus
        self._correlation_window = correlation_window
        self._event_bus.subscribe("alpha.signal", self._handle_signal)
        self._active_signals: dict[str, AlphaSignal] = {}
        self._signal_history: deque[dict[str, float]] = deque(maxlen=correlation_window)
        self._last_timestamp: int | None = None
        self._current_frame: dict[str, float] = {}
        self._all_strategies: set[str] = set()

    def _handle_signal(self, signal: AlphaSignal) -> None:
        ts = signal.timestamp
        sign = _direction_sign(signal.direction)

        if self._last_timestamp is not None and ts != self._last_timestamp:
            frame = {s: self._current_frame.get(s, 0.0) for s in self._all_strategies}
            if frame:
                self._signal_history.append(frame)

        self._all_strategies.add(signal.strategy_name)
        self._current_frame[signal.strategy_name] = sign
        self._last_timestamp = ts
        self._active_signals[signal.strategy_name] = signal
        self._recompute_and_publish()

    def _recompute_and_publish(self) -> None:
        if not self._active_signals:
            return

        allocations = self._compute_allocations()
        self._event_bus.publish("portfolio.allocation", allocations)

    def _compute_allocations(self) -> list[PositionAllocation]:
        signals = list(self._active_signals.values())
        if not signals:
            return []

        raw_weights = [max(0.0, s.strength * s.confidence) + _EPS for s in signals]
        names = [s.strategy_name for s in signals]

        if len(self._signal_history) >= 2:
            series = {n: [] for n in names}
            for frame in self._signal_history:
                for n in names:
                    series[n].append(frame.get(n, 0.0))

            avg_corrs: dict[str, float] = {}
            for i, name in enumerate(names):
                cors = []
                for j, other in enumerate(names):
                    if i != j and other in series and len(series[name]) == len(series[other]):
                        c = _corr(series[name], series[other])
                        cors.append(max(0.0, c))
                avg_corr = sum(cors) / len(cors) if cors else 0.0
                avg_corrs[name] = min(1.0, avg_corr)

            adjusted_weights = [
                rw * (1.0 - avg_corrs.get(n, 0.0))
                for rw, n in zip(raw_weights, names)
            ]
            for rw, adj, n in zip(raw_weights, adjusted_weights, names):
                _logger.info(
                    "allocation.correlation name=%s raw_weight=%.4f avg_corr=%.4f adjusted_weight=%.4f",
                    n,
                    rw,
                    avg_corrs.get(n, 0.0),
                    adj,
                )
            weights = [max(_EPS, w) for w in adjusted_weights]
        else:
            weights = raw_weights

        total = sum(weights)
        if total <= 0:
            total = 1.0

        allocations: list[PositionAllocation] = []
        for sig, weight in zip(signals, weights):
            capital_fraction = weight / total
            capital_fraction = max(0.0, min(1.0, capital_fraction))
            risk_weight = max(0.0, min(1.0, sig.confidence))
            allocations.append(
                PositionAllocation(
                    timestamp=sig.timestamp,
                    strategy_name=sig.strategy_name,
                    direction=sig.direction,
                    capital_fraction=capital_fraction,
                    risk_weight=risk_weight,
                )
            )

        total_alloc = sum(a.capital_fraction for a in allocations)
        if total_alloc > 1.0 + _EPS:
            scale = 1.0 / total_alloc
            allocations = [
                PositionAllocation(
                    timestamp=a.timestamp,
                    strategy_name=a.strategy_name,
                    direction=a.direction,
                    capital_fraction=a.capital_fraction * scale,
                    risk_weight=a.risk_weight,
                )
                for a in allocations
            ]

        return sorted(allocations, key=lambda a: a.strategy_name)
