"""Backtest orchestration engine."""

from __future__ import annotations

from typing import Any

from alpha_engine.allocation.base import BaseAllocator
from alpha_engine.risk.base import BaseRiskManager
from alpha_engine.strategies.base import BaseStrategy


class BacktestEngine:
    """Coordinates strategy, allocation, and risk layers in simulation."""

    def __init__(
        self,
        strategy: BaseStrategy,
        allocator: BaseAllocator,
        risk_manager: BaseRiskManager,
    ) -> None:
        self.strategy = strategy
        self.allocator = allocator
        self.risk_manager = risk_manager

    def run(self, data: Any) -> None:
        """Run backtest loop (stub for future implementation)."""
        _ = data
        return None

