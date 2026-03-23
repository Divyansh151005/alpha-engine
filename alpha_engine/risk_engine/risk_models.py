"""Risk state models for drawdown tracking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskState:
    """Current risk metrics: equity, peak, and drawdown."""

    equity: float
    peak_equity: float
    drawdown: float
