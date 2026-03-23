"""Allocation models for capital distribution across strategies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PositionAllocation:
    """Per-strategy capital and risk allocation."""

    timestamp: int
    strategy_name: str
    direction: str
    capital_fraction: float
    risk_weight: float
