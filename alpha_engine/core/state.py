"""Global process state models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GlobalState:
    """Shared runtime state for strategy, risk, and allocation layers."""

    equity: float
    drawdown: float
    current_vol: float
    active_positions: dict[str, Any] = field(default_factory=dict)

