"""Cooldown mechanism to prevent rapid consecutive trades."""

from __future__ import annotations

from alpha_engine.core.logger import get_logger

_logger = get_logger(__name__)


class CooldownTracker:
    """
    Track last trade timestamp/step per strategy.
    Prevents new trades within cooldown window (e.g. 5 candles).
    """

    def __init__(self, cooldown_candles: int = 5) -> None:
        self._cooldown_candles = cooldown_candles
        self._last_trade_step: dict[str, int] = {}

    def record_trade(self, strategy_names: list[str], step: int) -> None:
        """Record that a trade occurred for the given strategies at step."""
        for name in strategy_names:
            self._last_trade_step[name] = step

    def is_blocked(self, strategy_name: str, current_step: int) -> bool:
        """Return True if strategy is in cooldown (block new trades)."""
        last = self._last_trade_step.get(strategy_name)
        if last is None:
            return False
        return (current_step - last) < self._cooldown_candles

    def filter_blocked(
        self,
        strategy_names: list[str],
        current_step: int,
    ) -> set[str]:
        """
        Return set of strategy names that are blocked by cooldown.
        """
        return {
            name for name in strategy_names
            if self.is_blocked(name, current_step)
        }
