"""Trade frequency cap to limit overtrading."""

from __future__ import annotations

from collections import deque

from alpha_engine.core.logger import get_logger

_logger = get_logger(__name__)


class TradeFrequencyTracker:
    """
    Enforce max_trades_per_100_steps per strategy.
    Block trades if exceeded in rolling 100-step window.
    """

    def __init__(
        self,
        max_trades_per_100_steps: int = 20,
        window_size: int = 100,
    ) -> None:
        self._max_trades = max_trades_per_100_steps
        self._window_size = window_size
        # strategy -> deque of step indices when trades occurred
        self._trade_steps: dict[str, deque[int]] = {}

    def record_trade(self, strategy_names: list[str], step: int) -> None:
        """Record a trade for the given strategies at step."""
        for name in strategy_names:
            if name not in self._trade_steps:
                self._trade_steps[name] = deque(maxlen=500)
            q = self._trade_steps[name]
            q.append(step)
            # Keep only steps within window for count
            while q and (step - q[0]) > self._window_size:
                q.popleft()

    def trade_count_in_window(self, strategy_name: str, current_step: int) -> int:
        """Number of trades in the last window_size steps."""
        q = self._trade_steps.get(strategy_name)
        if not q:
            return 0
        # Count steps in [current_step - window_size, current_step]
        start = current_step - self._window_size
        return sum(1 for s in q if s > start and s <= current_step)

    def is_blocked(self, strategy_name: str, current_step: int) -> bool:
        """Return True if strategy has exceeded trade cap in last 100 steps."""
        count = self.trade_count_in_window(strategy_name, current_step)
        return count >= self._max_trades

    def filter_blocked(
        self,
        strategy_names: list[str],
        current_step: int,
    ) -> set[str]:
        """Return set of strategy names blocked by frequency cap."""
        blocked = set()
        for name in strategy_names:
            if self.is_blocked(name, current_step):
                blocked.add(name)
                _logger.info(
                    "filter.blocked reason=frequency_limit strategy=%s trades_in_window=%s max=%s",
                    name,
                    self.trade_count_in_window(name, current_step),
                    self._max_trades,
                )
        return blocked
