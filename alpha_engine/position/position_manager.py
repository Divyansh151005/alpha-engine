"""Persistent discrete position manager for strategies."""

from __future__ import annotations

from alpha_engine.core.logger import get_logger

_logger = get_logger(__name__)


class PositionManager:
    """Track and update per-strategy discrete positions (-1, 0, +1)."""

    def __init__(
        self,
        entry_threshold: float = 0.40,
        exit_threshold: float = 0.65,
        strong_flip_threshold: float = 0.7,
        strong_exit_threshold: float = 0.6,
        min_hold_bars: int = 5,
        max_hold_bars: int = 50,
        hold_zone: float = 0.2,
        cost_buffer: float = 0.05,
    ) -> None:
        self._entry_threshold = abs(entry_threshold)
        self._exit_threshold = abs(exit_threshold)
        self._strong_flip_threshold = abs(strong_flip_threshold)
        self._strong_exit_threshold = abs(strong_exit_threshold)
        self._min_hold_bars = max(0, min_hold_bars)
        self._max_hold_bars = max(self._min_hold_bars, max_hold_bars)
        self._hold_zone = abs(hold_zone)
        self._cost_buffer = max(0.0, cost_buffer)
        self._positions: dict[str, int] = {}
        self._last_trade_ts: dict[str, int] = {}
        self._reversal_count: dict[str, int] = {}

    def get_position(self, strategy: str) -> int:
        return self._positions.get(strategy, 0)

    def get_all_positions(self) -> dict[str, int]:
        return dict(self._positions)

    def update_position(self, strategy: str, signal: float, current_time: int | None = None) -> int:
        """Update discrete strategy position using deterministic threshold rules."""
        original_signal = signal
        _logger.info(
            "[TRACE] PM input signal=%.4f current_position=%s",
            signal,
            self.get_position(strategy),
        )
        old = self._positions.get(strategy, 0)
        new = old
        last_trade_ts = self._last_trade_ts.get(strategy)
        held_bars = (
            (current_time - last_trade_ts)
            if (old != 0 and current_time is not None and last_trade_ts is not None)
            else 0
        )

        if old != 0 and current_time is not None and last_trade_ts is not None and (
            current_time - last_trade_ts
        ) < self._min_hold_bars:
            _logger.info(
                "[TRACE] after_min_hold signal=%.4f",
                signal,
            )
            _logger.info(
                "stability.decision action=hold reason=min_hold strategy=%s signal=%.4f current=%s last_trade_ts=%s min_hold_bars=%s",
                strategy,
                signal,
                old,
                last_trade_ts,
                self._min_hold_bars,
            )
            _logger.info(
                "[TRACE] FINAL target=%.4f (from original_signal=%.4f)",
                float(old),
                original_signal,
            )
            return old

        if old == 0:
            self._reversal_count[strategy] = 0
            required_entry = self._entry_threshold + self._cost_buffer
            _logger.info(
                "[TRACE] after_cost_adjustment signal=%.4f (required_entry=%.4f)",
                signal,
                required_entry,
            )
            if signal >= required_entry:
                new = 1
                _logger.info(
                    "[TRACE] after_threshold signal=%.4f",
                    signal,
                )
                _logger.info(
                    "stability.decision action=enter reason=entry_threshold strategy=%s side=long signal=%.4f threshold=%.4f cost_buffer=%.4f",
                    strategy,
                    signal,
                    self._entry_threshold,
                    self._cost_buffer,
                )
                _logger.info(
                    "position.enter strategy=%s side=long signal=%.4f",
                    strategy,
                    signal,
                )
            elif signal <= -required_entry:
                new = -1
                _logger.info(
                    "[TRACE] after_threshold signal=%.4f",
                    signal,
                )
                _logger.info(
                    "stability.decision action=enter reason=entry_threshold strategy=%s side=short signal=%.4f threshold=%.4f cost_buffer=%.4f",
                    strategy,
                    signal,
                    self._entry_threshold,
                    self._cost_buffer,
                )
                _logger.info(
                    "position.enter strategy=%s side=short signal=%.4f",
                    strategy,
                    signal,
                )
            else:
                _logger.info(
                    "[TRACE] after_threshold signal=%.4f (below entry)",
                    signal,
                )
                _logger.info(
                    "stability.decision action=hold reason=below_entry strategy=%s signal=%.4f threshold=%.4f cost_buffer=%.4f",
                    strategy,
                    signal,
                    self._entry_threshold,
                    self._cost_buffer,
                )
                _logger.info(
                    "position.hold strategy=%s side=flat signal=%.4f",
                    strategy,
                    signal,
                )
        else:
            if old == 1 and signal > -self._hold_zone:
                new = 1
                self._reversal_count[strategy] = 0
                _logger.info(
                    "[TRACE] after_hold_zone signal=%.4f",
                    signal,
                )
                _logger.info(
                    "stability.decision action=hold reason=hold_zone strategy=%s side=long signal=%.4f hold_zone=%.4f",
                    strategy,
                    signal,
                    self._hold_zone,
                )
                _logger.info(
                    "position.hold strategy=%s side=long signal=%.4f",
                    strategy,
                    signal,
                )
            elif old == -1 and signal < self._hold_zone:
                new = -1
                self._reversal_count[strategy] = 0
                _logger.info(
                    "[TRACE] after_hold_zone signal=%.4f",
                    signal,
                )
                _logger.info(
                    "stability.decision action=hold reason=hold_zone strategy=%s side=short signal=%.4f hold_zone=%.4f",
                    strategy,
                    signal,
                    self._hold_zone,
                )
                _logger.info(
                    "position.hold strategy=%s side=short signal=%.4f",
                    strategy,
                    signal,
                )
            elif old == 1 and signal < -self._strong_exit_threshold:
                count = self._reversal_count.get(strategy, 0) + 1
                self._reversal_count[strategy] = count
                if count >= 1:
                    new = 0
                    _logger.info(
                        "stability.decision action=exit reason=confirmed_reversal strategy=%s signal=%.4f reversal_count=%s threshold=%.4f exit_threshold=%.4f",
                        strategy,
                        signal,
                        count,
                        self._strong_exit_threshold,
                        self._exit_threshold,
                    )
                    _logger.info(
                        "position.exit strategy=%s old=%s signal=%.4f",
                        strategy,
                        old,
                        signal,
                    )
                else:
                    _logger.info(
                        "stability.decision action=hold reason=reversal_pending strategy=%s side=long signal=%.4f reversal_count=%s threshold=%.4f",
                        strategy,
                        signal,
                        count,
                        self._strong_exit_threshold,
                    )
            elif old == -1 and signal > self._strong_exit_threshold:
                count = self._reversal_count.get(strategy, 0) + 1
                self._reversal_count[strategy] = count
                if count >= 1:
                    new = 0
                    _logger.info(
                        "stability.decision action=exit reason=confirmed_reversal strategy=%s signal=%.4f reversal_count=%s threshold=%.4f exit_threshold=%.4f",
                        strategy,
                        signal,
                        count,
                        self._strong_exit_threshold,
                        self._exit_threshold,
                    )
                    _logger.info(
                        "position.exit strategy=%s old=%s signal=%.4f",
                        strategy,
                        old,
                        signal,
                    )
                else:
                    _logger.info(
                        "stability.decision action=hold reason=reversal_pending strategy=%s side=short signal=%.4f reversal_count=%s threshold=%.4f",
                        strategy,
                        signal,
                        count,
                        self._strong_exit_threshold,
                    )
            elif old == 1 and held_bars > self._max_hold_bars and signal < 0:
                self._reversal_count[strategy] = 0
                new = 0
                _logger.info(
                    "stability.decision action=exit reason=max_hold_timeout strategy=%s side=long signal=%.4f held_bars=%s max_hold_bars=%s",
                    strategy,
                    signal,
                    held_bars,
                    self._max_hold_bars,
                )
            elif old == -1 and held_bars > self._max_hold_bars and signal > 0:
                self._reversal_count[strategy] = 0
                new = 0
                _logger.info(
                    "stability.decision action=exit reason=max_hold_timeout strategy=%s side=short signal=%.4f held_bars=%s max_hold_bars=%s",
                    strategy,
                    signal,
                    held_bars,
                    self._max_hold_bars,
                )
            else:
                self._reversal_count[strategy] = 0
                _logger.info(
                    "stability.decision action=hold reason=inside_hysteresis strategy=%s side=%s signal=%.4f",
                    strategy,
                    "long" if old > 0 else "short",
                    signal,
                )
                _logger.info(
                    "position.hold strategy=%s side=%s signal=%.4f",
                    strategy,
                    "long" if old > 0 else "short",
                    signal,
                )

        self._positions[strategy] = new
        if new != old and current_time is not None:
            self._last_trade_ts[strategy] = current_time
        _logger.info(
            "[TRACE] after_scaling signal=%.4f (discrete new=%s; PM does not scale float signal)",
            signal,
            new,
        )
        _logger.info(
            "[TRACE] after_clamp signal=%.4f (discrete position in {-1,0,1})",
            float(new),
        )
        _logger.info(
            "[TRACE] FINAL target=%.4f (from original_signal=%.4f)",
            float(new),
            original_signal,
        )
        return new
