"""Backtest orchestration engine."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

from alpha_engine.alpha import (
    AlphaEngine,
    AlphaSignal,
    BreakoutAlphaStrategy,
    MeanReversionAlphaStrategy,
    TrendAlphaStrategy,
    MomentumPersistenceAlphaStrategy,
)
from alpha_engine.alpha.base_alpha import BaseAlphaStrategy
from alpha_engine.core.event_bus import EventBus
from alpha_engine.core.logger import get_logger, trace_limit
from alpha_engine.data.models import OHLCV
from alpha_engine.features import FeatureEngine
from alpha_engine.filters import SignalFilter
from alpha_engine.regime import RegimeEngine
from alpha_engine.state_engine import MarketStateEngine, MarketStateVector

# Numerical stability
_EPS = 1e-12
_MIN_PRICE = 1e-12
_SIGNAL_DELTA_BASE_THRESHOLD = 0.03
_STRONG_SIGNAL_DELTA_BYPASS = 0.6
_SAME_DIRECTION_MIN_DELTA = 0.02
_STRONG_REVERSAL_SIGNAL_THRESHOLD = 0.6
_ENTRY_SIGNAL_THRESHOLD = 0.4
_EXIT_SIGNAL_THRESHOLD = 0.3
# Execution: trade only when current net signal is in top decile vs rolling history (not raw magnitude).
_SIGNAL_RANK_WINDOW = 50
_SIGNAL_RANK_MIN_PERCENTILE = 0.9

MAX_STEP = 0.2
HOLD_THRESHOLD = 0.2

# Execution friction (replay only): follow signal; avoid freezing on wrong side.
_EXEC_STRONG_MOVE_BYPASS = 0.4
# Same-direction: below this vs current position, hold (reduces micro-rebalance churn).
_EXEC_SAME_DIR_MIN_DELTA = 0.03

_logger = get_logger(__name__)

# After entry, ignore new signals until this many bars have passed (align hold with prediction horizon).
MIN_HOLD_STEPS = 12
# Default minimum bars between executed trades when ``min_steps_between_trades`` is not passed.
_DEFAULT_MIN_STEPS_BETWEEN_TRADES = 3


def _regime_scale_from_rr20_strength(regime_strength: float) -> float:
    """Scale exposure by |rr20| band: weak → smaller size, strong → full (no trade removal)."""
    if regime_strength < 0.002:
        return 0.3
    if regime_strength < 0.005:
        return 0.6
    return 1.0


def compute_target_position(
    signal: float,
    current_position: float,
    *,
    log_validation: bool = False,
    rolling_return_20: float | None = None,
) -> float:
    """Map |signal| to target fraction in [-0.8, 0.8]; flat when |signal| < 0.2 (linear scale).

    When ``rolling_return_20`` is set, multiply the signal-sized position by a regime factor
    from ``abs(rolling_return_20)``. When ``None``, regime scaling is skipped (factor 1.0).
    """
    _ = log_validation, current_position
    strength = abs(signal)
    if strength < 0.2:
        return 0.0
    scaled = (strength - 0.2) / (1.0 - 0.2)
    direction = 1.0 if signal > 0.0 else -1.0
    position = scaled * 0.8 * direction
    if rolling_return_20 is not None:
        rr20 = float(rolling_return_20)
        regime_strength = abs(rr20) if math.isfinite(rr20) else 0.0
        regime_scale = _regime_scale_from_rr20_strength(regime_strength)
        position = position * regime_scale
    return max(-1.0, min(1.0, position))


@dataclass
class BacktestResult:
    """Result of a backtest run."""

    equity_curve: list[tuple[int, float]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    trade_returns: list[float] = field(default_factory=list)


class StorageProtocol(Protocol):
    """Protocol for storage that can load candles."""

    def get_range(
        self,
        symbol: str,
        timeframe: str,
        start_ts: int,
        end_ts: int,
    ) -> list[OHLCV]:
        ...


def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation; 0.0 if undefined (constant series or length mismatch)."""
    n = len(xs)
    if n != len(ys) or n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    vx = sum((x - mx) ** 2 for x in xs) / n
    vy = sum((y - my) ** 2 for y in ys) / n
    denom = math.sqrt(max(vx, 0.0) * max(vy, 0.0))
    if denom < _EPS:
        return 0.0
    return max(-1.0, min(1.0, cov / denom))


def _sign_f(x: float) -> float:
    """Sign for directional match: -1, 0, or 1."""
    if x > _EPS:
        return 1.0
    if x < -_EPS:
        return -1.0
    return 0.0


def _empirical_percentile_rank_le(window: list[float], value: float) -> float:
    """Fraction of window values <= value (inclusive). Window non-empty."""
    n = len(window)
    if n == 0:
        return 0.0
    return sum(1 for x in window if x <= value) / n


_TREND_KEY = "TrendAlphaStrategy"
_MR_KEY = "MeanReversionAlphaStrategy"
_BRK_KEY = "BreakoutAlphaStrategy"
_ALPHA_DIAG_KEYS = (_MR_KEY, _BRK_KEY)


def _split_gross_pnl_by_signals(
    signals: dict[str, float], gross_pnl: float
) -> tuple[float, float, float]:
    """Attribute gross PnL by absolute filtered signal strength."""
    t = abs(float(signals.get(_TREND_KEY, 0.0)))
    m = abs(float(signals.get(_MR_KEY, 0.0)))
    b = abs(float(signals.get(_BRK_KEY, 0.0)))
    tot = t + m + b
    if tot < _EPS:
        return 0.0, 0.0, 0.0
    share = gross_pnl / tot
    return t * share, m * share, b * share


def compute_metrics(  # Exposed for testing
    equity_curve: list[tuple[int, float]],
    periods_per_year: float = 365.0 * 24.0 * 60.0,  # 1m candles
) -> dict[str, float]:
    """Compute total_return, CAGR, max_drawdown, sharpe_ratio. Numerically stable."""
    if len(equity_curve) < 2:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }

    initial = max(equity_curve[0][1], _EPS)
    final = max(equity_curve[-1][1], _EPS)
    total_return = (final - initial) / initial
    trace_limit("compute_metrics total_return clamp", "max(-1.0,min(1e6,...))")
    total_return = max(-1.0, min(1e6, total_return))  # clamp for stability

    t0 = equity_curve[0][0]
    t1 = equity_curve[-1][0]
    years = (t1 - t0) / 1000.0 / 3600.0 / 24.0 / 365.25
    if years < 1.0 / 365.25:  # Less than 1 day: use total_return
        cagr = total_return
    else:
        cagr = (final / initial) ** (1.0 / years) - 1.0
    cagr = max(-1.0, min(1e6, cagr))

    peak = initial
    max_dd = 0.0
    for _, eq in equity_curve:
        peak = max(peak, max(eq, _EPS))
        dd = (peak - eq) / max(peak, _EPS)
        max_dd = max(max_dd, dd)
    max_drawdown = max(0.0, min(1.0, max_dd))

    returns: list[float] = []
    for i in range(1, len(equity_curve)):
        prev = max(equity_curve[i - 1][1], _EPS)
        curr = max(equity_curve[i][1], _EPS)
        r = (curr - prev) / prev
        returns.append(max(-1.0, min(1e6, r)))

    if len(returns) < 2:
        sharpe = 0.0
    else:
        mean_ret = sum(returns) / len(returns)
        var = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(max(var, 0.0))
        if std < _EPS:
            sharpe = 0.0
        else:
            period_scale = math.sqrt(periods_per_year)
            sharpe = (mean_ret / std) * period_scale
            sharpe = max(-1e3, min(1e3, sharpe))

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
    }


class BacktestEngine:
    """Coordinates strategy, allocation, and risk layers in simulation.

    Loads historical candles from DB, replays them sequentially via the event bus,
    and simulates portfolio (cash, position, equity) with instant execution and no slippage.
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        initial_equity: float = 100.0,
        target_vol: float = 0.3,
        strategies: list[BaseAlphaStrategy] | None = None,
        diagnostics: bool = True,
        force_signal_debug: bool = False,
        force_full_exposure: bool = False,
        min_vol_scale: float | None = None,
        alpha_visibility: bool = True,
        fee_rate: float = 0.0004,  # 0.04% per trade
        slippage_rate: float = 0.0002,  # 0.02% per execution
        min_signal_threshold: float = 0.3,
        max_total_exposure: float = 1.0,
        min_holding_period: int = 2,
        cooldown_candles: int = 5,
        max_trades_per_100_steps: int = 20,
        position_change_threshold: float = 0.0,
        signal_filter: SignalFilter | None = None,
        signal_damping: float = 1.0,
        post_entry_ignore_bars: int = 0,
        vol_regime_filter: bool = False,
        vol_regime_signal: str = "volume_spike",
        vol_regime_high_threshold: float = 1.5,
        *,
        simple_signal_execution: bool = False,
        min_steps_between_trades: int | None = None,
        periods_per_year: float | None = None,
    ) -> None:
        self._event_bus = event_bus or EventBus()
        self._initial_equity = max(initial_equity, _EPS)
        self._target_vol = target_vol
        trace_limit("BacktestEngine.__init__ target_vol (stored, not applied in _replay)", target_vol)
        self._strategies = strategies or [MomentumPersistenceAlphaStrategy()]
        self._diagnostics = diagnostics
        self._fee_rate = max(0.0, fee_rate)
        self._slippage_rate = max(0.0, slippage_rate)
        self._min_signal_threshold = max(0.0, min_signal_threshold)
        self._max_total_exposure = max(0.0, min(1.0, max_total_exposure))
        trace_limit("BacktestEngine.__init__ max_total_exposure clamp", self._max_total_exposure)
        self._min_holding_period = max(0, min_holding_period)
        self._cooldown_candles = max(0, cooldown_candles)
        # Deprecated: execution always applies pending target when |delta| > eps (no major-scale floor).
        _ = position_change_threshold
        self._signal_filter = signal_filter
        # Kept for call-site compatibility; net_signal is not re-scaled in execution.
        _ = signal_damping
        self._post_entry_ignore_bars = max(0, post_entry_ignore_bars)
        self._vol_regime_filter_enabled = bool(vol_regime_filter)
        self._vol_regime_signal = vol_regime_signal
        self._vol_regime_high_threshold = float(vol_regime_high_threshold)
        self._simple_signal_execution = bool(simple_signal_execution)
        self._min_steps_between_trades = (
            _DEFAULT_MIN_STEPS_BETWEEN_TRADES
            if min_steps_between_trades is None
            else max(0, int(min_steps_between_trades))
        )
        self._periods_per_year = periods_per_year
        self._latest_market_state_for_execution: MarketStateVector | None = None
        self._strategy_weights = {
            strategy.__class__.__name__: 1.0 for strategy in self._strategies
        }
        self._last_trade_step = -10**9
        self._signal_loss = 0.0
        self._signal_count = 0
        self.exec_debug: dict[str, int] = {
            "signals_received": 0,
            "positions_opened": 0,
            "signals_blocked_position_exists": 0,
            "signals_blocked_cooldown": 0,
            "signals_blocked_other": 0,
        }
        # Wire up full trading pipeline (same as main.py)
        FeatureEngine(event_bus=self._event_bus)
        RegimeEngine(event_bus=self._event_bus)
        MarketStateEngine(event_bus=self._event_bus)
        # Capture market state before alpha runs so execution can scale by vol regime (risk control).
        self._event_bus.subscribe("market.state", self._capture_market_state_for_vol)
        self._alpha_engine = AlphaEngine(
            event_bus=self._event_bus,
            strategies=self._strategies,
            force_signal_debug=force_signal_debug,
            alpha_visibility=alpha_visibility,
            signal_filter=self._signal_filter,
            strategy_weights=self._strategy_weights,
            min_signal_keep_threshold=0.0 if self._simple_signal_execution else None,
        )

    def reset_exec_debug(self) -> None:
        """Reset execution funnel counters (call before each run / symbol / window)."""
        self.exec_debug = {
            "signals_received": 0,
            "positions_opened": 0,
            "signals_blocked_position_exists": 0,
            "signals_blocked_cooldown": 0,
            "signals_blocked_other": 0,
        }

    def _capture_market_state_for_vol(self, state: MarketStateVector) -> None:
        self._latest_market_state_for_execution = state

    def _is_high_vol_regime(self, state: MarketStateVector | None) -> bool:
        """HIGH_VOL when the chosen microstructure signal exceeds threshold (execution-only)."""
        if not self._vol_regime_filter_enabled or state is None:
            return False
        key = self._vol_regime_signal
        if key == "volume_spike":
            v = float(getattr(state, "volume_spike", 1.0))
        elif key in ("volatility_breakout", "vol_expansion"):
            v = float(getattr(state, "volatility_breakout", 1.0))
        else:
            v = float(getattr(state, "volume_spike", 1.0))
        return math.isfinite(v) and v > self._vol_regime_high_threshold

    def run(self, data: list[OHLCV] | None = None) -> BacktestResult:
        """Run backtest loop. Pass candles directly for deterministic replay."""
        if data is None:
            data = []
        candles = sorted(data, key=lambda c: (c.timestamp, c.symbol))
        return self._replay(candles)

    def run_from_storage(
        self,
        storage: StorageProtocol,
        symbol: str,
        timeframe: str,
        start_ts: int,
        end_ts: int,
    ) -> BacktestResult:
        """Load candles from storage and run backtest."""
        candles = storage.get_range(symbol, timeframe, start_ts, end_ts)
        return self._replay(candles)

    def _replay(self, candles: list[OHLCV]) -> BacktestResult:
        """Replay candles sequentially and simulate portfolio.

        Portfolio model: position = fraction * equity, pnl = position * price_change.
        Instant execution, no slippage.
        """
        self.reset_exec_debug()
        self._latest_market_state_for_execution = None
        equity = self._initial_equity
        gross_equity = self._initial_equity
        position_fraction = 0.0  # Executed position
        pending_position_fraction = 0.0  # Desired, executes next candle
        prev_position_fraction: float | None = None
        prev_close: float | None = None
        equity_curve: list[tuple[int, float]] = []
        trade_count = 0
        position_sum = 0.0
        bars_with_position = 0
        total_bars = len(candles)
        total_turnover = 0.0
        total_cost_paid = 0.0
        gross_return_sum = 0.0
        net_return_sum = 0.0
        bars_since_last_trade = self._min_holding_period
        bars_since_entry = self._post_entry_ignore_bars + 1
        last_gross_equity = gross_equity
        last_net_equity = equity
        latest_net_signal = 0.0
        last_net_for_delta: float | None = None
        current_step_ref: list[int] = [0]
        latest_raw_signal = 0.0
        latest_after_scale_signal = 0.0
        latest_after_smooth_signal = 0.0
        latest_target_position = 0.0
        signal_values: list[float] = []

        scale_loss_sum = 0.0
        smooth_loss_sum = 0.0
        mapping_loss_sum = 0.0
        execution_loss_sum = 0.0
        loss_count = 0
        signal_power_raw_sum = 0.0
        signal_power_final_sum = 0.0

        latest_alpha_signals: dict[str, float] = {}
        strategy_returns: dict[str, list[float]] = {}
        strategy_counts: dict[str, int] = {}
        for name in [
            "TrendAlphaStrategy",
            "MeanReversionAlphaStrategy",
            "BreakoutAlphaStrategy",
        ]:
            strategy_returns[name] = []
            strategy_counts[name] = 0
        returns: list[float] = []
        signals_after_bar: list[dict[str, float]] = []
        net_signals_per_bar: list[float] = []
        closes: list[float] = []
        trend_pnl = 0.0
        mr_pnl = 0.0
        breakout_pnl = 0.0
        equity_after_last_exec = self._initial_equity
        trade_returns: list[float] = []
        signal_rank_window: deque[float] = deque(maxlen=_SIGNAL_RANK_WINDOW)

        self._signal_loss = 0.0
        self._signal_count = 0
        self._alpha_engine.reset_net_scale_metrics()

        def _on_alpha(signal: AlphaSignal) -> None:
            if self._diagnostics:
                _logger.info(
                    "alpha.strategy name=%s signal=%.4f direction=%s",
                    signal.strategy_name,
                    signal.strength,
                    signal.direction,
                )

        def _on_net_components(payload: dict[str, float]) -> None:
            nonlocal latest_raw_signal, latest_after_scale_signal
            latest_raw_signal = float(payload.get("raw_signal", 0.0))
            latest_after_scale_signal = float(payload.get("after_scale", 0.0))

        def _on_alpha_signals(signals: dict[str, float]) -> None:
            nonlocal latest_alpha_signals
            latest_alpha_signals = dict(signals)

        def _on_net_signal(net_signal: float) -> None:
            nonlocal pending_position_fraction, latest_net_signal, last_net_for_delta, bars_since_entry
            nonlocal latest_after_smooth_signal, latest_target_position
            nonlocal scale_loss_sum, smooth_loss_sum, mapping_loss_sum, loss_count, signal_power_raw_sum
            nonlocal position_fraction
            original_signal = net_signal
            # Clip only. Net signal is already filtered/smoothed in alpha; do not damp again here.
            trace_limit("_on_net_signal clip net_signal to [-1,1]", "max(-1.0,min(1.0,net_signal))")
            latest_net_signal = max(-1.0, min(1.0, net_signal))
            latest_after_smooth_signal = latest_net_signal
            _logger.info(
                "[TRACE] after_clip signal=%.4f",
                latest_net_signal,
            )
            execution_net_signal = latest_net_signal
            _logger.info(
                "[TRACE] execution_signal=%.4f",
                execution_net_signal,
            )
            if abs(execution_net_signal) >= _EPS:
                self.exec_debug["signals_received"] += 1
            if abs(position_fraction) > _EPS and bars_since_entry < MIN_HOLD_STEPS:
                if abs(execution_net_signal) >= _EPS:
                    self.exec_debug["signals_blocked_position_exists"] += 1
                pending_position_fraction = position_fraction
                latest_target_position = position_fraction
                last_net_for_delta = execution_net_signal
                if self._diagnostics:
                    _logger.info(
                        "portfolio.hold reason=signal_horizon_hold "
                        "bars_since_entry=%s need>=%s",
                        bars_since_entry,
                        MIN_HOLD_STEPS,
                    )
                return
            has_strong_entry_signal = abs(execution_net_signal) > _ENTRY_SIGNAL_THRESHOLD
            trace_limit("_ENTRY_SIGNAL_THRESHOLD", _ENTRY_SIGNAL_THRESHOLD)
            if (
                not self._simple_signal_execution
                and abs(position_fraction) > _EPS
                and bars_since_entry <= self._post_entry_ignore_bars
                and not has_strong_entry_signal
            ):
                if abs(execution_net_signal) >= _EPS:
                    self.exec_debug["signals_blocked_other"] += 1
                if self._diagnostics:
                    _logger.info(
                        "portfolio.hold reason=post_entry_ignore step=%s bars_since_entry=%s threshold=%s signal=%.4f",
                        current_step_ref[0],
                        bars_since_entry,
                        self._post_entry_ignore_bars,
                        execution_net_signal,
                    )
                return

            allocation: list[tuple[str, str, float]]
            if execution_net_signal > 0.0:
                allocation = [("portfolio", "long", 1.0)]
            elif execution_net_signal < 0.0:
                allocation = [("portfolio", "short", 1.0)]
            else:
                allocation = []
            self._event_bus.publish("portfolio.allocation", allocation)
            rr20_for_target: float | None = None
            if self._latest_market_state_for_execution is not None:
                _rr = float(
                    getattr(
                        self._latest_market_state_for_execution,
                        "rolling_return_20",
                        0.0,
                    )
                )
                rr20_for_target = _rr if math.isfinite(_rr) else 0.0
            scale_loss = abs(latest_raw_signal - latest_after_scale_signal)
            smooth_loss = abs(latest_after_scale_signal - latest_after_smooth_signal)
            scale_loss_sum += scale_loss
            smooth_loss_sum += smooth_loss
            signal_power_raw_sum += latest_raw_signal * latest_raw_signal
            if self._simple_signal_execution:
                signal_for_sizing = execution_net_signal
                filtered_signal = execution_net_signal
                decision = "simple_signal_path"
                _logger.info(
                    "[TRACE] simple_signal_execution net=%.4f -> compute_target_position",
                    execution_net_signal,
                )
                target = compute_target_position(
                    filtered_signal,
                    current_position=position_fraction,
                    log_validation=self._diagnostics,
                    rolling_return_20=rr20_for_target,
                )
                if (
                    abs(execution_net_signal) >= _EPS
                    and abs(target) < _EPS
                ):
                    self.exec_debug["signals_blocked_other"] += 1
            else:
                signal_for_sizing = (
                    execution_net_signal
                    if abs(execution_net_signal) >= self._min_signal_threshold
                    else 0.0
                )
                trace_limit("BacktestEngine _min_signal_threshold", self._min_signal_threshold)
                _logger.info(
                    "[TRACE] after_min_signal_gate signal=%.4f (execution was %.4f)",
                    signal_for_sizing,
                    execution_net_signal,
                )
                decision = "proceed"
                last_sig = last_net_for_delta
                delta_abs = (
                    abs(execution_net_signal - last_sig)
                    if last_sig is not None
                    else float("inf")
                )
                delta_threshold = _SIGNAL_DELTA_BASE_THRESHOLD
                same_direction = False
                if last_sig is not None:
                    if abs(execution_net_signal) < _EPS and abs(last_sig) < _EPS:
                        same_direction = True
                    elif abs(execution_net_signal) >= _EPS and abs(last_sig) >= _EPS:
                        same_direction = (execution_net_signal > 0 and last_sig > 0) or (
                            execution_net_signal < 0 and last_sig < 0
                        )
                passes_delta_filter = (
                    last_sig is None
                    or abs(execution_net_signal) > _STRONG_SIGNAL_DELTA_BYPASS
                    or (same_direction and delta_abs > _SAME_DIRECTION_MIN_DELTA)
                    or delta_abs > delta_threshold
                )
                filtered_signal = signal_for_sizing if passes_delta_filter else position_fraction
                if not passes_delta_filter:
                    _logger.info(
                        "[TRACE] after_delta_filter signal=%.4f (held previous position_fraction)",
                        filtered_signal,
                    )
                    decision = "signal_delta_filter"
                else:
                    _logger.info(
                        "[TRACE] after_delta_filter signal=%.4f (applied mapped_target)",
                        filtered_signal,
                    )
                    decision = "enter_or_update"
                # Rank gate: top decile of |net| vs last 50 bars (symmetric for long/short).
                abs_net = abs(execution_net_signal)
                signal_rank_window.append(abs_net)
                w = list(signal_rank_window)
                if len(w) < _SIGNAL_RANK_WINDOW:
                    percentile_rank = 0.0
                else:
                    percentile_rank = _empirical_percentile_rank_le(w, abs_net)
                if percentile_rank < _SIGNAL_RANK_MIN_PERCENTILE:
                    target = 0.0
                    if abs(execution_net_signal) >= _EPS:
                        self.exec_debug["signals_blocked_other"] += 1
                    if self._diagnostics:
                        _logger.info(
                            "[TRACE] signal_rank_gate percentile=%.4f need>=%.4f window=%s",
                            percentile_rank,
                            _SIGNAL_RANK_MIN_PERCENTILE,
                            len(w),
                        )
                elif self._is_high_vol_regime(self._latest_market_state_for_execution):
                    target = 0.0
                    if abs(execution_net_signal) >= _EPS:
                        self.exec_debug["signals_blocked_other"] += 1
                    if self._diagnostics:
                        _logger.info(
                            "[TRACE] vol_regime_filter HIGH_VOL "
                            "signal=%s thr=%.4f target_gated=0.0",
                            self._vol_regime_signal,
                            self._vol_regime_high_threshold,
                        )
                else:
                    target = compute_target_position(
                        filtered_signal,
                        current_position=position_fraction,
                        log_validation=self._diagnostics,
                        rolling_return_20=rr20_for_target,
                    )
                    if (
                        abs(execution_net_signal) >= _EPS
                        and abs(target) < _EPS
                    ):
                        self.exec_debug["signals_blocked_other"] += 1
            print(
                "[PIPELINE FIX] "
                f"raw_signal={signal_for_sizing} "
                f"filtered_signal={filtered_signal} "
                f"target={target}"
            )
            latest_target_position = target
            _logger.info(
                "[TRACE] after_compute_target mapped=%.4f",
                target,
            )
            mapping_loss = abs(latest_after_smooth_signal - latest_target_position)
            mapping_loss_sum += mapping_loss
            loss_count += 1
            _logger.info(
                "[TRACE] LOSS_STEP step=%s raw=%.6f after_scale=%.6f after_smooth=%.6f target=%.6f scale_loss=%.6f smooth_loss=%.6f mapping_loss=%.6f",
                current_step_ref[0],
                latest_raw_signal,
                latest_after_scale_signal,
                latest_after_smooth_signal,
                target,
                scale_loss,
                smooth_loss,
                mapping_loss,
            )
            pending_position_fraction = target
            last_net_for_delta = execution_net_signal
            if self._diagnostics:
                _logger.info(
                    "execution.check signal=%.4f target=%.4f decision=%s",
                    execution_net_signal,
                    target,
                    decision,
                )
                _logger.info(
                    "[TRACE] portfolio.target final=%.4f net_signal=%.4f execution_net_signal=%.4f original_bus=%.4f",
                    pending_position_fraction,
                    latest_net_signal,
                    execution_net_signal,
                    original_signal,
                )
                _logger.info(
                    "portfolio.target target=%.4f net_signal=%.4f execution_net_signal=%.4f",
                    pending_position_fraction,
                    latest_net_signal,
                    execution_net_signal,
                )

        self._event_bus.subscribe("alpha.signal", _on_alpha)
        self._event_bus.subscribe("alpha.net_signal_components", _on_net_components)
        self._event_bus.subscribe("alpha.net_signal", _on_net_signal)
        self._event_bus.subscribe("alpha.signals", _on_alpha_signals)

        for i, candle in enumerate(candles):
            current_step_ref[0] = i
            close = max(candle.close, _MIN_PRICE)
            bar_return = 0.0
            if prev_close is not None and prev_close >= _MIN_PRICE:
                price_return = (close - prev_close) / prev_close
                price_return = max(-1.0, min(1e6, price_return))
                bar_return = price_return
                gross_pnl = gross_equity * position_fraction * price_return
                net_pnl = equity * position_fraction * price_return
                if i >= 1 and len(signals_after_bar) >= i:
                    dt, dm, db = _split_gross_pnl_by_signals(
                        signals_after_bar[i - 1], gross_pnl
                    )
                    trend_pnl += dt
                    mr_pnl += dm
                    breakout_pnl += db
                gross_equity = max(gross_equity + gross_pnl, _EPS)
                equity = max(equity + net_pnl, _EPS)
                gross_return_sum += (gross_equity - last_gross_equity) / max(last_gross_equity, _EPS)
                net_return_sum += (equity - last_net_equity) / max(last_net_equity, _EPS)
                last_gross_equity = gross_equity
                last_net_equity = equity

            # Execution delay: target from T-1 executes at end of current bar T.
            if i > 0:
                pending = pending_position_fraction
                target_position = pending
                _logger.info(
                    "[TRACE] HOLD STATE bars_since_trade=%s",
                    bars_since_last_trade,
                )
                # Pipeline target for this bar (from prior candle's net signal).
                want_move = abs(pending - position_fraction) > _EPS
                pos = position_fraction
                execution_branch = "UNKNOWN"
                if not want_move:
                    execution_branch = "NO_MOVE"
                    target_position = pending_position_fraction
                elif bars_since_last_trade < self._min_holding_period:
                    # Wrong side → follow signal; strong move → follow; else gradual (not hard freeze).
                    if pos * pending < 0.0:
                        execution_branch = "MIN_HOLD_FLIP"
                        target_position = pending_position_fraction
                    elif abs(pending - pos) > _EXEC_STRONG_MOVE_BYPASS:
                        execution_branch = "MIN_HOLD_STRONG_BYPASS"
                        target_position = pending_position_fraction
                    elif abs(pending - pos) < _EXEC_SAME_DIR_MIN_DELTA:
                        execution_branch = "MIN_HOLD_SMALL_DELTA"
                        target_position = pending_position_fraction
                    else:
                        execution_branch = "MIN_HOLD_DIRECT"
                        target_position = pending_position_fraction
                elif bars_since_last_trade < self._cooldown_candles:
                    # Same-direction freeze here (not blend): blend every bar caused churn vs weak alpha.
                    if pos * pending < 0.0:
                        execution_branch = "COOLDOWN_FLIP"
                        target_position = pending_position_fraction
                    elif abs(pending - pos) < _EXEC_SAME_DIR_MIN_DELTA:
                        execution_branch = "COOLDOWN_SMALL_DELTA"
                        target_position = pending_position_fraction
                    else:
                        execution_branch = "COOLDOWN_FREEZE"
                        target_position = pending_position_fraction
                elif abs(pending - pos) > _EXEC_STRONG_MOVE_BYPASS:
                    execution_branch = "STRONG_MOVE_BYPASS"
                    target_position = pending_position_fraction
                else:
                    execution_branch = "DIRECT_EXECUTION"
                    target_position = pending_position_fraction
                if (
                    want_move
                    and execution_branch == "COOLDOWN_FREEZE"
                    and abs(pending - pos) > _EPS
                ):
                    self.exec_debug["signals_blocked_cooldown"] += 1
                target_position = max(-1.0, min(1.0, target_position))
                executed_position = target_position
                mismatch = abs(executed_position - pending) > 1e-9
                if mismatch:
                    # Clamp + execution can make the final executed position differ
                    # from the desired pending target for this bar.
                    print(
                        "[BACKTEST SMOOTH TRACE] "
                        f"current={pos} "
                        f"target={pending} "
                        f"final={executed_position}"
                    )
                    print(
                        "[EXECUTION MISMATCH] "
                        f"step={i} "
                        f"current={pos} "
                        f"pending={pending} "
                        f"executed={executed_position} "
                        f"delta={pending - pos} "
                        f"branch={execution_branch} "
                        f"bars_since_last_trade={bars_since_last_trade} "
                        f"min_hold={self._min_holding_period} "
                        f"cooldown={self._cooldown_candles} "
                        f"strong_bypass={_EXEC_STRONG_MOVE_BYPASS} "
                        f"min_delta={_EXEC_SAME_DIR_MIN_DELTA}"
                    )
                position_adjustment_factor = (
                    (executed_position - pos) / (pending - pos)
                    if abs(pending - pos) > _EPS
                    else 0.0
                )
                print(
                    f"[EXECUTION CORE] current={pos} target={pending} executed={executed_position}"
                )
                print(
                    "[EXECUTION CORE] "
                    f"max_delta={_EXEC_STRONG_MOVE_BYPASS} "
                    f"step_size={_EXEC_SAME_DIR_MIN_DELTA} "
                    f"turnover_limit=None "
                    f"position_adjustment_factor={position_adjustment_factor} "
                    f"min_holding_period={self._min_holding_period} "
                    f"cooldown_candles={self._cooldown_candles}"
                )

                if self._diagnostics:
                    _logger.info(
                        "[EXEC FIX] pending=%.4f executed=%.4f delta=%.4f",
                        pending_position_fraction,
                        target_position,
                        abs(pending_position_fraction - target_position),
                    )

                self._signal_loss += abs(pending - target_position)
                self._signal_count += 1
                _logger.info(
                    "[TRACE] SIGNAL FLOW step=%s pending=%.4f executed=%.4f",
                    i,
                    pending,
                    target_position,
                )

                delta = target_position - position_fraction
                execution_loss = abs(latest_target_position - target_position)
                execution_loss_sum += execution_loss
                _logger.info(
                    "[TRACE] EXEC_STEP step=%s target_pre_exec=%.6f executed_target=%.6f execution_loss=%.6f",
                    i,
                    latest_target_position,
                    target_position,
                    execution_loss,
                )
                # No major-scale / position-change floor: follow full target whenever exposure differs.
                should_execute = abs(delta) > _EPS
                min_hold_ok = (i - self._last_trade_step) >= self._min_steps_between_trades
                if should_execute and min_hold_ok:
                    self.exec_debug["positions_opened"] += 1
                    # Position-change cost with explicit fee + slippage impact.
                    turnover = abs(delta)
                    cost_rate = self._fee_rate + self._slippage_rate
                    trade_cost = equity * turnover * cost_rate
                    tr_seg = (equity - equity_after_last_exec) / max(
                        equity_after_last_exec, _EPS
                    )
                    trade_returns.append(tr_seg)
                    equity = max(equity - trade_cost, _EPS)
                    equity_after_last_exec = equity
                    total_turnover += turnover
                    total_cost_paid += trade_cost
                    previous_position = position_fraction
                    position_fraction = target_position
                    self._last_trade_step = i
                    bars_since_last_trade = 0
                    if abs(previous_position) < _EPS and abs(target_position) > _EPS:
                        bars_since_entry = 0
                    elif abs(target_position) < _EPS:
                        bars_since_entry = self._post_entry_ignore_bars + 1
                    if self._diagnostics:
                        if abs(previous_position) < _EPS and abs(target_position) > _EPS:
                            action = "entry"
                        elif abs(previous_position) > _EPS and abs(target_position) < _EPS:
                            action = "exit"
                        else:
                            action = "flip"
                        _logger.info(
                            "portfolio.execution executed_trade=%s current=%.1f target=%.1f net_signal=%.4f turnover=%.6f cost=%.6f fee_rate=%.6f slippage_rate=%.6f",
                            action,
                            previous_position,
                            target_position,
                            latest_net_signal,
                            turnover,
                            trade_cost,
                            self._fee_rate,
                            self._slippage_rate,
                        )
                elif should_execute and not min_hold_ok:
                    self.exec_debug["signals_blocked_cooldown"] += 1
                    if self._diagnostics:
                        _logger.info(
                            "portfolio.hold reason=min_hold_steps step=%s last_trade_step=%s min_hold=%s",
                            i,
                            self._last_trade_step,
                            self._min_steps_between_trades,
                        )
                    bars_since_last_trade += 1
                else:
                    if abs(delta) > _EPS and self._diagnostics:
                        _logger.info(
                            "portfolio.hold reason=no_execute step=%s delta=%.4f",
                            i,
                            abs(delta),
                        )
                    bars_since_last_trade += 1
                final_position = position_fraction
                print(
                    "[FINAL POSITION CHECK] "
                    f"target={target_position} "
                    f"final={final_position}"
                )
                signal_power_final_sum += position_fraction * position_fraction
            else:
                bars_since_last_trade += 1
            if abs(position_fraction) > _EPS:
                bars_since_entry += 1

            equity_curve.append((candle.timestamp, equity))
            self._event_bus.publish("market.candle", candle)
            signals_after_bar.append(dict(latest_alpha_signals))
            net_signals_per_bar.append(latest_net_signal)
            signal_values.append(latest_net_signal)
            closes.append(close)
            returns.append(bar_return)
            signals = latest_alpha_signals
            for strategy_name, signal in signals.items():
                if strategy_name not in strategy_returns:
                    continue
                if i < len(returns):
                    r = returns[i]
                    strategy_returns[strategy_name].append(float(signal) * r)
                    strategy_counts[strategy_name] += 1
            prev_close = close

            # Trade tracking: count when position changes
            if prev_position_fraction is not None:
                if abs(position_fraction - prev_position_fraction) > _EPS:
                    trade_count += 1
            prev_position_fraction = position_fraction

            # Position size and time in market
            position_sum += abs(position_fraction)
            if abs(position_fraction) > _EPS:
                bars_with_position += 1

        # Unsubscribe to avoid leaks if engine is reused
        # (EventBus has no unsubscribe; handlers stay but engine typically single-use)

        if self._periods_per_year is not None:
            metrics = compute_metrics(
                equity_curve, periods_per_year=float(self._periods_per_year)
            )
        else:
            metrics = compute_metrics(equity_curve)
        metrics["vol_regime_filter"] = 1.0 if self._vol_regime_filter_enabled else 0.0
        avg_position_size = position_sum / total_bars if total_bars > 0 else 0.0
        time_in_market = (bars_with_position / total_bars * 100.0) if total_bars > 0 else 0.0

        metrics["trade_count"] = float(trade_count)
        metrics["avg_position_size"] = avg_position_size
        metrics["time_in_market"] = time_in_market
        metrics["turnover"] = total_turnover
        metrics["total_cost"] = total_cost_paid
        metrics["gross_total_return"] = (gross_equity - self._initial_equity) / self._initial_equity
        metrics["net_total_return"] = (equity - self._initial_equity) / self._initial_equity
        metrics["cost_impact"] = metrics["gross_total_return"] - metrics["net_total_return"]
        metrics["gross_return_sum"] = gross_return_sum
        metrics["net_return_sum"] = net_return_sum

        if self._diagnostics:
            _logger.info(
                "Backtest sanity: total_trades=%s avg_position_size=%.4f time_in_market=%.1f%% turnover=%.4f cost_impact=%.6f gross_ret=%.6f net_ret=%.6f",
                trade_count,
                avg_position_size,
                time_in_market,
                total_turnover,
                metrics["cost_impact"],
                metrics["gross_total_return"],
                metrics["net_total_return"],
            )

        avg_scale_loss = scale_loss_sum / max(1, loss_count)
        avg_smooth_loss = smooth_loss_sum / max(1, loss_count)
        avg_mapping_loss = mapping_loss_sum / max(1, loss_count)
        avg_execution_loss_breakdown = execution_loss_sum / max(1, loss_count)
        dominant_loss = max(
            ("scale", avg_scale_loss),
            ("smooth", avg_smooth_loss),
            ("mapping", avg_mapping_loss),
            ("execution", avg_execution_loss_breakdown),
            key=lambda x: x[1],
        )
        signal_power_raw = signal_power_raw_sum / max(1, loss_count)
        signal_power_final = signal_power_final_sum / max(1, loss_count)
        attenuation_ratio = (
            signal_power_final / signal_power_raw if signal_power_raw > _EPS else 0.0
        )
        _logger.info("[TRACE] LOSS_BREAKDOWN")
        _logger.info("scale=%.6f", avg_scale_loss)
        _logger.info("smooth=%.6f", avg_smooth_loss)
        _logger.info("mapping=%.6f", avg_mapping_loss)
        _logger.info("execution=%.6f", avg_execution_loss_breakdown)
        _logger.info(
            "[TRACE] DOMINANT_LOSS_SOURCE source=%s value=%.6f",
            dominant_loss[0],
            dominant_loss[1],
        )
        _logger.info("[TRACE] SIGNAL_ATTENUATION ratio=%.6f", attenuation_ratio)

        avg_net_scale_loss, net_scale_steps = self._alpha_engine.get_avg_net_scale_loss()
        avg_execution_loss = self._signal_loss / max(1, self._signal_count)
        _logger.info(
            "[TRACE] NET_SCALE avg_net_scale_loss=%.6f (aggregation_steps=%s)",
            avg_net_scale_loss,
            net_scale_steps,
        )
        _logger.info(
            "[TRACE] EXECUTION LOSS avg_delta=%.6f (execution_steps=%s)",
            avg_execution_loss,
            self._signal_count,
        )
        _logger.info(
            "[TRACE] COMPARE avg_net_scale_loss=%.6f avg_execution_loss=%.6f",
            avg_net_scale_loss,
            avg_execution_loss,
        )
        if avg_execution_loss > avg_net_scale_loss:
            _logger.info(
                "[TRACE] DOMINANT_LOSS execution_layer (avg_execution_loss > avg_net_scale_loss)",
            )
        elif avg_net_scale_loss > avg_execution_loss:
            _logger.info(
                "[TRACE] DOMINANT_LOSS alpha NET_SIGNAL_SCALE (avg_net_scale_loss > avg_execution_loss)",
            )
        else:
            _logger.info(
                "[TRACE] DOMINANT_LOSS tie / double damping (losses equal within float tie)",
            )

        total_return = float(metrics.get("total_return", 0.0))
        num_trades = int(trade_count)
        win_rate = (
            sum(1 for t in trade_returns if t > 0) / len(trade_returns)
            if trade_returns
            else 0.0
        )
        avg_trade_return = (
            sum(trade_returns) / len(trade_returns) if trade_returns else 0.0
        )
        max_drawdown = float(metrics.get("max_drawdown", 0.0))
        metrics["win_rate"] = win_rate
        metrics["avg_trade_return"] = avg_trade_return

        _sym = candles[0].symbol if candles else "unknown"
        print(f"[EXECUTION DEBUG] {_sym}")
        for _k, _v in self.exec_debug.items():
            print(f"  {_k}: {_v}")

        if not self._diagnostics:
            return BacktestResult(
                equity_curve=equity_curve,
                metrics=metrics,
                trade_returns=list(trade_returns),
            )

        def _avg(x: list[float]) -> float:
            return sum(x) / len(x) if x else 0.0

        print("[PER STRATEGY EDGE]")
        for k in strategy_returns:
            print(
                f"{k} "
                f"avg_signal_return={_avg(strategy_returns[k])} "
                f"count={strategy_counts[k]}"
            )

        print(
            "[PERF SUMMARY] "
            f"total_return={total_return} "
            f"num_trades={num_trades} "
            f"win_rate={win_rate} "
            f"avg_trade_return={avg_trade_return} "
            f"max_drawdown={max_drawdown}"
        )
        def _stats(x: list[float]) -> tuple[float, float, float, int]:
            if not x:
                return (0.0, 0.0, 0.0, 0)
            return (min(x), max(x), sum(x) / len(x), len(x))

        mn, mx, avg, cnt = _stats(signal_values)
        print(
            "[SIGNAL DIST] "
            f"min={mn} max={mx} avg={avg} count={cnt}"
        )

        correct_predictions = 0
        total_predictions = 0
        sig_corr: list[float] = []
        ret_corr: list[float] = []
        n_close = len(closes)
        strategy_sig_corr: dict[str, list[float]] = {k: [] for k in _ALPHA_DIAG_KEYS}
        strategy_ret_corr: dict[str, list[float]] = {k: [] for k in _ALPHA_DIAG_KEYS}
        strategy_hits: dict[str, int] = {k: 0 for k in _ALPHA_DIAG_KEYS}
        strategy_counts: dict[str, int] = {k: 0 for k in _ALPHA_DIAG_KEYS}
        strategy_long_sum: dict[str, float] = {k: 0.0 for k in _ALPHA_DIAG_KEYS}
        strategy_long_count: dict[str, int] = {k: 0 for k in _ALPHA_DIAG_KEYS}
        strategy_short_sum: dict[str, float] = {k: 0.0 for k in _ALPHA_DIAG_KEYS}
        strategy_short_count: dict[str, int] = {k: 0 for k in _ALPHA_DIAG_KEYS}
        trend_total_predictions = 0
        trend_correct_predictions = 0
        trend_outcome_printed = 0
        trend_outcome_limit = 20
        if n_close >= 2:
            for j in range(n_close - 1):
                sig_j = net_signals_per_bar[j]
                next_ret = (closes[j + 1] - closes[j]) / max(closes[j], _MIN_PRICE)
                if _sign_f(sig_j) == _sign_f(next_ret):
                    correct_predictions += 1
                total_predictions += 1
                sig_corr.append(sig_j)
                ret_corr.append(next_ret)

                per_bar_signals = (
                    signals_after_bar[j] if j < len(signals_after_bar) else {}
                )
                for strategy_name in _ALPHA_DIAG_KEYS:
                    strategy_signal = float(per_bar_signals.get(strategy_name, 0.0))
                    print(
                        "[ALPHA DIAG] "
                        f"strategy={strategy_name} "
                        f"signal={strategy_signal} "
                        f"next_return={next_ret}"
                    )
                    strategy_sig_corr[strategy_name].append(strategy_signal)
                    strategy_ret_corr[strategy_name].append(next_ret)
                    strategy_counts[strategy_name] += 1
                    if _sign_f(strategy_signal) == _sign_f(next_ret):
                        strategy_hits[strategy_name] += 1
                    if strategy_signal > 0.0:
                        strategy_long_sum[strategy_name] += next_ret
                        strategy_long_count[strategy_name] += 1
                    elif strategy_signal < 0.0:
                        strategy_short_sum[strategy_name] += next_ret
                        strategy_short_count[strategy_name] += 1

                # Trend-only extraction: correlate TrendAlphaStrategy direction vs next return.
                trend_signal = float(per_bar_signals.get("TrendAlphaStrategy", 0.0))
                # Count trend directional matches (same_direction means "trend contributes" by direction).
                trend_total_predictions += 1
                same_direction_trend = _sign_f(trend_signal) == _sign_f(next_ret)
                if same_direction_trend:
                    trend_correct_predictions += 1

                # For human-readable proof, skip "flat/undefined" windows where either side is near-zero.
                if (
                    trend_outcome_printed < trend_outcome_limit
                    and _sign_f(trend_signal) != 0.0
                    and _sign_f(next_ret) != 0.0
                ):
                    print(
                        "[TREND OUTCOME] "
                        f"signal={trend_signal} "
                        f"next_return={next_ret}"
                    )
                    print(
                        "[TREND SIGN CHECK] "
                        f"same_direction={same_direction_trend}"
                    )
                    trend_outcome_printed += 1
        hit_rate = (
            correct_predictions / total_predictions if total_predictions else 0.0
        )
        correlation = _pearson_correlation(sig_corr, ret_corr)
        print(
            "[ALPHA QUALITY] "
            f"hit_rate={hit_rate}"
        )
        print(
            "[ALPHA CORRELATION] "
            f"corr(signal, future_return)={correlation}"
        )
        long_returns: list[float] = []
        short_returns: list[float] = []
        for k in range(len(sig_corr)):
            s = sig_corr[k]
            r = ret_corr[k]
            if s > 0:
                long_returns.append(r)
            elif s < 0:
                short_returns.append(r)

        print(
            "[ALPHA EDGE] "
            f"avg_long_return={_avg(long_returns)} "
            f"avg_short_return={_avg(short_returns)}"
        )

        # Emit the requested trend anti-correlation count.
        trend_match_rate = (
            (trend_correct_predictions / trend_total_predictions)
            if trend_total_predictions
            else 0.0
        )
        print(
            "[TREND MATCH RATE] "
            f"trend_direction_match={trend_match_rate}"
        )
        print(
            "[STRATEGY CONTRIBUTION] "
            f"trend={trend_pnl} "
            f"mean_reversion={mr_pnl} "
            f"breakout={breakout_pnl}"
        )
        for strategy_name in _ALPHA_DIAG_KEYS:
            corr = _pearson_correlation(
                strategy_sig_corr[strategy_name], strategy_ret_corr[strategy_name]
            )
            hit_rate_strategy = (
                strategy_hits[strategy_name] / strategy_counts[strategy_name]
                if strategy_counts[strategy_name] > 0
                else 0.0
            )
            avg_long_return = (
                strategy_long_sum[strategy_name] / strategy_long_count[strategy_name]
                if strategy_long_count[strategy_name] > 0
                else 0.0
            )
            avg_short_return = (
                strategy_short_sum[strategy_name] / strategy_short_count[strategy_name]
                if strategy_short_count[strategy_name] > 0
                else 0.0
            )
            print("[ALPHA STRATEGY METRICS]")
            print(f"strategy={strategy_name}")
            print(f"corr={corr}")
            print(f"hit_rate={hit_rate_strategy}")
            print(f"avg_long_return={avg_long_return}")
            print(f"avg_short_return={avg_short_return}")

        return BacktestResult(
            equity_curve=equity_curve,
            metrics=metrics,
            trade_returns=list(trade_returns),
        )
