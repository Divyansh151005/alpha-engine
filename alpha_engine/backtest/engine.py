"""Backtest orchestration engine."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from alpha_engine.allocation_engine import CapitalAllocator
from alpha_engine.allocation_engine.allocation_models import PositionAllocation
from alpha_engine.alpha import (
    AlphaEngine,
    AlphaSignal,
    BreakoutAlphaStrategy,
    MeanReversionAlphaStrategy,
    TrendAlphaStrategy,
)
from alpha_engine.alpha.base_alpha import BaseAlphaStrategy
from alpha_engine.core.event_bus import EventBus
from alpha_engine.core.logger import get_logger
from alpha_engine.data.models import OHLCV
from alpha_engine.features import FeatureEngine
from alpha_engine.risk_engine import RiskEngine
from alpha_engine.state_engine import MarketStateEngine
from alpha_engine.volatility_engine import VolatilityTargetingEngine

# Numerical stability
_EPS = 1e-12
_MIN_PRICE = 1e-12

_logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Result of a backtest run."""

    equity_curve: list[tuple[int, float]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


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


def _direction_sign(direction: str) -> float:
    """Map direction to signed multiplier for position exposure."""
    if direction == "long":
        return 1.0
    if direction == "short":
        return -1.0
    return 0.0


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
    ) -> None:
        self._event_bus = event_bus or EventBus()
        self._initial_equity = max(initial_equity, _EPS)
        self._target_vol = target_vol
        self._strategies = strategies or [
            TrendAlphaStrategy(),
            MeanReversionAlphaStrategy(),
            BreakoutAlphaStrategy(),
        ]
        self._diagnostics = diagnostics

        # Wire up full trading pipeline (same as main.py)
        FeatureEngine(event_bus=self._event_bus)
        MarketStateEngine(event_bus=self._event_bus)
        AlphaEngine(
            event_bus=self._event_bus,
            strategies=self._strategies,
            force_signal_debug=force_signal_debug,
            alpha_visibility=alpha_visibility,
        )
        CapitalAllocator(event_bus=self._event_bus)
        VolatilityTargetingEngine(
            event_bus=self._event_bus,
            target_vol=target_vol,
            min_scale=min_vol_scale,
        )
        RiskEngine(
            event_bus=self._event_bus,
            initial_equity=self._initial_equity,
            force_full_exposure=force_full_exposure,
        )

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
        equity = self._initial_equity
        position_fraction = 0.0
        prev_position_fraction: float | None = None
        prev_close: float | None = None
        equity_curve: list[tuple[int, float]] = []
        trade_count = 0
        position_sum = 0.0
        bars_with_position = 0
        total_bars = len(candles)

        def _on_alpha(signal: AlphaSignal) -> None:
            if self._diagnostics:
                _logger.info(
                    "alpha.strategy name=%s signal=%.4f direction=%s",
                    signal.strategy_name,
                    signal.strength,
                    signal.direction,
                )

        def _on_allocation(allocations: list[PositionAllocation]) -> None:
            if self._diagnostics and allocations:
                fractions = [(a.strategy_name, a.direction, a.capital_fraction) for a in allocations]
                _logger.info(
                    "portfolio.allocation timestamp=%s fractions=%s",
                    allocations[0].timestamp if allocations else 0,
                    fractions,
                )

        def _on_position(positions: list[PositionAllocation]) -> None:
            if self._diagnostics and positions:
                fractions = [(p.strategy_name, p.capital_fraction) for p in positions]
                _logger.info(
                    "portfolio.position timestamp=%s volatility_adjusted=%s",
                    positions[0].timestamp if positions else 0,
                    fractions,
                )

        def _on_final(allocations: list[PositionAllocation]) -> None:
            nonlocal position_fraction
            if self._diagnostics and allocations:
                fractions = [(a.strategy_name, a.direction, a.capital_fraction) for a in allocations]
                _logger.info(
                    "portfolio.final timestamp=%s risk_adjusted=%s",
                    allocations[0].timestamp if allocations else 0,
                    fractions,
                )
            if not allocations:
                return
            net = sum(
                _direction_sign(a.direction) * a.capital_fraction
                for a in allocations
            )
            position_fraction = max(-1.0, min(1.0, net))

        self._event_bus.subscribe("alpha.signal", _on_alpha)
        self._event_bus.subscribe("portfolio.allocation", _on_allocation)
        self._event_bus.subscribe("portfolio.position", _on_position)
        self._event_bus.subscribe("portfolio.final", _on_final)

        for candle in candles:
            close = max(candle.close, _MIN_PRICE)
            if prev_close is not None and prev_close >= _MIN_PRICE:
                price_return = (close - prev_close) / prev_close
                price_return = max(-1.0, min(1e6, price_return))
                pnl = equity * position_fraction * price_return
                equity = max(equity + pnl, _EPS)

            equity_curve.append((candle.timestamp, equity))
            self._event_bus.publish("market.candle", candle)
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

        metrics = compute_metrics(equity_curve)
        avg_position_size = position_sum / total_bars if total_bars > 0 else 0.0
        time_in_market = (bars_with_position / total_bars * 100.0) if total_bars > 0 else 0.0

        metrics["trade_count"] = float(trade_count)
        metrics["avg_position_size"] = avg_position_size
        metrics["time_in_market"] = time_in_market

        if self._diagnostics:
            _logger.info(
                "Backtest sanity: total_trades=%s avg_position_size=%.4f time_in_market=%.1f%%",
                trade_count,
                avg_position_size,
                time_in_market,
            )

        return BacktestResult(equity_curve=equity_curve, metrics=metrics)
