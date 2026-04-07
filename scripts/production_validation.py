#!/usr/bin/env python3
"""
Production-grade validation: multiple backtests over time windows, offsets, and symbols.

Does not change strategy parameters — only replays the same BacktestEngine on different slices.
"""

from __future__ import annotations

import contextlib
import io
import logging
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.getLogger("alpha_engine").setLevel(logging.WARNING)

from dotenv import load_dotenv

from alpha_engine.backtest import BacktestEngine
from alpha_engine.data.models import OHLCV
from alpha_engine.data_engine.binance_client import BinanceHistoricalClient


def _trade_return_distribution(trs: list[float]) -> dict[str, float]:
    """Simple distribution stats for per-trade PnL fractions."""
    if not trs:
        z = 0.0
        return {
            "n": z,
            "min": z,
            "max": z,
            "mean": z,
            "stdev": z,
            "p25": z,
            "p50": z,
            "p75": z,
        }
    n = len(trs)
    mean = statistics.fmean(trs)
    stdev = statistics.pstdev(trs) if n > 1 else 0.0
    xs = sorted(trs)
    def _p(q: float) -> float:
        if n == 1:
            return xs[0]
        idx = q * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return xs[lo] * (1 - frac) + xs[hi] * frac

    return {
        "n": float(n),
        "min": xs[0],
        "max": xs[-1],
        "mean": mean,
        "stdev": stdev,
        "p25": _p(0.25),
        "p50": _p(0.50),
        "p75": _p(0.75),
    }


@dataclass(frozen=True)
class RunSpec:
    symbol: str
    start_offset: int
    window: int
    label: str


def _slice_candles(candles: list[OHLCV], start: int, window: int) -> list[OHLCV]:
    end = start + window
    if start < 0 or end > len(candles) or window < 50:
        return []
    return candles[start:end]


def main() -> None:
    load_dotenv()
    client = BinanceHistoricalClient()
    timeframe = "1m"
    fetch_limit = 1000

    symbols = ("BTC/USDT", "ETH/USDT")
    raw: dict[str, list[OHLCV]] = {}
    for sym in symbols:
        try:
            raw[sym] = client.fetch_ohlcv(symbol=sym, timeframe=timeframe, limit=fetch_limit)
        except Exception as exc:
            print(f"[WARN] skip symbol={sym} fetch failed: {exc}")
            raw[sym] = []

    windows = (200, 300, 400)
    offsets = (0, 50, 100, 150)

    specs: list[RunSpec] = []
    for sym in symbols:
        candles = raw.get(sym) or []
        if len(candles) < max(windows) + max(offsets):
            continue
        for w in windows:
            for off in offsets:
                if off + w <= len(candles):
                    specs.append(
                        RunSpec(
                            symbol=sym,
                            start_offset=off,
                            window=w,
                            label=f"{sym} off={off} w={w}",
                        )
                    )

    if not specs:
        print("No valid runs: could not fetch enough candles for any symbol.")
        sys.exit(1)

    rows: list[dict[str, float | str | int]] = []
    all_trade_returns: list[float] = []

    for spec in specs:
        candles = raw[spec.symbol]
        data = _slice_candles(candles, spec.start_offset, spec.window)
        if len(data) < 50:
            continue
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            engine = BacktestEngine(
                initial_equity=100.0,
                diagnostics=False,
                force_signal_debug=False,
                force_full_exposure=True,
                min_vol_scale=0.5,
                alpha_visibility=False,
            )
            result = engine.run(data)
        m = result.metrics
        tr = float(m.get("total_return", 0.0))
        atr = float(m.get("avg_trade_return", 0.0))
        tc = int(m.get("trade_count", 0))
        wr = float(m.get("win_rate", 0.0))
        mdd = float(m.get("max_drawdown", 0.0))
        rows.append(
            {
                "label": spec.label,
                "total_return": tr,
                "avg_trade_return": atr,
                "trade_count": tc,
                "win_rate": wr,
                "max_drawdown": mdd,
            }
        )
        all_trade_returns.extend(result.trade_returns)

    if not rows:
        print("No backtest rows produced.")
        sys.exit(1)

    returns_series = [float(r["total_return"]) for r in rows]
    avg_trade_series = [float(r["avg_trade_return"]) for r in rows if int(r["trade_count"]) > 0]
    mdd_series = [float(r["max_drawdown"]) for r in rows]

    mean_total = statistics.fmean(returns_series)
    std_total = statistics.pstdev(returns_series) if len(returns_series) > 1 else 0.0
    mean_avg_trade = statistics.fmean(avg_trade_series) if avg_trade_series else 0.0
    worst_mdd = max(mdd_series) if mdd_series else 0.0

    # Dominance: best run vs median (avoids misleading ratios when sums cross zero)
    best_ret = max(returns_series)
    med_ret = statistics.median(returns_series)
    dominance_ratio = (
        best_ret / med_ret
        if med_ret not in (0.0, -0.0) and len(returns_series) > 1
        else float("nan")
    )

    print("=" * 72)
    print("PRODUCTION VALIDATION — multi-window / offset / symbol")
    print("=" * 72)
    print(f"{'run':<42} {'total_ret':>12} {'avg_trade':>11} {'trades':>7} {'win%':>7} {'max_dd':>8}")
    print("-" * 72)
    for r in rows:
        print(
            f"{r['label']:<42} "
            f"{float(r['total_return']):12.6f} "
            f"{float(r['avg_trade_return']):11.6f} "
            f"{int(r['trade_count']):7d} "
            f"{float(r['win_rate']):7.2%} "
            f"{float(r['max_drawdown']):8.4f}"
        )

    dist = _trade_return_distribution(all_trade_returns)
    print("-" * 72)
    print("AGGREGATE trade-return distribution (all runs, pooled):")
    for k in ("n", "min", "max", "mean", "stdev", "p25", "p50", "p75"):
        print(f"  {k}: {dist[k]:.6f}")

    print("=" * 72)
    print("STABILITY (manual thresholds — interpret with sample size)")
    print(f"  runs with total_return >= 0: {sum(1 for x in returns_series if x >= 0)}/{len(returns_series)}")
    print(f"  runs with total_return in [-0.01, 0.01]: "
          f"{sum(1 for x in returns_series if -0.01 <= x <= 0.01)}/{len(returns_series)}")
    print(f"  runs with avg_trade_return > 0 (when trades > 0): "
          f"{sum(1 for r in rows if int(r['trade_count']) > 0 and float(r['avg_trade_return']) > 0)}/"
          f"{sum(1 for r in rows if int(r['trade_count']) > 0)}")
    print(f"  best vs median total_return ratio: {dominance_ratio:.4f} (NaN if median≈0)")
    print("=" * 72)
    print("SUMMARY")
    print(f"  mean_total_return:   {mean_total:.6f}")
    print(f"  std_total_return:    {std_total:.6f}")
    print(f"  mean_trade_return:   {mean_avg_trade:.6f}  (mean of per-run avg_trade_return)")
    print(f"  max_drawdown:        {worst_mdd:.6f}  (max of per-run max_drawdown)")
    print("=" * 72)


if __name__ == "__main__":
    main()
