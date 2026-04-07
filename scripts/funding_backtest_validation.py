#!/usr/bin/env python3
"""
Standalone backtest: funding contrarian with optional price confirmation.

By default uses ``FundingContrarianAlphaStrategy`` + ``PriceDivergenceContrarianConfirmAlphaStrategy``
(shared buffer, AND-gated in ``AlphaEngine``): strong funding tilt plus
aligned recent return) with ``DEFAULT_FUNDING_CONFIRM_ASSET_CONFIG`` (per-asset
``min_funding_score`` / ``enabled``). Use ``--no-per-asset-config`` for one global
``--min-funding-score``. Pass ``--baseline-only`` for ``FundingContrarianAlphaStrategy`` only.

- Funding and open interest are backward-aligned to each candle (no lookahead).
  OI is fetched via Binance USDM ``fetch_open_interest_history`` (optional data for research);
  signal logic uses funding and price confirmation only.
- Uses BacktestEngine ``simple_signal_execution``: sizing follows ``compute_target_position``.
- By default only ``ETH/USDT`` is run (``_DEFAULT_ENABLED_SYMBOLS``); override with ``--enabled-symbols``.

Robustness (no parameter changes across windows):

- ``--robustness-split``: fetch ``2 * window-bars`` candles, split into older half (A) and
  recent half (B), same engine config on each slice.
- ``--robustness-timeframes``: optional list (e.g. ``1h 15m 4h``) to repeat the split test
  per timeframe; if omitted with ``--robustness-split``, uses ``--timeframe`` only.

Regime band-pass (funding + confirm only; funding alpha / confirm / asset config unchanged):

- ``--regime-min-abs-rr20`` / ``--regime-max-abs-rr20``: finite ``rr20`` and
  ``min <= abs(rolling_return_20) <= max`` (omit max for lower bound only).
  Typical band: min ~0.012, max ~0.05.

Funding structure (before price confirm): rolling ``z`` and ``std`` of funding. **Static** mode
needs ``--funding-regime-z-min`` and ``--funding-regime-std-min``. **Adaptive** mode
(``--funding-regime-adaptive``): percentile thresholds on histories of ``abs(z)`` and ``std``
(see ``--funding-regime-adaptive-*``). Optional ``--funding-regime-same-sign-bars``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Backtest + event_bus emit many INFO traces; this script only prints the summary block.
logging.disable(logging.INFO)

from dotenv import load_dotenv

from alpha_engine.alpha.funding_asset_config import DEFAULT_FUNDING_CONFIRM_ASSET_CONFIG
from alpha_engine.alpha.funding_contrarian_alpha import FundingContrarianAlphaStrategy
from alpha_engine.alpha.funding_price_confirm_alpha import (
    FundingPriceConfirmContrarianAlphaStrategy,
)
from alpha_engine.alpha.price_divergence_confirm_alpha import (
    PriceDivergenceContrarianConfirmAlphaStrategy,
)
from alpha_engine.backtest.engine import BacktestEngine
from alpha_engine.data.models import OHLCV
from alpha_engine.data.funding_rates import (
    attach_funding_to_ohlcv,
    fetch_funding_rate_history_usdm,
    spot_to_usdm_perp,
    timeframe_ms,
)
from alpha_engine.data.open_interest import (
    attach_open_interest_to_ohlcv,
    fetch_open_interest_history_usdm,
    get_oi_coverage_range,
    open_interest_period_for_candle_timeframe,
)
from alpha_engine.data_engine.binance_client import BinanceHistoricalClient

_MAX_CHUNK = 1000

# Per-asset activation: strategy runs only on these symbols unless overridden by ``--enabled-symbols``.
_DEFAULT_ENABLED_SYMBOLS = frozenset({"ETH/USDT"})


def _filter_symbols_by_enabled(symbols: list[str], enabled: frozenset[str]) -> list[str]:
    """Drop symbols not in ``enabled``; print a skip line for each dropped name."""
    out: list[str] = []
    for s in symbols:
        if s not in enabled:
            print(f"[SKIP] {s} — no edge for this strategy")
        else:
            out.append(s)
    return out


def fetch_ohlcv_up_to(
    client: BinanceHistoricalClient,
    symbol: str,
    timeframe: str,
    target_bars: int,
) -> list[OHLCV]:
    """Most recent ``target_bars`` OHLCV rows (or fewer), oldest first, via ccxt paging."""
    ms = timeframe_ms(timeframe)
    tip = client.fetch_ohlcv(symbol, timeframe, limit=1)
    if not tip:
        return []
    end_ts = tip[-1].timestamp
    start_ts = end_ts - int((target_bars + 100) * ms)
    collected: list[OHLCV] = []
    cursor = start_ts
    while len(collected) < target_bars + 1:
        batch = client.fetch_ohlcv(
            symbol,
            timeframe,
            limit=min(_MAX_CHUNK, target_bars + 100),
            since=cursor,
        )
        if not batch:
            break
        if collected and batch[0].timestamp <= collected[-1].timestamp:
            batch = [c for c in batch if c.timestamp > collected[-1].timestamp]
        if not batch:
            break
        collected.extend(batch)
        cursor = batch[-1].timestamp + 1
        if batch[-1].timestamp >= end_ts or len(batch) < _MAX_CHUNK:
            break
    if not collected:
        return []
    by_ts = {c.timestamp: c for c in collected}
    ordered = sorted(by_ts.values(), key=lambda c: c.timestamp)
    return ordered[-target_bars:] if len(ordered) >= target_bars else ordered


def _periods_per_year_for_timeframe(timeframe: str) -> float:
    tf = timeframe.strip().lower()
    if tf.endswith("h"):
        return 365.0 * 24.0 / float(tf[:-1])
    if tf.endswith("m"):
        return 365.0 * 24.0 * 60.0 / float(tf[:-1])
    return 365.0 * 24.0 * 60.0


def _fetch_and_attach_funding(
    symbol: str,
    candles: list[OHLCV],
    timeframe: str,
    z_window: int,
) -> tuple[list[OHLCV], int, int]:
    ts_arr = sorted(c.timestamp for c in candles)
    t0, t1 = ts_arr[0], ts_arr[-1]
    warm_ms = max(
        14 * 86400 * 1000,
        (z_window + 50) * timeframe_ms(timeframe),
    )
    ev = fetch_funding_rate_history_usdm(symbol, t0 - warm_ms, t1)
    out = attach_funding_to_ohlcv(candles, ev)
    oi_tf = open_interest_period_for_candle_timeframe(timeframe)
    oi_ev = fetch_open_interest_history_usdm(
        symbol, t0 - warm_ms, t1, oi_tf
    )
    out = attach_open_interest_to_ohlcv(out, oi_ev)

    oi_start, oi_end = get_oi_coverage_range(oi_ev)
    if oi_start is not None and oi_end is not None:
        ev_trim = [e for e in ev if oi_start <= e[0] <= oi_end]
        oi_ev_trim = [e for e in oi_ev if oi_start <= e[0] <= oi_end]
        candles_trim = [c for c in out if oi_start <= c.timestamp <= oi_end]
        out = attach_funding_to_ohlcv(candles_trim, ev_trim)
        out = attach_open_interest_to_ohlcv(out, oi_ev_trim)
        print(
            f"[OI ALIGNMENT] Trimmed data to OI coverage: {oi_start} → {oi_end} "
            f"| candles={len(out)}"
        )
        if len(out) < 200:
            print(
                "[OI ALIGNMENT WARNING] Very few candles after trimming "
                "— check timeframe or OI fetch"
            )
        if out:
            print(
                "[BACKTEST RANGE AFTER OI ALIGNMENT] "
                f"start={out[0].timestamp} end={out[-1].timestamp} "
                f"total_bars={len(out)}"
            )
        return out, len(ev_trim), len(oi_ev_trim)

    if out:
        print(
            "[BACKTEST RANGE AFTER OI ALIGNMENT] "
            f"start={out[0].timestamp} end={out[-1].timestamp} total_bars={len(out)}"
        )
    return out, len(ev), len(oi_ev)


def _make_engine(
    *,
    timeframe: str,
    z_window: int,
    initial_equity: float,
    use_price_confirm: bool,
    min_funding_score: float,
    per_asset_config: bool,
    regime_abs_return_20_min: float | None,
    regime_abs_return_20_max: float | None,
    funding_regime_z_abs_min: float | None,
    funding_regime_std_min: float | None,
    funding_regime_adaptive: bool,
    funding_regime_adaptive_history: int,
    funding_regime_adaptive_z_percentile: float,
    funding_regime_adaptive_std_percentile: float,
    funding_regime_adaptive_min_history: int,
    funding_regime_window: int | None,
    funding_regime_same_sign_bars: int,
    signal_debug_flow: bool = False,
) -> BacktestEngine:
    if use_price_confirm:
        cfg = DEFAULT_FUNDING_CONFIRM_ASSET_CONFIG if per_asset_config else None
        fund = FundingContrarianAlphaStrategy(z_window=z_window)
        confirm = PriceDivergenceContrarianConfirmAlphaStrategy(
            z_window=z_window,
            funding=fund,
            min_funding_score=min_funding_score,
            asset_config=cfg,
            regime_abs_return_20_min=regime_abs_return_20_min,
            regime_abs_return_20_max=regime_abs_return_20_max,
            funding_regime_z_abs_min=funding_regime_z_abs_min,
            funding_regime_std_min=funding_regime_std_min,
            funding_regime_adaptive=funding_regime_adaptive,
            funding_regime_adaptive_history=funding_regime_adaptive_history,
            funding_regime_adaptive_z_percentile=funding_regime_adaptive_z_percentile,
            funding_regime_adaptive_std_percentile=funding_regime_adaptive_std_percentile,
            funding_regime_adaptive_min_history=funding_regime_adaptive_min_history,
            funding_regime_window=funding_regime_window,
            funding_regime_same_sign_bars=funding_regime_same_sign_bars,
            signal_debug_flow=signal_debug_flow,
        )
        strat_list: list[FundingContrarianAlphaStrategy | PriceDivergenceContrarianConfirmAlphaStrategy] = [
            fund,
            confirm,
        ]
    else:
        strat_list = [FundingContrarianAlphaStrategy(z_window=z_window)]

    return BacktestEngine(
        initial_equity=initial_equity,
        strategies=strat_list,
        signal_filter=None,
        diagnostics=False,
        alpha_visibility=False,
        vol_regime_filter=False,
        simple_signal_execution=True,
        min_holding_period=0,
        cooldown_candles=0,
        min_steps_between_trades=0,
        periods_per_year=_periods_per_year_for_timeframe(timeframe),
        min_signal_threshold=0.0,
    )


def _run_metrics_on_candles_f(
    candles_f: list[OHLCV],
    *,
    timeframe: str,
    z_window: int,
    initial_equity: float,
    use_price_confirm: bool,
    min_funding_score: float,
    per_asset_config: bool,
    regime_abs_return_20_min: float | None,
    regime_abs_return_20_max: float | None,
    funding_regime_z_abs_min: float | None,
    funding_regime_std_min: float | None,
    funding_regime_adaptive: bool,
    funding_regime_adaptive_history: int,
    funding_regime_adaptive_z_percentile: float,
    funding_regime_adaptive_std_percentile: float,
    funding_regime_adaptive_min_history: int,
    funding_regime_window: int | None,
    funding_regime_same_sign_bars: int,
    symbol: str | None = None,
    signal_debug_flow: bool = False,
) -> dict[str, float]:
    buf = io.StringIO()
    dbg_confirm: FundingPriceConfirmContrarianAlphaStrategy | None = None
    with contextlib.redirect_stdout(buf):
        engine = _make_engine(
            timeframe=timeframe,
            z_window=z_window,
            initial_equity=initial_equity,
            use_price_confirm=use_price_confirm,
            min_funding_score=min_funding_score,
            per_asset_config=per_asset_config,
            regime_abs_return_20_min=regime_abs_return_20_min,
            regime_abs_return_20_max=regime_abs_return_20_max,
            funding_regime_z_abs_min=funding_regime_z_abs_min,
            funding_regime_std_min=funding_regime_std_min,
            funding_regime_adaptive=funding_regime_adaptive,
            funding_regime_adaptive_history=funding_regime_adaptive_history,
            funding_regime_adaptive_z_percentile=funding_regime_adaptive_z_percentile,
            funding_regime_adaptive_std_percentile=funding_regime_adaptive_std_percentile,
            funding_regime_adaptive_min_history=funding_regime_adaptive_min_history,
            funding_regime_window=funding_regime_window,
            funding_regime_same_sign_bars=funding_regime_same_sign_bars,
            signal_debug_flow=signal_debug_flow,
        )
        if signal_debug_flow and use_price_confirm:
            for strat in engine._strategies:
                if isinstance(strat, FundingPriceConfirmContrarianAlphaStrategy):
                    dbg_confirm = strat
                    break
            if dbg_confirm is not None:
                dbg_confirm.reset_signal_debug_stats()
        result = engine.run(candles_f)
    if dbg_confirm is not None:
        dbg_confirm.print_signal_debug_stats(symbol or "")
    m = result.metrics
    return {
        "total_return": float(m.get("total_return", 0.0)),
        "sharpe_ratio": float(m.get("sharpe_ratio", 0.0)),
        "avg_trade_return": float(m.get("avg_trade_return", 0.0)),
        "trade_count": int(m.get("trade_count", 0)),
        "max_drawdown": float(m.get("max_drawdown", 0.0)),
    }


def _run_single(
    *,
    client: BinanceHistoricalClient,
    symbol: str,
    timeframe: str,
    limit: int,
    z_window: int,
    initial_equity: float,
    use_price_confirm: bool,
    min_funding_score: float,
    per_asset_config: bool,
    regime_abs_return_20_min: float | None,
    regime_abs_return_20_max: float | None,
    funding_regime_z_abs_min: float | None,
    funding_regime_std_min: float | None,
    funding_regime_adaptive: bool,
    funding_regime_adaptive_history: int,
    funding_regime_adaptive_z_percentile: float,
    funding_regime_adaptive_std_percentile: float,
    funding_regime_adaptive_min_history: int,
    funding_regime_window: int | None,
    funding_regime_same_sign_bars: int,
    signal_debug_flow: bool = False,
) -> None:
    candles = fetch_ohlcv_up_to(client, symbol, timeframe, limit)
    if len(candles) < 200:
        print(f"[ERROR] {symbol}: need >= 200 candles, got {len(candles)}")
        return

    candles_f, n_ev, n_oi = _fetch_and_attach_funding(
        symbol, candles, timeframe, z_window
    )

    m = _run_metrics_on_candles_f(
        candles_f,
        timeframe=timeframe,
        z_window=z_window,
        initial_equity=initial_equity,
        use_price_confirm=use_price_confirm,
        min_funding_score=min_funding_score,
        per_asset_config=per_asset_config,
        regime_abs_return_20_min=regime_abs_return_20_min,
        regime_abs_return_20_max=regime_abs_return_20_max,
        funding_regime_z_abs_min=funding_regime_z_abs_min,
        funding_regime_std_min=funding_regime_std_min,
        funding_regime_window=funding_regime_window,
        funding_regime_same_sign_bars=funding_regime_same_sign_bars,
        funding_regime_adaptive=funding_regime_adaptive,
        funding_regime_adaptive_history=funding_regime_adaptive_history,
        funding_regime_adaptive_z_percentile=funding_regime_adaptive_z_percentile,
        funding_regime_adaptive_std_percentile=funding_regime_adaptive_std_percentile,
        funding_regime_adaptive_min_history=funding_regime_adaptive_min_history,
        symbol=symbol,
        signal_debug_flow=signal_debug_flow,
    )
    total_ret = m["total_return"]
    sharpe = m["sharpe_ratio"]
    avg_trade = m["avg_trade_return"]
    n_trades = int(m["trade_count"])
    mdd = m["max_drawdown"]

    mode = "funding + price confirm" if use_price_confirm else "funding only"
    cfg_note = " per-asset" if (use_price_confirm and per_asset_config) else ""
    print("=" * 72)
    print(f"FUNDING CONTRARIAN — standalone backtest ({mode}{cfg_note})")
    print("=" * 72)
    print(f"  symbol={symbol} perp={spot_to_usdm_perp(symbol)}")
    print(f"  timeframe={timeframe}  n_bars={len(candles_f)}  z_window={z_window}")
    if use_price_confirm and regime_abs_return_20_min is not None:
        if regime_abs_return_20_max is not None:
            print(
                f"  regime: {regime_abs_return_20_min} <= abs(rr20) <= "
                f"{regime_abs_return_20_max} (confirm only)"
            )
        else:
            print(
                f"  regime: abs(rr20) >= {regime_abs_return_20_min} (confirm only)"
            )
    if use_price_confirm and funding_regime_adaptive:
        print(
            f"  funding structure (adaptive): hist={funding_regime_adaptive_history} "
            f"|z| pct>{funding_regime_adaptive_z_percentile} "
            f"std pct>{funding_regime_adaptive_std_percentile} "
            f"min_hist={funding_regime_adaptive_min_history} "
            f"win={funding_regime_window or z_window} sign={funding_regime_same_sign_bars}"
        )
    elif use_price_confirm and funding_regime_z_abs_min is not None:
        print(
            f"  funding structure (static): |z|>{funding_regime_z_abs_min} "
            f"std>{funding_regime_std_min} "
            f"window={funding_regime_window or z_window} "
            f"same_sign_bars={funding_regime_same_sign_bars}"
        )
    print(f"  funding_events_fetched={n_ev}  open_interest_points_fetched={n_oi}")
    print("-" * 72)
    print(f"  total_return:      {total_ret:+.6f}")
    print(f"  sharpe_ratio:      {sharpe:.4f}")
    print(f"  avg_trade_return:  {avg_trade:+.6f}")
    print(f"  trade_count:       {n_trades}")
    print(f"  max_drawdown:      {mdd:.6f}")
    print("-" * 72)

    ok_ret = total_ret >= 0.0
    ok_edge = avg_trade > 1e-4
    ok_trades = n_trades >= 3
    print(
        "  validation: "
        f"total_return>=0 ({ok_ret})  "
        f"avg_trade_return>0 ({ok_edge})  "
        f"trade_count>=3 ({ok_trades})"
    )
    if ok_ret and ok_edge and ok_trades:
        print("  outcome: PASS (heuristic criteria)")
    else:
        print("  outcome: FAIL or inconclusive — extend sample or refine hypothesis")
    print("=" * 72)


def _robustness_consistency_checks(
    symbol: str,
    m_a: dict[str, float],
    m_b: dict[str, float],
) -> list[str]:
    issues: list[str] = []
    ta = int(m_a["trade_count"])
    tb = int(m_b["trade_count"])
    avga = m_a["avg_trade_return"]
    avgb = m_b["avg_trade_return"]
    ra = m_a["total_return"]
    rb = m_b["total_return"]

    if symbol == "SOL/USDT":
        if ta != 0 or tb != 0:
            issues.append("SOL expected disabled (0 trades) in both windows")
    else:
        if ra <= 0.0:
            issues.append(f"window A total_return not positive ({ra:+.4f})")
        if rb <= 0.0:
            issues.append(f"window B total_return not positive ({rb:+.4f})")
        strong_a = ra > 0.05
        strong_b = rb < -0.10
        if strong_a and strong_b:
            issues.append("strong→very_negative flip risk (A strong+, B very-)")
        strong_b_pos = rb > 0.05
        strong_a_neg = ra < -0.10
        if strong_b_pos and strong_a_neg:
            issues.append("strong→very_negative flip risk (B strong+, A very-)")

    if ta >= 3 and tb >= 3:
        if avga > 5e-4 and avgb < -5e-4:
            issues.append("avg_trade_return asymmetric (A ok, B weak negative)")
        if avgb > 5e-4 and avga < -5e-4:
            issues.append("avg_trade_return asymmetric (B ok, A weak negative)")

    return issues


def _run_robustness_split(
    *,
    client: BinanceHistoricalClient,
    symbols: list[str],
    timeframes: list[str],
    window_bars: int,
    z_window: int,
    initial_equity: float,
    use_price_confirm: bool,
    min_funding_score: float,
    per_asset_config: bool,
    regime_abs_return_20_min: float | None,
    regime_abs_return_20_max: float | None,
    funding_regime_z_abs_min: float | None,
    funding_regime_std_min: float | None,
    funding_regime_adaptive: bool,
    funding_regime_adaptive_history: int,
    funding_regime_adaptive_z_percentile: float,
    funding_regime_adaptive_std_percentile: float,
    funding_regime_adaptive_min_history: int,
    funding_regime_window: int | None,
    funding_regime_same_sign_bars: int,
    signal_debug_flow: bool = False,
) -> None:
    need = 2 * window_bars
    print("=" * 72)
    print(f"ROBUSTNESS — split windows (same config, no parameter tuning)")
    print(
        f"  fetch target: {need} bars (2× --window-bars); "
        f"A/B = equal halves of post-OI aligned candles"
    )
    print(f"  per-asset config: {per_asset_config}  price_confirm: {use_price_confirm}")
    if use_price_confirm and regime_abs_return_20_min is not None:
        if regime_abs_return_20_max is not None:
            print(
                f"  regime: {regime_abs_return_20_min} <= abs(rr20) <= "
                f"{regime_abs_return_20_max}"
            )
        else:
            print(f"  regime: abs(rr20) >= {regime_abs_return_20_min}")
    if use_price_confirm and funding_regime_adaptive:
        print(
            f"  funding structure (adaptive): hist={funding_regime_adaptive_history} "
            f"z_pct={funding_regime_adaptive_z_percentile} "
            f"std_pct={funding_regime_adaptive_std_percentile} "
            f"min_hist={funding_regime_adaptive_min_history}"
        )
    elif use_price_confirm and funding_regime_z_abs_min is not None:
        print(
            f"  funding structure (static): |z|>{funding_regime_z_abs_min} "
            f"std>{funding_regime_std_min} "
            f"win={funding_regime_window or z_window} "
            f"sign={funding_regime_same_sign_bars}"
        )
    print("=" * 72)

    for timeframe in timeframes:
        print("")
        print("-" * 72)
        print(f"TIMEFRAME {timeframe}")
        print("-" * 72)
        for symbol in symbols:
            candles = fetch_ohlcv_up_to(client, symbol, timeframe, need)
            if len(candles) < need:
                print(
                    f"[ERROR] {symbol} {timeframe}: need {need} bars, got {len(candles)}"
                )
                continue
            candles_f, n_ev, n_oi = _fetch_and_attach_funding(
                symbol, candles, timeframe, z_window
            )
            total_bars = len(candles_f)
            split_idx = total_bars // 2
            window_a = candles_f[:split_idx]
            window_b = candles_f[split_idx:]
            print(
                f"windows: A={len(window_a)} bars | B={len(window_b)} bars "
                f"| fetch_total={total_bars}"
            )
            if len(window_b) == 0:
                print(
                    "[ERROR] Window B is empty — check data split logic"
                )
                continue

            m_a = _run_metrics_on_candles_f(
                window_a,
                timeframe=timeframe,
                z_window=z_window,
                initial_equity=initial_equity,
                use_price_confirm=use_price_confirm,
                min_funding_score=min_funding_score,
                per_asset_config=per_asset_config,
                regime_abs_return_20_min=regime_abs_return_20_min,
                regime_abs_return_20_max=regime_abs_return_20_max,
                funding_regime_z_abs_min=funding_regime_z_abs_min,
                funding_regime_std_min=funding_regime_std_min,
                funding_regime_window=funding_regime_window,
                funding_regime_same_sign_bars=funding_regime_same_sign_bars,
                funding_regime_adaptive=funding_regime_adaptive,
                funding_regime_adaptive_history=funding_regime_adaptive_history,
                funding_regime_adaptive_z_percentile=funding_regime_adaptive_z_percentile,
                funding_regime_adaptive_std_percentile=funding_regime_adaptive_std_percentile,
                funding_regime_adaptive_min_history=funding_regime_adaptive_min_history,
                symbol=f"{symbol} window=A ({timeframe})",
                signal_debug_flow=signal_debug_flow,
            )
            m_b = _run_metrics_on_candles_f(
                window_b,
                timeframe=timeframe,
                z_window=z_window,
                initial_equity=initial_equity,
                use_price_confirm=use_price_confirm,
                min_funding_score=min_funding_score,
                per_asset_config=per_asset_config,
                regime_abs_return_20_min=regime_abs_return_20_min,
                regime_abs_return_20_max=regime_abs_return_20_max,
                funding_regime_z_abs_min=funding_regime_z_abs_min,
                funding_regime_std_min=funding_regime_std_min,
                funding_regime_window=funding_regime_window,
                funding_regime_same_sign_bars=funding_regime_same_sign_bars,
                funding_regime_adaptive=funding_regime_adaptive,
                funding_regime_adaptive_history=funding_regime_adaptive_history,
                funding_regime_adaptive_z_percentile=funding_regime_adaptive_z_percentile,
                funding_regime_adaptive_std_percentile=funding_regime_adaptive_std_percentile,
                funding_regime_adaptive_min_history=funding_regime_adaptive_min_history,
                symbol=f"{symbol} window=B ({timeframe})",
                signal_debug_flow=signal_debug_flow,
            )

            issues = _robustness_consistency_checks(symbol, m_a, m_b)
            status = "OK" if not issues else "REVIEW"

            print(
                f"\n  {symbol}  (funding_events={n_ev} open_interest_points={n_oi})"
            )
            print(
                f"    window A (older)  ret={m_a['total_return']:+.6f}  "
                f"avg_trade={m_a['avg_trade_return']:+.6f}  trades={int(m_a['trade_count'])}"
            )
            print(
                f"    window B (recent) ret={m_b['total_return']:+.6f}  "
                f"avg_trade={m_b['avg_trade_return']:+.6f}  trades={int(m_b['trade_count'])}"
            )
            print(f"    consistency: {status}")
            for msg in issues:
                print(f"      - {msg}")
        print("")

    print("=" * 72)
    print("Interpretation: look for REVIEW lines (flip risk / SOL trading / avg_trade gap).")
    print("=" * 72)


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description="Backtest funding contrarian alpha only")
    p.add_argument(
        "--symbol",
        default="ETH/USDT",
        help="Single symbol when --symbols is not set (default aligns with per-asset allowlist).",
    )
    p.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        metavar="SYM",
        help="Run several symbols (e.g. BTC/USDT ETH/USDT SOL/USDT). Overrides --symbol when set.",
    )
    p.add_argument(
        "--enabled-symbols",
        nargs="+",
        default=None,
        metavar="SYM",
        help=(
            "Only run backtests for these symbols. "
            "Default: ETH/USDT only (per-asset edge filter)."
        ),
    )
    p.add_argument("--timeframe", default="1h")
    p.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Target bar count (paged up to ~10k via 1k chunks).",
    )
    p.add_argument("--z-window", type=int, default=168)
    p.add_argument("--initial-equity", type=float, default=100.0)
    p.add_argument(
        "--baseline-only",
        action="store_true",
        help="Use FundingContrarianAlphaStrategy only (no price confirmation).",
    )
    p.add_argument(
        "--min-funding-score",
        type=float,
        default=0.12,
        help="Default minimum |funding score| for symbols not listed in per-asset config.",
    )
    p.add_argument(
        "--no-per-asset-config",
        action="store_true",
        help="Ignore DEFAULT_FUNDING_CONFIRM_ASSET_CONFIG (single global min_funding_score).",
    )
    p.add_argument(
        "--robustness-split",
        action="store_true",
        help="Non-overlapping A/B windows (older vs recent), same thresholds and asset config.",
    )
    p.add_argument(
        "--window-bars",
        type=int,
        default=3500,
        help="Bars per window when using --robustness-split (fetches 2× this many).",
    )
    p.add_argument(
        "--robustness-timeframes",
        nargs="+",
        default=None,
        metavar="TF",
        help="With --robustness-split: run each timeframe (e.g. 1h 15m 4h). Default: --timeframe only.",
    )
    p.add_argument(
        "--regime-min-abs-rr20",
        type=float,
        default=None,
        help=(
            "Funding+confirm only: band-pass lower bound on abs(rolling_return_20) "
            "(e.g. 0.012). Omit to disable regime filter."
        ),
    )
    p.add_argument(
        "--regime-max-abs-rr20",
        type=float,
        default=None,
        help=(
            "Funding+confirm only: band-pass upper bound (e.g. 0.05). "
            "Requires --regime-min-abs-rr20."
        ),
    )
    p.add_argument(
        "--funding-regime-z-min",
        type=float,
        default=None,
        help="Funding+confirm: require |z-score of funding| > this (needs --funding-regime-std-min).",
    )
    p.add_argument(
        "--funding-regime-std-min",
        type=float,
        default=None,
        help="Funding+confirm: require rolling std(funding) > this (rate scale, e.g. 5e-6).",
    )
    p.add_argument(
        "--funding-regime-window",
        type=int,
        default=None,
        help="Bars for funding mean/std (default: same as --z-window).",
    )
    p.add_argument(
        "--funding-regime-same-sign-bars",
        type=int,
        default=0,
        help="If >0, last N funded updates must share the same strict sign.",
    )
    p.add_argument(
        "--funding-regime-adaptive",
        action="store_true",
        help=(
            "Percentile-based funding |z| and std gates (exclude --funding-regime-z-min/std-min)."
        ),
    )
    p.add_argument(
        "--funding-regime-adaptive-history",
        type=int,
        default=400,
        help="Max length of abs(z) and std histories (e.g. 200–500).",
    )
    p.add_argument(
        "--funding-regime-adaptive-z-pct",
        type=float,
        default=75.0,
        help="Trade only if abs(z) exceeds this percentile of past abs(z) (e.g. 70–80).",
    )
    p.add_argument(
        "--funding-regime-adaptive-std-pct",
        type=float,
        default=55.0,
        help="Trade only if std exceeds this percentile of past std (e.g. 50–60).",
    )
    p.add_argument(
        "--funding-regime-adaptive-min-history",
        type=int,
        default=80,
        help="Bars of history before adaptive percentiles apply (warm-up allows all).",
    )
    p.add_argument(
        "--debug-signal-flow",
        action="store_true",
        help=(
            "After each run, print PriceDivergence / funding+confirm stage counts "
            "(regime rr20 → funding leg → price confirm)."
        ),
    )
    args = p.parse_args()

    if args.regime_max_abs_rr20 is not None and args.regime_min_abs_rr20 is None:
        p.error("--regime-max-abs-rr20 requires --regime-min-abs-rr20")
    if (
        args.regime_min_abs_rr20 is not None
        and args.regime_max_abs_rr20 is not None
        and float(args.regime_max_abs_rr20) <= float(args.regime_min_abs_rr20)
    ):
        p.error("--regime-max-abs-rr20 must be greater than --regime-min-abs-rr20")

    if args.funding_regime_adaptive:
        if args.funding_regime_z_min is not None or args.funding_regime_std_min is not None:
            p.error(
                "--funding-regime-adaptive is incompatible with static "
                "--funding-regime-z-min / --funding-regime-std-min"
            )
    elif (args.funding_regime_z_min is None) ^ (args.funding_regime_std_min is None):
        p.error(
            "--funding-regime-z-min and --funding-regime-std-min must be set together or omitted"
        )
    if args.funding_regime_window is not None and int(args.funding_regime_window) < 10:
        p.error("--funding-regime-window must be >= 10 when set")
    if args.funding_regime_adaptive:
        ah = max(50, int(args.funding_regime_adaptive_history))
        am = int(args.funding_regime_adaptive_min_history)
        if am < 10 or am > ah:
            p.error(
                "--funding-regime-adaptive-min-history must be in [10, adaptive-history]"
            )

    syms = args.symbols if args.symbols else [args.symbol]
    enabled_set = (
        frozenset(args.enabled_symbols)
        if args.enabled_symbols is not None
        else _DEFAULT_ENABLED_SYMBOLS
    )
    syms = _filter_symbols_by_enabled(syms, enabled_set)
    if not syms:
        print("[ERROR] No symbols to run after enabled-symbols filter.")
        sys.exit(1)

    client = BinanceHistoricalClient()
    use_confirm = not args.baseline_only
    per_asset = not args.no_per_asset_config
    regime_min = (
        float(args.regime_min_abs_rr20)
        if args.regime_min_abs_rr20 is not None
        else None
    )
    regime_max = (
        float(args.regime_max_abs_rr20)
        if args.regime_max_abs_rr20 is not None
        else None
    )
    fr_z = (
        float(args.funding_regime_z_min)
        if args.funding_regime_z_min is not None
        else None
    )
    fr_std = (
        float(args.funding_regime_std_min)
        if args.funding_regime_std_min is not None
        else None
    )
    fr_win = args.funding_regime_window
    fr_sign = max(0, int(args.funding_regime_same_sign_bars))
    fr_adaptive = bool(args.funding_regime_adaptive)
    fr_ad_hist = max(50, int(args.funding_regime_adaptive_history))
    fr_ad_zp = float(args.funding_regime_adaptive_z_pct)
    fr_ad_sp = float(args.funding_regime_adaptive_std_pct)
    fr_ad_minh = int(args.funding_regime_adaptive_min_history)

    if args.robustness_split:
        tfs = (
            args.robustness_timeframes
            if args.robustness_timeframes
            else [args.timeframe]
        )
        _run_robustness_split(
            client=client,
            symbols=syms,
            timeframes=tfs,
            window_bars=max(500, args.window_bars),
            z_window=args.z_window,
            initial_equity=args.initial_equity,
            use_price_confirm=use_confirm,
            min_funding_score=args.min_funding_score,
            per_asset_config=per_asset,
            regime_abs_return_20_min=regime_min,
            regime_abs_return_20_max=regime_max,
            funding_regime_z_abs_min=fr_z,
            funding_regime_std_min=fr_std,
            funding_regime_window=fr_win,
            funding_regime_same_sign_bars=fr_sign,
            funding_regime_adaptive=fr_adaptive,
            funding_regime_adaptive_history=fr_ad_hist,
            funding_regime_adaptive_z_percentile=fr_ad_zp,
            funding_regime_adaptive_std_percentile=fr_ad_sp,
            funding_regime_adaptive_min_history=fr_ad_minh,
            signal_debug_flow=bool(args.debug_signal_flow),
        )
        return

    for sym in syms:
        _run_single(
            client=client,
            symbol=sym,
            timeframe=args.timeframe,
            limit=max(200, args.limit),
            z_window=args.z_window,
            initial_equity=args.initial_equity,
            use_price_confirm=use_confirm,
            min_funding_score=args.min_funding_score,
            per_asset_config=per_asset,
            regime_abs_return_20_min=regime_min,
            regime_abs_return_20_max=regime_max,
            funding_regime_z_abs_min=fr_z,
            funding_regime_std_min=fr_std,
            funding_regime_window=fr_win,
            funding_regime_same_sign_bars=fr_sign,
            funding_regime_adaptive=fr_adaptive,
            funding_regime_adaptive_history=fr_ad_hist,
            funding_regime_adaptive_z_percentile=fr_ad_zp,
            funding_regime_adaptive_std_percentile=fr_ad_sp,
            funding_regime_adaptive_min_history=fr_ad_minh,
            signal_debug_flow=bool(args.debug_signal_flow),
        )


if __name__ == "__main__":
    main()
