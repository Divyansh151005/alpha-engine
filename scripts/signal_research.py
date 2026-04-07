#!/usr/bin/env python3
"""
Standalone signal research: correlate candidate signals with a forward *target* (same validation always).

Uses Binance OHLCV for data (same client as production_validation).
Does not import or modify the backtest engine, execution, or alpha strategies.

Next-phase research (no engine changes):
  * **Target:** ``--target return`` (default) or ``--target vol`` — forward realized vol of the
    next ``horizon`` bar returns (Option C: predict volatility, not direction).
  * **Timeframe:** ``--timeframe 5m`` / ``15m`` / ``1h`` (Option B: same features, different bar clock).
  * **Funding (USDT-M)** — fetched via ``ccxt.binanceusdm``, backward-aligned to candle timestamps,
    signal ``funding_contrarian_z`` (rolling z of rate, negated so extreme negative funding → high signal).
    Disable with ``--no-funding``. Open interest / book imbalance remain future extensions.

Accept a candidate only if ALL hold (economic edge, not statistical noise):

  1. abs(correlation(signal, target)) > 0.03
  2. top-decile mean target > 0  (for ``vol``, RV is naturally positive when defined)
  3. top-decile mean > overall mean target (same valid sample)
  4. top-decile mean > abs(bottom-decile mean)
  5. edge = top_decile_mean - bottom_decile_mean > 0

Top-decile one-sided p-value is printed for reference only — not used for accept/reject.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.getLogger("alpha_engine").setLevel(logging.WARNING)

from dotenv import load_dotenv

from alpha_engine.data.funding_rates import (
    align_funding_to_timestamps,
    fetch_funding_rate_history_usdm,
    signal_funding_contrarian_z,
    spot_to_usdm_perp,
    timeframe_ms,
)
from alpha_engine.data.models import OHLCV
from alpha_engine.data_engine.binance_client import BinanceHistoricalClient


# ---------------------------------------------------------------------------
# Arrays from candles
# ---------------------------------------------------------------------------


def _ohlcv_arrays(candles: list[OHLCV]) -> dict[str, np.ndarray]:
    """Sorted by time; returns float arrays."""
    xs = sorted(candles, key=lambda c: c.timestamp)
    return {
        "ts": np.array([c.timestamp for c in xs], dtype=np.int64),
        "open": np.array([c.open for c in xs], dtype=np.float64),
        "high": np.array([c.high for c in xs], dtype=np.float64),
        "low": np.array([c.low for c in xs], dtype=np.float64),
        "close": np.array([c.close for c in xs], dtype=np.float64),
        "volume": np.array([c.volume for c in xs], dtype=np.float64),
    }


def forward_returns(close: np.ndarray, horizon: int) -> np.ndarray:
    """Simple return from t to t+horizon: close[t+h]/close[t] - 1. Last `horizon` bars are NaN."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if horizon <= 0 or n <= horizon:
        return out
    out[: n - horizon] = close[horizon:] / close[: n - horizon] - 1.0
    return out


def forward_realized_vol(close: np.ndarray, horizon: int) -> np.ndarray:
    """
    Per bar t: sample std (ddof=1) of the next ``horizon`` simple bar returns
    (returns ending at bars t+1 .. t+horizon). Last ``horizon`` indices undefined.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if horizon < 2 or n <= horizon:
        return out
    lr = np.zeros(n, dtype=np.float64)
    lr[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-12)
    for t in range(0, n - horizon):
        chunk = lr[t + 1 : t + horizon + 1]
        out[t] = float(np.std(chunk, ddof=1))
    return out


def build_forward_target(close: np.ndarray, horizon: int, target: str) -> np.ndarray:
    """``target`` is ``return`` or ``vol`` (forward realized volatility)."""
    if target == "return":
        return forward_returns(close, horizon)
    if target == "vol":
        return forward_realized_vol(close, horizon)
    raise ValueError(f"unknown target={target!r} (use 'return' or 'vol')")


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    c = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
    out[window - 1 :] = (c[window:] - c[:-window]) / window
    return out


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    m = _rolling_mean(x, window)
    m2 = _rolling_mean(x * x, window)
    var = m2 - m * m
    var = np.maximum(var, 0.0)
    return np.sqrt(var)


# ---------------------------------------------------------------------------
# Candidate signals (high values = hypothesized long edge where applicable)
# ---------------------------------------------------------------------------


def signal_vol_expansion(close: np.ndarray, short_w: int = 5, long_w: int = 20) -> np.ndarray:
    """Log-return volatility short / long - 1; expansion when positive."""
    n = len(close)
    lr = np.zeros(n, dtype=np.float64)
    lr[1:] = np.log(close[1:] / close[:-1])
    s = _rolling_std(lr, short_w)
    l = _rolling_std(lr, long_w)
    out = np.full_like(close, np.nan)
    mask = np.isfinite(s) & np.isfinite(l) & (l > 1e-12)
    out[mask] = s[mask] / l[mask] - 1.0
    return out


def signal_mean_reversion_long(close: np.ndarray, window: int = 20) -> np.ndarray:
    """Oversold -> positive signal: -(close - SMA) / rolling_std."""
    ma = _rolling_mean(close, window)
    sd = _rolling_std(close, window)
    out = np.full_like(close, np.nan)
    mask = np.isfinite(ma) & np.isfinite(sd) & (sd > 1e-12)
    out[mask] = -(close[mask] - ma[mask]) / sd[mask]
    return out


def signal_rsi_long(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Rolling RSI in [0,100]; long-bias score (50 - RSI) / 50 so oversold -> positive."""
    n = len(close)
    delta = np.zeros(n, dtype=np.float64)
    delta[1:] = close[1:] - close[:-1]
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)
    avg_g = _rolling_mean(gain, period)
    avg_l = _rolling_mean(loss, period)
    rs = np.full(n, np.nan, dtype=np.float64)
    mask = np.isfinite(avg_g) & np.isfinite(avg_l) & (avg_l > 1e-12)
    rs[mask] = avg_g[mask] / avg_l[mask]
    rsi = np.full(n, np.nan, dtype=np.float64)
    ok = np.isfinite(rs)
    rsi[ok] = 100.0 - (100.0 / (1.0 + rs[ok]))
    score = np.full(n, np.nan, dtype=np.float64)
    ok2 = np.isfinite(rsi)
    score[ok2] = (50.0 - rsi[ok2]) / 50.0
    return score


def signal_volume_spike(volume: np.ndarray, window: int = 20) -> np.ndarray:
    """Relative volume vs SMA - 1."""
    ma = _rolling_mean(volume, window)
    out = np.full_like(volume, np.nan, dtype=np.float64)
    mask = np.isfinite(ma) & (ma > 1e-12)
    out[mask] = volume[mask] / ma[mask] - 1.0
    return out


def signal_short_term_reversal(close: np.ndarray, vol_window: int = 20) -> np.ndarray:
    """
    Extreme latest bar return vs recent volatility: fade the spike.

    recent_return = latest_bar_return (simple return ending this bar).
    volatility = sample std of the last ``vol_window`` bar returns (including current).
    """
    n = len(close)
    lr = np.full(n, np.nan, dtype=np.float64)
    lr[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-12)
    vol = np.full(n, np.nan, dtype=np.float64)
    for i in range(vol_window - 1, n):
        w = lr[i - (vol_window - 1) : i + 1]
        if np.all(np.isfinite(w)):
            vol[i] = float(np.std(w, ddof=1))

    out = np.full(n, np.nan, dtype=np.float64)
    eps = 1e-12
    for i in range(vol_window - 1, n):
        r = lr[i]
        v = vol[i]
        if not (math.isfinite(r) and math.isfinite(v) and v > eps):
            continue
        if r > 2.0 * v:
            out[i] = -1.0
        elif r < -2.0 * v:
            out[i] = 1.0
        else:
            out[i] = 0.0
    return out


def signal_delayed_reversal(
    close: np.ndarray,
    vol_window: int = 20,
    shift: int = 1,
) -> np.ndarray:
    """
    Same extreme detection as short_term_reversal, but trade after a delay.

    At bar ``t`` we flag spike_up / spike_down when ``abs(recent_return) > 2 * vol``.
    The tradable signal is placed ``shift`` bars later (liquidity exhaustion / not immediate).

    - spike_up  (r > 2v) -> signal at t+shift = -1 (short)
    - spike_down (r < -2v) -> signal at t+shift = +1 (long)
    """
    if shift < 1:
        raise ValueError("shift must be >= 1")

    n = len(close)
    lr = np.full(n, np.nan, dtype=np.float64)
    lr[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-12)
    vol = np.full(n, np.nan, dtype=np.float64)
    for i in range(vol_window - 1, n):
        w = lr[i - (vol_window - 1) : i + 1]
        if np.all(np.isfinite(w)):
            vol[i] = float(np.std(w, ddof=1))

    spike_up = np.zeros(n, dtype=bool)
    spike_down = np.zeros(n, dtype=bool)
    eps = 1e-12
    for i in range(vol_window - 1, n):
        r = lr[i]
        v = vol[i]
        if not (math.isfinite(r) and math.isfinite(v) and v > eps):
            continue
        if abs(r) <= 2.0 * v:
            continue
        if r > 2.0 * v:
            spike_up[i] = True
        elif r < -2.0 * v:
            spike_down[i] = True

    out = np.full(n, np.nan, dtype=np.float64)
    start = vol_window - 1 + shift
    for i in range(start, n):
        src = i - shift
        if spike_up[src]:
            out[i] = -1.0
        elif spike_down[src]:
            out[i] = 1.0
        else:
            out[i] = 0.0
    return out


def signal_momentum_persistence(close: np.ndarray) -> np.ndarray:
    """
    Short-term return clustering: three consecutive same-sign bar returns.

    r1 = latest bar return, r2 = one bar ago, r3 = two bars ago (simple returns).
    All three > 0 -> +1; all three < 0 -> -1; else 0.
    """
    n = len(close)
    lr = np.full(n, np.nan, dtype=np.float64)
    lr[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-12)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(2, n):
        r1, r2, r3 = lr[i], lr[i - 1], lr[i - 2]
        if not (math.isfinite(r1) and math.isfinite(r2) and math.isfinite(r3)):
            continue
        if r1 > 0.0 and r2 > 0.0 and r3 > 0.0:
            out[i] = 1.0
        elif r1 < 0.0 and r2 < 0.0 and r3 < 0.0:
            out[i] = -1.0
        else:
            out[i] = 0.0
    return out


def signal_cross_asset_ratio_momentum(
    close_a: np.ndarray,
    close_b: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """log(A/B) z-score change: short-term vs longer mean — aligned arrays required (same length)."""
    if len(close_a) != len(close_b):
        raise ValueError("close_a and close_b must have same length")
    ratio = np.log(close_a / np.maximum(close_b, 1e-12))
    ma = _rolling_mean(ratio, window)
    sd = _rolling_std(ratio, window)
    z = np.full_like(ratio, np.nan)
    mask = np.isfinite(ma) & np.isfinite(sd) & (sd > 1e-12)
    z[mask] = (ratio[mask] - ma[mask]) / sd[mask]
    return z


def align_candles_by_timestamp(
    a: list[OHLCV],
    b: list[OHLCV],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    """Inner join on timestamp; returns array dicts of equal length."""
    map_b = {c.timestamp: c for c in b}
    aa: list[OHLCV] = []
    bb: list[OHLCV] = []
    for ca in sorted(a, key=lambda c: c.timestamp):
        cb = map_b.get(ca.timestamp)
        if cb is not None:
            aa.append(ca)
            bb.append(cb)
    if len(aa) < 50:
        return None
    return _ohlcv_arrays(aa), _ohlcv_arrays(bb)


# ---------------------------------------------------------------------------
# Metrics & acceptance
# ---------------------------------------------------------------------------


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def one_sided_pvalue_mean_gt_zero(x: np.ndarray) -> tuple[float, float, float]:
    """
    H0: mean <= 0 vs H1: mean > 0.
    Returns (mean, t_stat, p_value_one_sided) using normal approximation.
    """
    v = x[np.isfinite(x)]
    n = int(v.size)
    if n == 0:
        return float("nan"), float("nan"), 1.0
    m = float(np.mean(v))
    if n == 1:
        return m, float("inf") if m > 0 else float("-inf"), 0.0 if m > 0 else 1.0
    s = float(np.std(v, ddof=1))
    se = s / math.sqrt(n)
    if se <= 1e-18:
        return m, float("inf") if m > 0 else float("-inf"), 0.0 if m > 0 else 1.0
    t = m / se
    p = 1.0 - _normal_cdf(t)
    return m, t, p


@dataclass(frozen=True)
class SignalEval:
    name: str
    n: int
    correlation: float
    mean_top_decile: float
    mean_bottom_decile: float
    mean_overall: float
    edge: float
    top_decile_p_gt_zero: float
    accepted: bool
    reason: str


def evaluate_signal(
    name: str,
    signal: np.ndarray,
    fwd: np.ndarray,
    *,
    corr_threshold: float = 0.03,
    decile: float = 0.10,
) -> SignalEval:
    mask = np.isfinite(signal) & np.isfinite(fwd)
    s = signal[mask]
    r = fwd[mask]
    n = int(s.size)
    if n < 30:
        return SignalEval(
            name=name,
            n=n,
            correlation=float("nan"),
            mean_top_decile=float("nan"),
            mean_bottom_decile=float("nan"),
            mean_overall=float("nan"),
            edge=float("nan"),
            top_decile_p_gt_zero=1.0,
            accepted=False,
            reason="too_few_pairs",
        )

    corr = float(np.corrcoef(s, r)[0, 1]) if n >= 2 else float("nan")
    mean_overall = float(np.mean(r))

    order = np.argsort(s)
    k = max(1, int(math.ceil(decile * n)))
    idx_bottom = order[:k]
    idx_top = order[-k:]
    top_r = r[idx_top]
    bot_r = r[idx_bottom]

    mean_top = float(np.mean(top_r))
    mean_bot = float(np.mean(bot_r))
    edge = mean_top - mean_bot
    _, _, p_top = one_sided_pvalue_mean_gt_zero(top_r)

    ok_abs_corr = not math.isnan(corr) and abs(corr) > corr_threshold
    ok_top_pos = mean_top > 0.0
    ok_top_vs_overall = mean_top > mean_overall
    ok_top_vs_abs_bot = mean_top > abs(mean_bot)
    ok_edge = edge > 0.0

    accepted = ok_abs_corr and ok_top_pos and ok_top_vs_overall and ok_top_vs_abs_bot and ok_edge
    reason = "accept" if accepted else "reject"

    return SignalEval(
        name=name,
        n=n,
        correlation=corr,
        mean_top_decile=mean_top,
        mean_bottom_decile=mean_bot,
        mean_overall=mean_overall,
        edge=edge,
        top_decile_p_gt_zero=p_top,
        accepted=accepted,
        reason=reason,
    )


SIGNAL_BUILDERS: dict[str, Callable[[dict[str, np.ndarray]], np.ndarray]] = {
    "vol_expansion": lambda a: signal_vol_expansion(a["close"], 5, 20),
    "mean_reversion_z": lambda a: signal_mean_reversion_long(a["close"], 20),
    "rsi_long": lambda a: signal_rsi_long(a["close"], 14),
    "volume_spike": lambda a: signal_volume_spike(a["volume"], 20),
    "short_term_reversal": lambda a: signal_short_term_reversal(a["close"], 20),
    "delayed_reversal": lambda a: signal_delayed_reversal(a["close"], 20, shift=1),
    "delayed_reversal_2": lambda a: signal_delayed_reversal(a["close"], 20, shift=2),
    "momentum_persistence": lambda a: signal_momentum_persistence(a["close"]),
}


def run_single_asset(
    arrays: dict[str, np.ndarray],
    horizon: int,
    corr_threshold: float,
    target: str,
    *,
    funding_z_window: int | None = None,
) -> list[SignalEval]:
    fwd = build_forward_target(arrays["close"], horizon, target)
    builders: dict[str, Callable[[dict[str, np.ndarray]], np.ndarray]] = dict(SIGNAL_BUILDERS)
    if funding_z_window is not None and "funding_rate" in arrays:
        w = int(funding_z_window)
        builders["funding_contrarian_z"] = lambda a, zz=w: signal_funding_contrarian_z(
            a["funding_rate"], zz
        )
    out: list[SignalEval] = []
    for name, builder in builders.items():
        sig = builder(arrays)
        out.append(
            evaluate_signal(
                name,
                sig,
                fwd,
                corr_threshold=corr_threshold,
            )
        )
    return out


def run_cross_asset(
    ar_a: dict[str, np.ndarray],
    ar_b: dict[str, np.ndarray],
    horizon: int,
    corr_threshold: float,
    target: str,
) -> SignalEval:
    sig = signal_cross_asset_ratio_momentum(ar_a["close"], ar_b["close"], 20)
    fwd = build_forward_target(ar_a["close"], horizon, target)
    return evaluate_signal(
        "cross_asset_ratio_z",
        sig,
        fwd,
        corr_threshold=corr_threshold,
    )


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(
        description="Standalone signal research: same validation vs forward return or forward realized vol."
    )
    p.add_argument("--symbol", default="BTC/USDT", help="Primary symbol (default BTC/USDT)")
    p.add_argument("--symbol-b", default="ETH/USDT", help="Second symbol for cross-asset (default ETH/USDT)")
    p.add_argument(
        "--timeframe",
        default="1m",
        help="Bar size (e.g. 1m, 5m, 15m, 1h) — same pipeline, different clock",
    )
    p.add_argument("--limit", type=int, default=1500, help="Number of candles to fetch")
    p.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Forward horizon in bars (for return: total return; for vol: std of next h bar returns)",
    )
    p.add_argument(
        "--target",
        choices=("return", "vol"),
        default="return",
        help="Forward target: total simple return (return) or forward realized vol of next h returns (vol)",
    )
    p.add_argument(
        "--corr-threshold",
        type=float,
        default=0.03,
        help="Require abs(correlation) > this (default 0.03)",
    )
    p.add_argument("--no-cross-asset", action="store_true", help="Skip cross-asset signal")
    p.add_argument(
        "--no-funding",
        action="store_true",
        help="Skip USDT-M perpetual funding rate fetch (non-price source)",
    )
    p.add_argument(
        "--funding-z-window",
        type=int,
        default=240,
        help="Rolling bars for funding z-score (default 240; use smaller TF → more bars per day)",
    )
    args = p.parse_args()

    client = BinanceHistoricalClient()
    try:
        candles = client.fetch_ohlcv(symbol=args.symbol, timeframe=args.timeframe, limit=args.limit)
    except Exception as exc:
        print(f"[ERROR] fetch failed for {args.symbol}: {exc}")
        sys.exit(1)

    if len(candles) < 100:
        print(f"[ERROR] need at least 100 candles, got {len(candles)}")
        sys.exit(1)

    ar = _ohlcv_arrays(candles)
    funding_z_window: int | None = None
    if not args.no_funding:
        ts_arr = ar["ts"]
        t0, t1 = int(ts_arr[0]), int(ts_arr[-1])
        warm_ms = max(14 * 86400 * 1000, (args.funding_z_window + 50) * timeframe_ms(args.timeframe))
        try:
            ev = fetch_funding_rate_history_usdm(args.symbol, t0 - warm_ms, t1)
            aligned = align_funding_to_timestamps(ts_arr, ev)
            ar = {**ar, "funding_rate": aligned}
            funding_z_window = args.funding_z_window
            n_cov = int(np.sum(np.isfinite(aligned)))
            print(
                f"[INFO] funding USDM perp={spot_to_usdm_perp(args.symbol)} "
                f"events={len(ev)} aligned_non_nan={n_cov}/{len(aligned)} z_window={args.funding_z_window}"
            )
        except Exception as exc:
            print(f"[WARN] funding series unavailable (non-price research degraded): {exc}")

    results = run_single_asset(
        ar,
        args.horizon,
        args.corr_threshold,
        args.target,
        funding_z_window=funding_z_window,
    )

    cross_eval: SignalEval | None = None
    if not args.no_cross_asset:
        try:
            other = client.fetch_ohlcv(symbol=args.symbol_b, timeframe=args.timeframe, limit=args.limit)
        except Exception as exc:
            print(f"[WARN] cross-asset fetch failed for {args.symbol_b}: {exc}")
            other = []
        aligned = align_candles_by_timestamp(candles, other)
        if aligned is not None:
            a1, a2 = aligned
            cross_eval = run_cross_asset(
                a1, a2, args.horizon, args.corr_threshold, args.target
            )

    tgt_label = "fwd_ret" if args.target == "return" else "fwd_rv"
    print("=" * 112)
    print("SIGNAL RESEARCH (standalone — not engine validation)")
    print("=" * 112)
    print(
        f"symbol={args.symbol} timeframe={args.timeframe} n_candles={len(candles)} "
        f"horizon={args.horizon} target={args.target}"
    )
    print(
        "accept if ALL: "
        f"|corr|>{args.corr_threshold} "
        "AND top_decile_mean>0 "
        "AND top_mean>overall_mean "
        "AND top_mean>|bot_mean| "
        "AND edge>0"
    )
    print("  (p-value column is report-only)")
    print("-" * 112)
    hdr = (
        f"{'signal':<22} {'n':>6} {'corr':>10} {f'overall_{tgt_label}':>12} "
        f"{f'top10%_{tgt_label}':>12} {f'bot10%_{tgt_label}':>12} {'edge':>12} {'p(top>0)':>10} {'verdict':>8}"
    )
    print(hdr)
    print("-" * 112)

    for ev in results:
        verdict = "ACCEPT" if ev.accepted else "reject"
        print(
            f"{ev.name:<22} {ev.n:6d} {ev.correlation:10.4f} {ev.mean_overall:12.6f} "
            f"{ev.mean_top_decile:12.6f} {ev.mean_bottom_decile:12.6f} {ev.edge:12.6f} "
            f"{ev.top_decile_p_gt_zero:10.4f} {verdict:>8}"
        )

    if cross_eval is not None:
        verdict = "ACCEPT" if cross_eval.accepted else "reject"
        print(
            f"{cross_eval.name:<22} {cross_eval.n:6d} {cross_eval.correlation:10.4f} "
            f"{cross_eval.mean_overall:12.6f} {cross_eval.mean_top_decile:12.6f} "
            f"{cross_eval.mean_bottom_decile:12.6f} {cross_eval.edge:12.6f} "
            f"{cross_eval.top_decile_p_gt_zero:10.4f} {verdict:>8}"
        )
    elif not args.no_cross_asset:
        print(
            f"{'cross_asset_ratio_z':<22} {'—':>6} {'—':>10} {'—':>12} {'—':>12} "
            f"{'—':>12} {'—':>12} {'—':>10} {'skip':>8}"
        )

    print("=" * 112)
    accepted_any = any(ev.accepted for ev in results) or (cross_eval is not None and cross_eval.accepted)
    if accepted_any:
        print("At least one candidate PASSED filters (see table). Next: validate in isolation, then consider integration.")
    else:
        print("No candidates passed filters on this sample — reject or refine hypothesis (not thresholds blindly).")


if __name__ == "__main__":
    main()
