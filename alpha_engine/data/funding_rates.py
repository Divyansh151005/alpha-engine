"""USDT-M perpetual funding: fetch, align to bars, contrarian z-score signal."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Sequence

import ccxt
import numpy as np

from alpha_engine.data.models import OHLCV


def timeframe_ms(timeframe: str) -> int:
    """CCXT-style timeframe to milliseconds (e.g. 1m, 5m, 15m, 1h, 4h, 1d)."""
    tf = timeframe.strip().lower()
    if len(tf) < 2:
        raise ValueError(f"invalid timeframe: {timeframe!r}")
    num_s = tf[:-1]
    unit = tf[-1]
    if not num_s.isdigit():
        raise ValueError(f"invalid timeframe: {timeframe!r}")
    n = int(num_s)
    mult = {
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
    }.get(unit)
    if mult is None:
        raise ValueError(f"unsupported timeframe unit: {timeframe!r}")
    return n * mult


def spot_to_usdm_perp(spot_symbol: str) -> str:
    """Map CCXT spot symbol (e.g. BTC/USDT) to USDT-M perpetual for funding."""
    if ":USDT" in spot_symbol or spot_symbol.endswith(":USDT"):
        return spot_symbol
    parts = spot_symbol.split("/")
    if len(parts) != 2:
        return spot_symbol
    base, quote = parts[0], parts[1]
    if quote == "USDT":
        return f"{base}/{quote}:USDT"
    return f"{base}/{quote}:{quote}"


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


def fetch_funding_rate_history_usdm(
    spot_symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    page_limit: int = 1000,
) -> list[tuple[int, float]]:
    """
    Binance USDT-M funding (~8h cadence). Returns sorted (funding_time_ms, rate).
    Paginates until coverage passes ``end_ms`` or API exhaustion.
    """
    perp = spot_to_usdm_perp(spot_symbol)
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    events: list[tuple[int, float]] = []
    seen: set[int] = set()
    since = int(start_ms)
    safety = 0
    max_pages = 50
    while safety < max_pages:
        safety += 1
        batch = ex.fetch_funding_rate_history(perp, since=since, limit=page_limit)
        if not batch:
            break
        for row in batch:
            ts = int(row["timestamp"])
            if ts in seen:
                continue
            seen.add(ts)
            fr = row.get("fundingRate")
            if fr is None and isinstance(row.get("info"), dict):
                fr = row["info"].get("fundingRate")
            rate = float(fr) if fr is not None else float("nan")
            if math.isfinite(rate):
                events.append((ts, rate))
        last_ts = int(batch[-1]["timestamp"])
        if last_ts >= end_ms and len(batch) < page_limit:
            break
        next_since = last_ts + 1
        if next_since <= since:
            break
        since = next_since
        if last_ts >= end_ms:
            break
    events.sort(key=lambda x: x[0])
    return events


def align_funding_to_timestamps(
    candle_ts: np.ndarray,
    funding_events: list[tuple[int, float]],
) -> np.ndarray:
    """Last known funding at or before each candle close time (backward as-of)."""
    n = int(candle_ts.size)
    out = np.full(n, np.nan, dtype=np.float64)
    if not funding_events:
        return out
    j = 0
    for i in range(n):
        t = int(candle_ts[i])
        while j + 1 < len(funding_events) and funding_events[j + 1][0] <= t:
            j += 1
        if funding_events[j][0] <= t:
            out[i] = funding_events[j][1]
    return out


def signal_funding_contrarian_z(funding: np.ndarray, z_window: int) -> np.ndarray:
    """
    High signal = hypothesized long forward edge: minus rolling z-score of funding
    (very negative funding → positive score).
    """
    if z_window < 10:
        raise ValueError("z_window should be at least 10 for a stable z-score")
    mu = _rolling_mean(funding, z_window)
    sd = _rolling_std(funding, z_window)
    z = np.full_like(funding, np.nan, dtype=np.float64)
    mask = np.isfinite(mu) & np.isfinite(sd) & (sd > 1e-18) & np.isfinite(funding)
    z[mask] = (funding[mask] - mu[mask]) / sd[mask]
    return -z


def funding_contrarian_score_tail(window_values: Sequence[float] | np.ndarray) -> float:
    """
    Incremental helper: ``window_values`` length = z_window, last element = current rate.
    Returns the same -*z* as ``signal_funding_contrarian_z`` at the final bar.
    """
    xs = np.asarray(window_values, dtype=np.float64)
    if xs.size < 10:
        return float("nan")
    mu = float(np.mean(xs))
    sd = float(np.std(xs, ddof=1))
    cur = float(xs[-1])
    if not math.isfinite(sd) or sd <= 1e-18:
        return float("nan")
    return -(cur - mu) / sd


def attach_funding_to_ohlcv(
    candles: list[OHLCV],
    funding_events: list[tuple[int, float]],
) -> list[OHLCV]:
    """
    Align last-known funding at or before each candle timestamp (no lookahead).
    Returns candles sorted by (timestamp, symbol) with ``funding_rate`` set.
    """
    sorted_candles = sorted(candles, key=lambda c: (c.timestamp, c.symbol))
    if not sorted_candles:
        return []
    ts_arr = np.array([c.timestamp for c in sorted_candles], dtype=np.int64)
    aligned = align_funding_to_timestamps(ts_arr, funding_events)
    out: list[OHLCV] = []
    for c, fr in zip(sorted_candles, aligned, strict=True):
        v = float(fr)
        fr_opt = v if math.isfinite(v) else None
        out.append(replace(c, funding_rate=fr_opt))
    return out
