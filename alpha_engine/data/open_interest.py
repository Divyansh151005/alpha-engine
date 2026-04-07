"""USDT-M open interest: fetch, align to bars, attach to OHLCV."""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Sequence

import ccxt
import numpy as np

from alpha_engine.data.funding_rates import spot_to_usdm_perp, timeframe_ms
from alpha_engine.data.models import OHLCV


def _period_to_ms(period: str) -> int:
    p = period.strip().lower()
    if p == "1m":
        return 60 * 1000
    if p == "5m":
        return 5 * 60 * 1000
    if p == "15m":
        return 15 * 60 * 1000
    if p == "30m":
        return 30 * 60 * 1000
    if p == "1h":
        return 60 * 60 * 1000
    if p == "2h":
        return 2 * 60 * 60 * 1000
    if p == "4h":
        return 4 * 60 * 60 * 1000
    if p == "6h":
        return 6 * 60 * 60 * 1000
    if p == "12h":
        return 12 * 60 * 60 * 1000
    if p == "1d":
        return 24 * 60 * 60 * 1000
    raise ValueError(f"Unsupported period: {period}")


# Binance openInterestHist rejects ``startTime`` older than roughly this window (ms).
_OI_HIST_RETENTION_MS = 30 * 24 * 60 * 60 * 1000

_OI_PERIODS_MS = (
    ("5m", 5 * 60_000),
    ("15m", 15 * 60_000),
    ("30m", 30 * 60_000),
    ("1h", 60 * 60_000),
    ("2h", 2 * 60 * 60_000),
    ("4h", 4 * 60 * 60_000),
    ("6h", 6 * 60 * 60_000),
    ("12h", 12 * 60 * 60_000),
    ("1d", 24 * 60 * 60_000),
)


def open_interest_period_for_candle_timeframe(candle_timeframe: str) -> str:
    """Map candle timeframe to the coarsest Binance OI history period that covers the bar length."""
    tf = candle_timeframe.strip().lower()
    allowed = {p for p, _ in _OI_PERIODS_MS}
    if tf in allowed:
        return tf
    ms = timeframe_ms(candle_timeframe)
    for label, m in _OI_PERIODS_MS:
        if ms <= m:
            return label
    return "1d"


def fetch_open_interest_history_usdm(
    spot_symbol: str,
    start_ms: int,
    end_ms: int,
    oi_timeframe: str,
    *,
    page_limit: int = 500,
) -> list[tuple[int, float]]:
    """
    Binance USDT-M open interest history (``fetch_open_interest_history``).
    Chunked requests with ``endTime`` to avoid invalid wide ``since`` ranges.
    Returns sorted ``(timestamp_ms, open_interest_amount)``, deduped by timestamp.
    """
    start_time = int(start_ms)
    end_time = int(end_ms)
    if start_time >= end_time:
        return []

    period = oi_timeframe.strip()
    period_ms = _period_to_ms(period)
    max_points = min(500, max(1, int(page_limit)))
    step_ms = period_ms * max_points

    perp = spot_to_usdm_perp(spot_symbol)
    ex = ccxt.binanceusdm({"enableRateLimit": True})

    try:
        now_ms = int(ex.milliseconds())
    except Exception:
        now_ms = int(time.time() * 1000)
    earliest_allowed = now_ms - _OI_HIST_RETENTION_MS
    if start_time < earliest_allowed:
        print(
            f"[OI FETCH] adjusting start_time {start_time} -> {earliest_allowed} "
            f"(Binance open interest history retention ~{_OI_HIST_RETENTION_MS // 86400000}d)"
        )
        start_time = earliest_allowed
    if start_time >= end_time:
        print(f"[OI FETCH DONE] total_points=0")
        return []

    print(f"[OI FETCH] start={start_time} end={end_time} period={period}")

    all_data: list[dict] = []
    cursor = start_time
    _safeguard = 0
    _max_chunk_iterations = 50_000

    while cursor < end_time:
        _safeguard += 1
        if _safeguard > _max_chunk_iterations:
            print("[OI FETCH ERROR] safeguard max_chunk_iterations exceeded — stopping")
            break

        chunk_end = min(cursor + step_ms, end_time)

        try:
            batch = ex.fetch_open_interest_history(
                perp,
                timeframe=period,
                since=cursor,
                limit=max_points,
                params={"endTime": chunk_end},
            )
        except Exception as e:
            print(f"[OI FETCH ERROR] {e} | cursor={cursor} → skipping")
            cursor = chunk_end
            continue

        if not batch:
            cursor = chunk_end
            continue

        all_data.extend(batch)

        last_ts = int(batch[-1]["timestamp"])
        if last_ts <= cursor:
            cursor = chunk_end
        else:
            cursor = last_ts + 1

    all_data.sort(key=lambda x: int(x["timestamp"]))
    by_ts: dict[int, dict] = {}
    for row in all_data:
        by_ts[int(row["timestamp"])] = row
    deduped_rows = [by_ts[k] for k in sorted(by_ts.keys())]

    print(f"[OI FETCH DONE] total_points={len(deduped_rows)}")

    events: list[tuple[int, float]] = []
    for row in deduped_rows:
        ts = int(row["timestamp"])
        oi = row.get("openInterestAmount")
        if oi is None:
            oi = row.get("baseVolume")
        val = float(oi) if oi is not None else float("nan")
        if math.isfinite(val) and val >= 0.0:
            events.append((ts, val))
    return events


def get_oi_coverage_range(
    oi_series: list[tuple[int, float]],
) -> tuple[int | None, int | None]:
    """
    Return ``(min_ts, max_ts)`` over available OI points (exchange sample times).

    ``oi_series``: ``(timestamp_ms, open_interest_amount)`` as returned by
    :func:`fetch_open_interest_history_usdm`.
    """
    if not oi_series:
        return None, None
    timestamps = [ts for ts, _ in oi_series]
    return min(timestamps), max(timestamps)


def align_open_interest_to_timestamps(
    candle_ts: np.ndarray,
    oi_events: list[tuple[int, float]],
) -> np.ndarray:
    """Last known open interest at or before each candle close time (backward as-of)."""
    n = int(candle_ts.size)
    out = np.full(n, np.nan, dtype=np.float64)
    if not oi_events:
        return out
    j = 0
    for i in range(n):
        t = int(candle_ts[i])
        while j + 1 < len(oi_events) and oi_events[j + 1][0] <= t:
            j += 1
        if oi_events[j][0] <= t:
            out[i] = oi_events[j][1]
    return out


def attach_open_interest_to_ohlcv(
    candles: list[OHLCV],
    oi_events: list[tuple[int, float]],
) -> list[OHLCV]:
    """Align last-known OI at or before each candle timestamp (no lookahead)."""
    sorted_candles = sorted(candles, key=lambda c: (c.timestamp, c.symbol))
    if not sorted_candles:
        return []
    ts_arr = np.array([c.timestamp for c in sorted_candles], dtype=np.int64)
    aligned = align_open_interest_to_timestamps(ts_arr, oi_events)
    out: list[OHLCV] = []
    for c, oi in zip(sorted_candles, aligned, strict=True):
        v = float(oi)
        oi_opt = v if math.isfinite(v) else None
        out.append(replace(c, open_interest=oi_opt))
    return out


def oi_change_series(oi_levels: Sequence[float] | np.ndarray) -> np.ndarray:
    """``(OI_t - OI_{t-1}) / OI_{t-1}``; NaN where undefined or prior OI <= 0."""
    xs = np.asarray(oi_levels, dtype=np.float64)
    n = xs.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out
    prev = xs[:-1]
    cur = xs[1:]
    mask = np.isfinite(prev) & np.isfinite(cur) & (prev > 0.0)
    ch = np.full(n - 1, np.nan, dtype=np.float64)
    ch[mask] = (cur[mask] - prev[mask]) / prev[mask]
    out[1:] = ch
    return out
