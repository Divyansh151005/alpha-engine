<div align="center">

# Alpha Engine

**An event-driven, regime-adaptive, multi-alpha research and trading engine for crypto markets.**

[![tests](https://github.com/Divyansh151005/alpha-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/Divyansh151005/alpha-engine/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-135%20passing-brightgreen)
![Exchange](https://img.shields.io/badge/exchange-Binance-F0B90B?logo=binance&logoColor=white)
![Storage](https://img.shields.io/badge/storage-PostgreSQL-4169E1?logo=postgresql&logoColor=white)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

---

Alpha Engine turns raw Binance candles into risk-managed portfolio targets through a chain of small, independent engines that talk only over an in-process event bus:

> **candles** &rarr; **features** &rarr; **market regime + normalized state** &rarr; **alpha signals** &rarr; **capital allocation** &rarr; **volatility targeting** &rarr; **drawdown control** &rarr; **final positions**

The same feature, state, and alpha code is reused by live data ingestion, a cost-aware backtester, a paper-trading loop, and a set of research and validation scripts, so research results come from the exact logic that runs live.

## Table of contents

- [Key features](#key-features)
- [Alpha strategy catalog](#alpha-strategy-catalog)

## Key features

| | Feature | What it does |
|---|---|---|
| :satellite: | **Market data engine** | Paged Binance OHLCV history via `ccxt`, a live kline WebSocket stream, and PostgreSQL persistence. Also pulls USDT-M perpetual **funding rates** and **open interest** and aligns them to bars. |
| :bar_chart: | **Feature engine** | 20+ per-bar features: realized vol, ATR, trend strength, multi-horizon returns, return z-score, volume spike, wick ratios, Bollinger distance, range compression/expansion, momentum acceleration. |
| :compass: | **Regime detection** | Classifies every bar as `TREND`, `MEAN_REVERSION`, `HIGH_VOL`, or `LOW_VOL` and reweights strategy families accordingly. |
| :brain: | **Multi-alpha engine** | Eight independent strategies emit scores in `[-1, 1]`. They are filtered, AND-gated where paired, and combined into an EMA-smoothed net signal that is never allowed to flip the raw direction. |
| :scales: | **Allocation and sizing** | Regime-weighted capital allocation followed by volatility targeting (`target_vol / realized_vol`, floored and bounded). |
| :shield: | **Risk engine** | Deterministic drawdown ladder (1.0x / 0.8x / 0.5x / 0.2x) with a hard **kill switch** at 20% drawdown. |
| :hourglass: | **Overtrading controls** | Volatility-aware signal threshold, per-strategy cooldown, trade-frequency cap, minimum holding period, and a discrete position manager. |
| :test_tube: | **Cost-aware backtester** | Fees (4 bps) and slippage (2 bps), a top-decile signal-rank gate, a regime exposure band, an optional HIGH_VOL execution filter, and 20+ reported metrics. |
| :microscope: | **Research tooling** | Scripts for signal research with p-values, multi-window and multi-symbol robustness validation, filter A/B comparison, and paper trading. |

## Alpha strategy catalog

All strategies live in [`alpha_engine/alpha/`](alpha_engine/alpha) and implement `BaseAlphaStrategy`. Positive score means long, negative means short.

| Strategy | Family | Signal logic |
|---|---|---|
| `TrendAlphaStrategy` | Trend | Weighted multi-timeframe momentum (`0.5*r5 + 0.3*r10 + 0.2*r20`), volatility-normalized, then averaged with EMA-crossover strength and the normalized trend state, clipped to `[-1, 1]`. |
| `MeanReversionAlphaStrategy` | Mean reversion | Fades stretched moves: return z-score above `+2` goes short, below `-2` goes long. |
| `BreakoutAlphaStrategy` | Breakout | Fires in the direction of the bar when `abs(latest return)` exceeds 2x the 20-bar return volatility. |
| `MomentumPersistenceAlphaStrategy` | Momentum | Short-only: three consecutive negative bar returns give a full short. |
| `VolumeSpikeAlphaStrategy` | Volume | `volume / SMA20(volume)` above 2.0 is long, below 0.5 is short, otherwise flat. |
| `FundingContrarianAlphaStrategy` | Funding | Negated rolling z-score of perpetual funding (default 168-bar window): crowded longs lead to a short tilt, crowded shorts to a long tilt. |
| `FundingPriceConfirmContrarianAlphaStrategy` | Funding + price | Requires an extreme funding score **and** an agreeing 10-bar return, with optional `abs(r20)` regime band, adaptive percentile funding-structure gates, and sign-persistence checks. Per-asset overrides live in `funding_asset_config.py`. |
| `PriceDivergenceContrarianConfirmAlphaStrategy` | Confirmation | Second leg of the funding thesis. When registered with the funding contrarian, `AlphaEngine` only lets the pair trade if **both** agree. |
