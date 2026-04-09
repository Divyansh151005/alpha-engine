"""Funding contrarian with price confirmation: trade only when funding is extreme and price agrees.

**Option B (funding + price divergence):** crowded shorts should coincide with weak recent
returns; crowded longs with strong recent returns — otherwise no trade.

Keeps :class:`FundingContrarianAlphaStrategy` as the primary signal; adds an independent
confirmation from recent total return (crowded-short legs → price should be weak;
crowded-long legs → price should be strong). If confirmation fails, no signal is emitted.

For an explicit **two-alpha** setup (funding leg + confirmation leg), pass a shared
``FundingContrarianAlphaStrategy`` into ``funding=`` and add
:class:`PriceDivergenceContrarianConfirmAlphaStrategy` as the second strategy; :class:`AlphaEngine`
then keeps funding trades only when the confirmation alpha also fires.

Optional band-pass on ``abs(rolling_return_20)``: when ``regime_abs_return_20_min`` is set,
require finite ``rolling_return_20``, ``abs(rr20) >= min``, and (if ``regime_abs_return_20_max``
is set) ``abs(rr20) <= max`` before funding or confirmation; otherwise ``None``.

Optional **funding-structure** gate (before price confirm): rolling mean/std over recent funding,
``z = (funding - mean) / std``. **Static** mode: ``abs(z) > z_min`` and ``std > std_min``. **Adaptive**
mode: keep histories of ``abs(z)`` and ``std`` (last N bars), require ``abs(z)`` above the
``z_percentile`` of past ``abs(z)`` and ``std`` above the ``std_percentile`` of past ``std``
(no manual scale tuning). Optional: last ``k`` funding updates share the same strict sign.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Mapping

import numpy as np

from alpha_engine.alpha.alpha_signal import AlphaSignal
from alpha_engine.alpha.base_alpha import BaseAlphaStrategy
from alpha_engine.alpha.funding_asset_config import (
    FundingConfirmAssetConfig,
    resolve_funding_confirm_asset_params,
)
from alpha_engine.alpha.funding_contrarian_alpha import FundingContrarianAlphaStrategy
from alpha_engine.state_engine.market_state_engine import MarketStateEngine
from alpha_engine.state_engine.state_models import MarketStateVector


class FundingPriceConfirmContrarianAlphaStrategy(BaseAlphaStrategy):
    """Strong funding tilt + aligned 10-bar return; delegates funding math to FundingContrarian."""

    def __init__(
        self,
        z_window: int = 168,
        *,
        funding: FundingContrarianAlphaStrategy | None = None,
        min_funding_score: float = 0.12,
        asset_config: Mapping[str, FundingConfirmAssetConfig] | None = None,
        regime_abs_return_20_min: float | None = None,
        regime_abs_return_20_max: float | None = None,
        funding_regime_z_abs_min: float | None = None,
        funding_regime_std_min: float | None = None,
        funding_regime_adaptive: bool = False,
        funding_regime_adaptive_history: int = 400,
        funding_regime_adaptive_z_percentile: float = 75.0,
        funding_regime_adaptive_std_percentile: float = 55.0,
        funding_regime_adaptive_min_history: int = 80,
        funding_regime_window: int | None = None,
        funding_regime_same_sign_bars: int = 0,
        ret_key: str = "rolling_return_10",
        ret_epsilon: float = 0.0,
        signal_debug_flow: bool = False,
    ) -> None:
        if min_funding_score <= 0.0:
            raise ValueError("min_funding_score must be > 0")
        if regime_abs_return_20_min is not None and regime_abs_return_20_min < 0.0:
            raise ValueError("regime_abs_return_20_min must be >= 0")
        if regime_abs_return_20_max is not None and regime_abs_return_20_min is None:
            raise ValueError(
                "regime_abs_return_20_max requires regime_abs_return_20_min to be set"
            )
        if regime_abs_return_20_max is not None:
            if regime_abs_return_20_max <= float(regime_abs_return_20_min):
                raise ValueError(
                    "regime_abs_return_20_max must be > regime_abs_return_20_min"
                )
        fr_z = funding_regime_z_abs_min
        fr_s = funding_regime_std_min
        if funding_regime_adaptive:
            if fr_z is not None or fr_s is not None:
                raise ValueError(
                    "funding_regime_adaptive=True excludes funding_regime_z_abs_min / "
                    "funding_regime_std_min (use percentile thresholds only)"
                )
        elif (fr_z is None) ^ (fr_s is None):
            raise ValueError(
                "funding_regime_z_abs_min and funding_regime_std_min must both be set "
                "or both omitted (unless funding_regime_adaptive=True)"
            )
        if funding_regime_same_sign_bars < 0:
            raise ValueError("funding_regime_same_sign_bars must be >= 0")
        hist_n = int(funding_regime_adaptive_history)
        min_hist = int(funding_regime_adaptive_min_history)
        z_pct = float(funding_regime_adaptive_z_percentile)
        std_pct = float(funding_regime_adaptive_std_percentile)
        if funding_regime_adaptive:
            if hist_n < 50:
                raise ValueError("funding_regime_adaptive_history must be >= 50")
            if min_hist < 10 or min_hist > hist_n:
                raise ValueError(
                    "funding_regime_adaptive_min_history must be in [10, history]"
                )
            if not (0.0 < z_pct < 100.0) or not (0.0 < std_pct < 100.0):
                raise ValueError("adaptive percentiles must be in (0, 100)")
        self._fr_window = int(funding_regime_window) if funding_regime_window is not None else int(z_window)
        if self._fr_window < 10:
            raise ValueError("funding_regime_window / z_window must be >= 10")
        self._owns_funding = funding is None
        self._funding = (
            funding if funding is not None else FundingContrarianAlphaStrategy(z_window=z_window)
        )
        self._default_min_funding_score = float(min_funding_score)
        self._asset_config = asset_config
        self._regime_abs_return_20_min = (
            float(regime_abs_return_20_min)
            if regime_abs_return_20_min is not None
            else None
        )
        self._regime_abs_return_20_max = (
            float(regime_abs_return_20_max)
            if regime_abs_return_20_max is not None
            else None
        )
        self._ret_key = ret_key
        self._ret_epsilon = float(ret_epsilon)
        self._funding_regime_z_abs_min = (
            float(fr_z) if fr_z is not None else None
        )
        self._funding_regime_std_min = (
            float(fr_s) if fr_s is not None else None
        )
        self._funding_regime_same_sign_bars = int(funding_regime_same_sign_bars)
        self._funding_regime_buf: deque[float] = deque(maxlen=self._fr_window + 50)
        self._funding_regime_adaptive = bool(funding_regime_adaptive)
        self._fr_adaptive_min_hist = min_hist
        self._fr_adaptive_z_pct = z_pct
        self._fr_adaptive_std_pct = std_pct
        self._z_abs_hist: deque[float] = deque(maxlen=hist_n)
        self._std_hist: deque[float] = deque(maxlen=hist_n)
        self._fr_adaptive_ok: bool = True
        self.signal_debug_flow = bool(signal_debug_flow)
        self._debug_stat_keys: tuple[str, ...] = (
            "total_bars",
            "regime_pass",
            "funding_pass",
            "price_confirm_pass",
            "final_signals",
        )
        self.debug_stats: dict[str, int] = {k: 0 for k in self._debug_stat_keys}

    def reset_signal_debug_stats(self) -> None:
        """Reset :attr:`debug_stats` (call before each backtest / symbol run)."""
        self.debug_stats = {k: 0 for k in self._debug_stat_keys}

    def print_signal_debug_stats(self, symbol: str = "") -> None:
        """Print accumulated stats when :attr:`signal_debug_flow` is True."""
        if not self.signal_debug_flow:
            return
        label = symbol if symbol else "(unknown symbol)"
        print(f"[DEBUG SIGNAL FLOW] {label}")
        for k in self._debug_stat_keys:
            print(f"  {k}: {self.debug_stats[k]}")

    def note_funding(self, rate: float) -> None:
        self._funding.note_funding(rate)
        if math.isfinite(rate):
            self._funding_regime_buf.append(float(rate))

    def note_funding_batch(self, rates: list[float]) -> None:
        self._funding.note_funding_batch(rates)
        for r in rates:
            if math.isfinite(r):
                self._funding_regime_buf.append(float(r))

    def on_market_state(self, state: MarketStateVector) -> None:
        if self._owns_funding:
            self._funding.on_market_state(state)
        fr = float(getattr(state, "funding_rate", float("nan")))
        if math.isfinite(fr):
            self._funding_regime_buf.append(fr)
        self._fr_adaptive_ok = True
        if not self._funding_regime_adaptive:
            return
        w = self._fr_window
        if len(self._funding_regime_buf) < w:
            self._fr_adaptive_ok = False
            return
        zs = self._compute_funding_z_std()
        if zs is None:
            self._fr_adaptive_ok = False
            return
        _z, std, z_abs = zs
        if len(self._z_abs_hist) >= self._fr_adaptive_min_hist:
            z_thr = float(
                np.percentile(
                    np.asarray(self._z_abs_hist, dtype=np.float64),
                    self._fr_adaptive_z_pct,
                )
            )
            std_thr = float(
                np.percentile(
                    np.asarray(self._std_hist, dtype=np.float64),
                    self._fr_adaptive_std_pct,
                )
            )
            self._fr_adaptive_ok = (z_abs >= z_thr) and (std >= std_thr)
        self._z_abs_hist.append(z_abs)
        self._std_hist.append(std)

    def _passes_rr20_regime(self, state: MarketStateVector) -> bool:
        """Band-pass on ``abs(rolling_return_20)`` when configured."""
        if self._regime_abs_return_20_min is None:
            return True
        rr20 = float(getattr(state, "rolling_return_20", float("nan")))
        if not math.isfinite(rr20):
            return False
        abs_rr20 = abs(rr20)
        if abs_rr20 < self._regime_abs_return_20_min:
            return False
        if (
            self._regime_abs_return_20_max is not None
            and abs_rr20 > self._regime_abs_return_20_max
        ):
            return False
        return True

    def _funding_leg_signal(
        self, state: MarketStateVector, min_funding_score: float
    ) -> AlphaSignal | None:
        """Contrarian funding + min score + optional funding-structure gates."""
        inner = self._funding.generate_signal(state)
        if inner is None:
            return None
        if abs(inner.score) < min_funding_score:
            return None

        funding_structure_on = self._funding_regime_adaptive or (
            self._funding_regime_z_abs_min is not None
        )
        if funding_structure_on:
            if self._funding_regime_adaptive:
                if not self._fr_adaptive_ok:
                    return None
            else:
                if not self._funding_structure_allows_trade_static():
                    return None
            if not self._funding_same_sign_persistence_ok():
                return None
        return inner

    def _funding_valid_signal_before_price_confirm(
        self, state: MarketStateVector
    ) -> AlphaSignal | None:
        sym = getattr(state, "symbol", None)
        enabled, min_funding_score = resolve_funding_confirm_asset_params(
            sym if isinstance(sym, str) else None,
            self._asset_config,
            default_min_funding_score=self._default_min_funding_score,
        )
        if not enabled:
            return None
        if not self._passes_rr20_regime(state):
            return None
        return self._funding_leg_signal(state, min_funding_score)

    def _price_divergence_aligns(self, inner: AlphaSignal, state: MarketStateVector) -> bool:
        ret = float(getattr(state, self._ret_key, float("nan")))
        if not math.isfinite(ret):
            return False
        if inner.score > 0.0:
            return ret <= self._ret_epsilon
        return ret >= -self._ret_epsilon

    def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
        dbg = self.signal_debug_flow
        if dbg:
            self.debug_stats["total_bars"] += 1

        if not bool(getattr(state, "high_volatility", True)):
            return None

        sym = getattr(state, "symbol", None)
        enabled, min_funding_score = resolve_funding_confirm_asset_params(
            sym if isinstance(sym, str) else None,
            self._asset_config,
            default_min_funding_score=self._default_min_funding_score,
        )
        if not enabled:
            return None

        # Regime: abs(rr20) band (documentation labels this "regime filter").
        if not self._passes_rr20_regime(state):
            return None
        if dbg:
            self.debug_stats["regime_pass"] += 1

        # Funding: extreme contrarian + min |score| + funding-structure gates.
        inner = self._funding_leg_signal(state, min_funding_score)
        if inner is None:
            return None
        if dbg:
            self.debug_stats["funding_pass"] += 1

        if not self._price_divergence_aligns(inner, state):
            return None
        if dbg:
            self.debug_stats["price_confirm_pass"] += 1

        ret = float(getattr(state, self._ret_key, float("nan")))
        align = min(1.0, abs(ret) * 25.0)
        conf = min(1.0, inner.confidence * (0.5 + 0.5 * align))
        if dbg:
            self.debug_stats["final_signals"] += 1
        return AlphaSignal(
            timestamp=state.timestamp,
            strategy_name=self.__class__.__name__,
            score=inner.score,
            confidence=conf,
        )

    def _compute_funding_z_std(self) -> tuple[float, float, float] | None:
        """Current-window z-score, std, and abs(z); None if undefined."""
        w = self._fr_window
        if len(self._funding_regime_buf) < w:
            return None
        tail = np.asarray(
            list(self._funding_regime_buf)[-w:],
            dtype=np.float64,
        )
        cur = float(tail[-1])
        mean = float(np.mean(tail))
        std = float(np.std(tail, ddof=1))
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 1e-18:
            return None
        z = (cur - mean) / std
        if not math.isfinite(z):
            return None
        return (z, std, abs(z))

    def _funding_structure_allows_trade_static(self) -> bool:
        """Static z/std floors on funding (no persistence; handled separately)."""
        assert self._funding_regime_z_abs_min is not None
        assert self._funding_regime_std_min is not None
        zs = self._compute_funding_z_std()
        if zs is None:
            return False
        _z, std, z_abs = zs
        if std <= self._funding_regime_std_min:
            return False
        if z_abs <= self._funding_regime_z_abs_min:
            return False
        return True

    def _funding_same_sign_persistence_ok(self) -> bool:
        k = self._funding_regime_same_sign_bars
        if k <= 0:
            return True
        if len(self._funding_regime_buf) < k:
            return False
        recent = list(self._funding_regime_buf)[-k:]
        if any(r == 0.0 or not math.isfinite(r) for r in recent):
            return False
        pos = all(r > 0.0 for r in recent)
        neg = all(r < 0.0 for r in recent)
        return pos or neg
