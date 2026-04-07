"""Per-asset parameters for funding + price-confirmation strategy (not the core alpha math)."""

from __future__ import annotations

from typing import Mapping, TypedDict


class FundingConfirmAssetConfig(TypedDict, total=False):
    """Optional overrides per spot symbol (e.g. ``BTC/USDT``)."""

    enabled: bool
    min_funding_score: float


DEFAULT_FUNDING_CONFIRM_ASSET_CONFIG: dict[str, FundingConfirmAssetConfig] = {
    "BTC/USDT": {"min_funding_score": 0.12, "enabled": True},
    "ETH/USDT": {"min_funding_score": 0.08, "enabled": True},
    "SOL/USDT": {"enabled": False},
}


def resolve_funding_confirm_asset_params(
    symbol: str | None,
    asset_config: Mapping[str, FundingConfirmAssetConfig] | None,
    *,
    default_min_funding_score: float,
) -> tuple[bool, float]:
    """Return (enabled, min_funding_score) for this bar's symbol."""
    if not symbol or asset_config is None:
        return (True, float(default_min_funding_score))
    row = asset_config.get(symbol)
    if row is None:
        return (True, float(default_min_funding_score))
    enabled = row.get("enabled", True)
    min_score = float(row.get("min_funding_score", default_min_funding_score))
    return (bool(enabled), min_score)
