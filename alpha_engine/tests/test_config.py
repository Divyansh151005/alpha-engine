"""Tests for settings loading."""

from __future__ import annotations

from alpha_engine.core.config import get_settings


def test_settings_load_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("EXCHANGE_NAME", "kraken")
    monkeypatch.setenv("DEFAULT_SYMBOL", "ETHUSD")
    monkeypatch.setenv("BASE_RISK_PER_TRADE", "0.02")
    monkeypatch.setenv("TARGET_ANNUAL_VOL", "0.25")
    monkeypatch.setenv("MAX_DRAWDOWN", "0.10")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.exchange_name == "kraken"
    assert settings.default_symbol == "ETHUSD"
    assert settings.base_risk_per_trade == 0.02
    assert settings.target_annual_vol == 0.25
    assert settings.max_drawdown == 0.10

