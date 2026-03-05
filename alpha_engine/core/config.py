"""Application configuration management."""

from __future__ import annotations

from functools import lru_cache

from pydantic.v1 import BaseSettings, Field


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    exchange_name: str = Field(default="binance", env="EXCHANGE_NAME")
    default_symbol: str = Field(default="BTCUSDT", env="DEFAULT_SYMBOL")
    base_risk_per_trade: float = Field(default=0.01, env="BASE_RISK_PER_TRADE")
    target_annual_vol: float = Field(default=0.2, env="TARGET_ANNUAL_VOL")
    max_drawdown: float = Field(default=0.15, env="MAX_DRAWDOWN")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()

