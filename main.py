"""Alpha Engine entry point."""

from __future__ import annotations

from alpha_engine.core.config import get_settings
from alpha_engine.core.logger import get_logger


def main() -> None:
    settings = get_settings()
    logger = get_logger(__name__)
    logger.info(
        "loaded settings exchange=%s symbol=%s",
        settings.exchange_name,
        settings.default_symbol,
    )
    print("Alpha Engine Initialized")


if __name__ == "__main__":
    main()

