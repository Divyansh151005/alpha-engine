"""Alpha signal model emitted by strategy modules."""

from __future__ import annotations

from dataclasses import dataclass

_MIN_NORMALIZED = 0.0
_MAX_NORMALIZED = 1.0
_VALID_DIRECTIONS = {"long", "short", "flat"}


@dataclass(frozen=True)
class AlphaSignal:
    """Normalized strategy signal."""

    timestamp: int
    strategy_name: str
    direction: str
    strength: float
    confidence: float

    def __post_init__(self) -> None:
        if self.direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {_VALID_DIRECTIONS}, got {self.direction!r}"
            )
        self._validate_normalized("strength", self.strength)
        self._validate_normalized("confidence", self.confidence)

    @staticmethod
    def _validate_normalized(name: str, value: float) -> None:
        if not (_MIN_NORMALIZED <= value <= _MAX_NORMALIZED):
            raise ValueError(f"{name} must be within [0, 1], got {value!r}")
