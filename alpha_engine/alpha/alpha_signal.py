"""Alpha signal model emitted by strategy modules."""

from __future__ import annotations

from dataclasses import dataclass

_MIN_NORMALIZED = 0.0
_MAX_NORMALIZED = 1.0
_SCORE_MIN = -1.0
_SCORE_MAX = 1.0
_VALID_DIRECTIONS = {"long", "short", "flat"}


def _score_to_direction(score: float) -> str:
    """Derive direction from continuous score."""
    if score > 0:
        return "long"
    if score < 0:
        return "short"
    return "flat"


@dataclass(frozen=True)
class AlphaSignal:
    """Score-based strategy signal. Positive score = long weight, negative = short weight."""

    timestamp: int
    strategy_name: str
    score: float  # Continuous signal in [-1, 1]
    confidence: float  # Signal reliability [0, 1]

    def __post_init__(self) -> None:
        if not (_SCORE_MIN <= self.score <= _SCORE_MAX):
            raise ValueError(
                f"score must be within [{_SCORE_MIN}, {_SCORE_MAX}], got {self.score!r}"
            )
        if not (_MIN_NORMALIZED <= self.confidence <= _MAX_NORMALIZED):
            raise ValueError(
                f"confidence must be within [0, 1], got {self.confidence!r}"
            )

    @property
    def direction(self) -> str:
        """Derived from score for backward compatibility."""
        return _score_to_direction(self.score)

    @property
    def strength(self) -> float:
        """Magnitude of signal [0, 1]."""
        return abs(self.score)
