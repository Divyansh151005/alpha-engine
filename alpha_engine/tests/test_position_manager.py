"""Tests for persistent discrete position manager."""

from __future__ import annotations

from alpha_engine.position import PositionManager


def test_entry_hold_exit_rules() -> None:
    manager = PositionManager(
        entry_threshold=0.5,
        exit_threshold=0.1,
        strong_flip_threshold=0.7,
        hold_zone=0.3,
        cost_buffer=0.1,
    )

    # Below entry threshold: remain flat
    assert manager.update_position("s1", 0.5) == 0
    # Strong entry
    assert manager.update_position("s1", 0.8) == 1
    # Hold while still meaningful and same-side
    assert manager.update_position("s1", 0.4) == 1
    # Do not exit on weak neutral signal; requires actual reversal.
    assert manager.update_position("s1", 0.1) == 1
    assert manager.update_position("s1", -0.61) == 0


def test_strong_opposite_signal_flips_position() -> None:
    manager = PositionManager(
        entry_threshold=0.5,
        exit_threshold=0.1,
        strong_flip_threshold=0.7,
        hold_zone=0.3,
        cost_buffer=0.1,
    )
    assert manager.update_position("s1", 0.8) == 1
    # Stability guard forbids instant long->short flip: must pass through flat.
    assert manager.update_position("s1", -0.8) == 0
    assert manager.update_position("s1", -0.8) == -1


def test_min_holding_period_prevents_early_changes() -> None:
    manager = PositionManager(
        entry_threshold=0.5,
        exit_threshold=0.1,
        strong_flip_threshold=0.7,
        min_hold_bars=5,
        hold_zone=0.3,
        cost_buffer=0.1,
    )
    assert manager.update_position("s1", 0.8, current_time=10) == 1
    # Before 5 bars elapsed, no position change is allowed.
    assert manager.update_position("s1", 0.0, current_time=13) == 1
    # At the holding boundary, exit is allowed.
    assert manager.update_position("s1", -0.61, current_time=15) == 0
