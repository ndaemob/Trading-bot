"""Tests for position sizing and risk management."""

from __future__ import annotations

import pytest

from src import config
from src.risk import calculate_position_size, check_allocation


# --------------------------------------------------------------------------- #
# The 2% max-risk-per-trade rule is never violated
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "portfolio,entry,stop",
    [
        (10_000, 100, 95),
        (10_000, 50, 48),
        (25_000, 200, 180),
        (5_000, 30, 29),
        (100_000, 1000, 900),
    ],
)
def test_risk_never_exceeds_two_percent(portfolio, entry, stop):
    """Worst-case loss to the stop is always <= 2% of the portfolio."""
    result = calculate_position_size(portfolio, entry, stop)
    max_allowed = portfolio * config.MAX_RISK_PER_TRADE
    # Allow a tiny epsilon for floating point.
    assert result.risk_amount <= max_allowed + 1e-6
    assert result.risk_pct <= config.MAX_RISK_PER_TRADE + 1e-9


def test_risk_amount_matches_shares_times_per_share_risk():
    result = calculate_position_size(10_000, 100, 90)
    assert result.risk_amount == pytest.approx(result.shares * (100 - 90))


# --------------------------------------------------------------------------- #
# The 20% allocation cap and its warning
# --------------------------------------------------------------------------- #
def test_allocation_never_exceeds_twenty_percent():
    """Even with a tiny stop (huge risk-based size), allocation is capped."""
    result = calculate_position_size(10_000, 100, 99.9)
    assert result.allocation_pct <= config.MAX_ALLOCATION_PER_STOCK + 1e-9


def test_allocation_cap_triggers_warning():
    """When the allocation cap binds, the user is warned."""
    result = calculate_position_size(10_000, 100, 99.9)
    assert any("allocation" in w.lower() for w in result.warnings)


def test_check_allocation_warns_above_limit():
    """A position worth >20% of the portfolio produces a warning."""
    warnings = check_allocation(portfolio_value=10_000, position_value=3_000)
    assert warnings
    assert any("exceeds" in w.lower() for w in warnings)


def test_check_allocation_silent_within_limit():
    """A position within 20% produces no warnings."""
    warnings = check_allocation(portfolio_value=10_000, position_value=1_500)
    assert warnings == []


# --------------------------------------------------------------------------- #
# Leverage is structurally impossible
# --------------------------------------------------------------------------- #
def test_never_uses_leverage():
    result = calculate_position_size(10_000, 100, 95)
    assert result.uses_leverage is False
    assert result.position_value <= 10_000 + 1e-6


def test_shares_are_whole_and_non_negative():
    result = calculate_position_size(10_000, 100, 95)
    assert isinstance(result.shares, int)
    assert result.shares >= 0


# --------------------------------------------------------------------------- #
# Invalid inputs are rejected clearly
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "portfolio,entry,stop",
    [
        (0, 100, 95),  # non-positive portfolio
        (10_000, 0, 95),  # non-positive entry
        (10_000, 100, 0),  # non-positive stop
        (10_000, 100, 105),  # stop above entry (would be negative risk)
        (10_000, 100, 100),  # stop equal to entry (zero risk -> div by zero)
    ],
)
def test_invalid_inputs_raise(portfolio, entry, stop):
    with pytest.raises(ValueError):
        calculate_position_size(portfolio, entry, stop)


def test_wide_stop_small_capital_yields_zero_with_warning():
    """If a single share would breach the rules, size is 0 with a warning."""
    # Entry far above what 2% of a small portfolio can cover.
    result = calculate_position_size(100, 1000, 500)
    assert result.shares == 0
    assert any("0 shares" in w for w in result.warnings)
