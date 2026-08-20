"""Position sizing and risk management.

Implements conservative, rules-based position sizing:

* Risk at most **2%** of the portfolio on any single trade.
* Allocate at most **20%** of the portfolio to any single stock.
* **Never** use leverage — a position can never exceed available capital.

The functions only *describe* a suggested size; they never place an order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import config


@dataclass
class PositionSize:
    """Result of a position-sizing calculation.

    Attributes:
        ticker: Symbol the sizing was computed for.
        shares: Whole number of shares suggested (never fractional, never < 0).
        entry_price: Price the sizing assumed.
        stop_loss: Stop-loss price used to derive per-share risk.
        position_value: ``shares * entry_price``.
        risk_amount: Worst-case loss to the stop = ``shares * risk_per_share``.
        risk_pct: ``risk_amount`` as a fraction of the portfolio.
        allocation_pct: ``position_value`` as a fraction of the portfolio.
        uses_leverage: Always ``False`` — the bot forbids leverage.
        warnings: Any limit breaches or notes for the user.
    """

    ticker: str
    shares: int
    entry_price: float
    stop_loss: float
    position_value: float
    risk_amount: float
    risk_pct: float
    allocation_pct: float
    uses_leverage: bool = False
    warnings: list[str] = field(default_factory=list)


def check_allocation(
    portfolio_value: float,
    position_value: float,
    max_allocation: float = config.MAX_ALLOCATION_PER_STOCK,
) -> list[str]:
    """Return warnings if ``position_value`` breaches allocation limits.

    Args:
        portfolio_value: Total portfolio value.
        position_value: Proposed dollar value of the position.
        max_allocation: Maximum fraction allowed in one stock (default 20%).

    Returns:
        A list of warning strings (empty if the position is within limits).
    """
    warnings: list[str] = []
    if portfolio_value <= 0:
        return ["Portfolio value must be positive."]

    allocation = position_value / portfolio_value
    if allocation > max_allocation + 1e-9:
        warnings.append(
            f"Allocation {allocation:.1%} exceeds the " f"{max_allocation:.0%} per-stock limit."
        )
    if position_value > portfolio_value + 1e-9:
        warnings.append("Position value exceeds available capital — leverage is not allowed.")
    return warnings


def calculate_position_size(
    portfolio_value: float,
    entry_price: float,
    stop_loss: float,
    ticker: str = "",
    max_risk_per_trade: float = config.MAX_RISK_PER_TRADE,
    max_allocation: float = config.MAX_ALLOCATION_PER_STOCK,
) -> PositionSize:
    """Calculate a risk- and allocation-constrained position size.

    The number of shares is the **smaller** of:

    * the risk-based size: ``(portfolio * 2%) / (entry - stop)``, and
    * the allocation cap:   ``(portfolio * 20%) / entry``.

    Whichever constraint binds, the result respects *both* the 2% risk rule and
    the 20% allocation rule, and never uses leverage.

    Args:
        portfolio_value: Total portfolio value (must be > 0).
        entry_price: Intended entry price (must be > 0).
        stop_loss: Protective stop price (must be > 0 and < ``entry_price``).
        ticker: Optional symbol for reporting.
        max_risk_per_trade: Max fraction of portfolio to risk (default 2%).
        max_allocation: Max fraction of portfolio per stock (default 20%).

    Returns:
        A :class:`PositionSize` describing the suggested trade.

    Raises:
        ValueError: If inputs are non-positive or the stop is not below entry.
    """
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive.")
    if entry_price <= 0:
        raise ValueError("entry_price must be positive.")
    if stop_loss <= 0:
        raise ValueError("stop_loss must be positive.")
    if stop_loss >= entry_price:
        raise ValueError("stop_loss must be below entry_price for a long position.")
    if math.isnan(entry_price) or math.isnan(stop_loss):
        raise ValueError("entry_price and stop_loss must be valid numbers.")

    warnings: list[str] = []

    risk_per_share = entry_price - stop_loss
    risk_budget = portfolio_value * max_risk_per_trade
    shares_by_risk = risk_budget / risk_per_share

    allocation_budget = portfolio_value * max_allocation
    shares_by_allocation = allocation_budget / entry_price

    # Honour the tighter of the two constraints, rounded *down* so we never
    # exceed either limit, and never go negative.
    shares = int(max(0, math.floor(min(shares_by_risk, shares_by_allocation))))

    if shares_by_allocation < shares_by_risk:
        warnings.append(
            f"Risk-based size reduced to respect the {max_allocation:.0%} " f"allocation cap."
        )
    if shares == 0:
        warnings.append(
            "Calculated size is 0 shares — the stop is too wide or capital too "
            "small for these limits."
        )

    position_value = shares * entry_price
    risk_amount = shares * risk_per_share
    risk_pct = risk_amount / portfolio_value
    allocation_pct = position_value / portfolio_value

    warnings.extend(check_allocation(portfolio_value, position_value, max_allocation))

    return PositionSize(
        ticker=ticker.upper() if ticker else "",
        shares=shares,
        entry_price=round(entry_price, 2),
        stop_loss=round(stop_loss, 2),
        position_value=round(position_value, 2),
        risk_amount=round(risk_amount, 2),
        risk_pct=risk_pct,
        allocation_pct=allocation_pct,
        uses_leverage=False,  # Leverage is structurally impossible here.
        warnings=warnings,
    )
