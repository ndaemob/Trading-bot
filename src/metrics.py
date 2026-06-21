"""Professional performance metrics for equity curves and trades.

Pure, dependency-light functions used by the back- and portfolio-testers to
report risk-adjusted performance the way a real analyst would: CAGR, annualised
volatility, Sharpe, Sortino, Calmar, max drawdown, profit factor and win rate.

Every function is defensive: degenerate inputs (empty, single-point, or
zero-variance series) return ``0.0`` rather than raising or producing ``inf``/
``nan``, so callers never have to special-case them.
"""

from __future__ import annotations

import math

import pandas as pd

from . import config


def daily_returns(equity: pd.Series) -> pd.Series:
    """Simple period-over-period returns of an equity curve (NaNs dropped)."""
    if equity is None or len(equity) < 2:
        return pd.Series(dtype="float64")
    return equity.pct_change().dropna()


def total_return(equity: pd.Series) -> float:
    """Overall fractional return from first to last equity point."""
    if equity is None or len(equity) < 2:
        return 0.0
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    if start <= 0:
        return 0.0
    return end / start - 1.0


def cagr(equity: pd.Series, periods_per_year: int = config.TRADING_DAYS_PER_YEAR) -> float:
    """Compound annual growth rate implied by the equity curve."""
    if equity is None or len(equity) < 2:
        return 0.0
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    if start <= 0:
        return 0.0
    years = (len(equity) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    if end <= 0:
        return -1.0
    return (end / start) ** (1.0 / years) - 1.0


def annualized_volatility(
    returns: pd.Series, periods_per_year: int = config.TRADING_DAYS_PER_YEAR
) -> float:
    """Annualised standard deviation of periodic returns."""
    if returns is None or len(returns) < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    if math.isnan(std):
        return 0.0
    return std * math.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = config.RISK_FREE_RATE,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sharpe ratio (excess return per unit of total volatility)."""
    if returns is None or len(returns) < 2:
        return 0.0
    excess = returns - risk_free / periods_per_year
    std = float(excess.std(ddof=1))
    if std == 0 or math.isnan(std):
        return 0.0
    return float(excess.mean()) / std * math.sqrt(periods_per_year)


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = config.RISK_FREE_RATE,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sortino ratio (excess return per unit of *downside* volatility)."""
    if returns is None or len(returns) < 2:
        return 0.0
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    dstd = float(downside.std(ddof=1))
    if dstd == 0 or math.isnan(dstd):
        return 0.0
    return float(excess.mean()) / dstd * math.sqrt(periods_per_year)


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline as a positive fraction (``0.25`` == -25%)."""
    if equity is None or len(equity) == 0:
        return 0.0
    running_max = equity.cummax()
    drawdowns = (equity - running_max) / running_max
    worst = float(drawdowns.min())
    return abs(worst) if worst < 0 else 0.0


def calmar_ratio(equity: pd.Series, periods_per_year: int = config.TRADING_DAYS_PER_YEAR) -> float:
    """CAGR divided by max drawdown — return per unit of worst-case pain."""
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return cagr(equity, periods_per_year) / mdd


def profit_factor(trade_profits: list[float]) -> float:
    """Gross profit divided by gross loss across trades.

    Returns ``0.0`` if there are no trades, and a large sentinel (``999.0``)
    when there are only winners, so the value is always JSON/printf-friendly.
    """
    gains = sum(p for p in trade_profits if p > 0)
    losses = -sum(p for p in trade_profits if p < 0)
    if losses == 0:
        return 999.0 if gains > 0 else 0.0
    return gains / losses


def win_rate(trade_profits: list[float]) -> float:
    """Fraction of trades that closed profitably (0-1)."""
    if not trade_profits:
        return 0.0
    wins = sum(1 for p in trade_profits if p > 0)
    return wins / len(trade_profits)


def summary(equity: pd.Series, trade_profits: list[float] | None = None) -> dict[str, float]:
    """Bundle the headline metrics for an equity curve (and optional trades)."""
    profits = trade_profits or []
    return {
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "volatility": annualized_volatility(daily_returns(equity)),
        "sharpe": sharpe_ratio(daily_returns(equity)),
        "sortino": sortino_ratio(daily_returns(equity)),
        "calmar": calmar_ratio(equity),
        "max_drawdown": max_drawdown(equity),
        "profit_factor": profit_factor(profits),
        "win_rate": win_rate(profits),
    }
