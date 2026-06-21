"""Portfolio-level backtesting across multiple tickers.

Unlike the single-ticker :mod:`~src.backtest`, this simulates managing a whole
watchlist at once under the project's risk rules: each new position is sized by
:func:`risk.calculate_position_size` (max 2% risk, max 20% allocation), capital
is shared across names, and performance is compared against an equal-weight
**buy-and-hold benchmark**.

It is still a simplified, long-only simulation — no broker, no real orders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from . import backtest, config, metrics, risk, strategy


@dataclass
class PortfolioResult:
    """Aggregate outcome of a multi-ticker portfolio backtest."""

    tickers: list[str]
    initial_capital: float
    final_value: float
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    profit_factor: float
    benchmark_return: float
    benchmark_cagr: float
    equity_curve: pd.Series | None = None
    benchmark_curve: pd.Series | None = None
    trades: list[backtest.Trade] = field(default_factory=list)


def _safe(value: float) -> float:
    """Return ``value`` as a float, mapping NaN to 0.0."""
    v = float(value)
    return 0.0 if math.isnan(v) else v


def _holdings_value(holdings: dict[str, float], closes: pd.DataFrame, date: object) -> float:
    """Mark open holdings to market at ``date``."""
    return sum(shares * _safe(closes.at[date, t]) for t, shares in holdings.items())


def run_portfolio_backtest(
    data: dict[str, pd.DataFrame],
    initial_capital: float = config.DEFAULT_PORTFOLIO_VALUE,
    *,
    commission_pct: float = config.COMMISSION_PCT,
    slippage_pct: float = config.SLIPPAGE_PCT,
    risk_per_trade: float = config.MAX_RISK_PER_TRADE,
    max_allocation: float = config.MAX_ALLOCATION_PER_STOCK,
    max_positions: int | None = None,
) -> PortfolioResult:
    """Backtest a watchlist as a single risk-managed portfolio.

    On each shared trading day: ``SELL`` signals are liquidated first (freeing
    capital), then ``BUY`` signals open risk-sized positions while cash and the
    optional ``max_positions`` cap allow. Costs model commission and slippage.

    Args:
        data: Mapping of ticker -> DataFrame already processed by
            :func:`indicators.add_indicators`.
        initial_capital: Starting capital for the whole portfolio.
        commission_pct: Commission per trade as a fraction of notional.
        slippage_pct: Adverse slippage applied to each fill.
        risk_per_trade: Risk budget per position (default 2%).
        max_allocation: Allocation cap per position (default 20%).
        max_positions: Optional cap on concurrent open positions.

    Returns:
        A :class:`PortfolioResult` with risk-adjusted metrics and a benchmark.

    Raises:
        ValueError: If ``data`` is empty or the tickers share too little history.
    """
    if not data:
        raise ValueError("No data provided for portfolio backtest.")

    tickers = sorted(data)

    index: pd.Index | None = None
    for df in data.values():
        index = df.index if index is None else index.intersection(df.index)
    if index is None or len(index) < 2:
        raise ValueError("Tickers do not share enough overlapping history.")
    index = index.sort_values()

    closes = pd.DataFrame({t: data[t]["Close"] for t in tickers}).reindex(index)
    signals = pd.DataFrame({t: strategy.signal_series(data[t]) for t in tickers}).reindex(index)
    has_atr = all("ATR" in data[t].columns for t in tickers)
    atrs = pd.DataFrame({t: data[t]["ATR"] for t in tickers}).reindex(index) if has_atr else None

    cash = float(initial_capital)
    holdings: dict[str, float] = {}
    entry_price: dict[str, float] = {}
    entry_cost: dict[str, float] = {}
    entry_date: dict[str, object] = {}
    trades: list[backtest.Trade] = []
    equity_points: list[float] = []

    for date in index:
        # 1) Liquidate SELLs first so freed cash is available to BUYs.
        for t in list(holdings):
            if signals.at[date, t] != strategy.SELL:
                continue
            price = _safe(closes.at[date, t])
            if price <= 0:
                continue
            fill = price * (1 - slippage_pct)
            proceeds = holdings[t] * fill
            net = proceeds - proceeds * commission_pct
            cash += net
            trades.append(
                backtest.Trade(
                    entry_date=entry_date[t],
                    entry_price=entry_price[t],
                    exit_date=date,
                    exit_price=fill,
                    shares=holdings[t],
                    entry_cost=entry_cost[t],
                    exit_proceeds=net,
                )
            )
            del holdings[t], entry_price[t], entry_cost[t], entry_date[t]

        # 2) Open new positions, sizing each against current total equity.
        equity = cash + _holdings_value(holdings, closes, date)
        for t in tickers:
            if t in holdings or signals.at[date, t] != strategy.BUY:
                continue
            if max_positions is not None and len(holdings) >= max_positions:
                break
            price = _safe(closes.at[date, t])
            if price <= 0:
                continue
            fill = price * (1 + slippage_pct)
            atr = float(atrs.at[date, t]) if atrs is not None else float("nan")
            try:
                sized = risk.calculate_position_size(
                    portfolio_value=equity,
                    entry_price=fill,
                    stop_loss=backtest._stop_for_entry(fill, atr),
                    max_risk_per_trade=risk_per_trade,
                    max_allocation=max_allocation,
                )
            except ValueError:
                continue
            affordable = math.floor(cash / (fill * (1 + commission_pct)))
            qty = float(min(sized.shares, affordable))
            if qty <= 0:
                continue
            cost = qty * fill
            total = cost + cost * commission_pct
            if total <= cash + 1e-9:
                cash -= total
                holdings[t] = qty
                entry_price[t] = fill
                entry_cost[t] = total
                entry_date[t] = date

        equity_points.append(cash + _holdings_value(holdings, closes, date))

    equity_curve = pd.Series(equity_points, index=index, name="equity")
    final_value = float(equity_curve.iloc[-1])
    returns = metrics.daily_returns(equity_curve)
    trade_profits = [t.profit for t in trades]

    benchmark_curve = _benchmark_curve(closes, tickers, initial_capital)
    benchmark_return = metrics.total_return(benchmark_curve)
    benchmark_cagr = metrics.cagr(benchmark_curve)

    return PortfolioResult(
        tickers=tickers,
        initial_capital=float(initial_capital),
        final_value=round(final_value, 2),
        total_return=metrics.total_return(equity_curve),
        cagr=metrics.cagr(equity_curve),
        volatility=metrics.annualized_volatility(returns),
        sharpe=metrics.sharpe_ratio(returns),
        sortino=metrics.sortino_ratio(returns),
        max_drawdown=metrics.max_drawdown(equity_curve),
        num_trades=len(trades),
        win_rate=metrics.win_rate(trade_profits),
        profit_factor=metrics.profit_factor(trade_profits),
        benchmark_return=benchmark_return,
        benchmark_cagr=benchmark_cagr,
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
        trades=trades,
    )


def _benchmark_curve(closes: pd.DataFrame, tickers: list[str], initial_capital: float) -> pd.Series:
    """Equal-weight buy-and-hold equity curve over the same window."""
    per_ticker = initial_capital / len(tickers)
    shares = {}
    for t in tickers:
        first = float(closes[t].iloc[0])
        if first > 0:
            shares[t] = per_ticker / first
    if not shares:
        return pd.Series(initial_capital, index=closes.index, name="benchmark")
    weights = pd.Series(shares)
    curve = (closes[list(shares)] * weights).sum(axis=1)
    curve.name = "benchmark"
    return curve


def format_result(result: PortfolioResult) -> str:
    """Render a :class:`PortfolioResult` as a readable multi-line string."""
    edge = result.total_return - result.benchmark_return
    lines = [
        f"Portfolio Backtest: {', '.join(result.tickers)}",
        f"  Initial Capital: {result.initial_capital:,.2f}",
        f"  Final Value:     {result.final_value:,.2f}",
        f"  Total Return:    {result.total_return:+.2%}",
        f"  CAGR:            {result.cagr:+.2%}",
        f"  Sharpe / Sortino:{result.sharpe:>6.2f} / {result.sortino:.2f}",
        f"  Volatility:      {result.volatility:.2%}",
        f"  Max Drawdown:    {result.max_drawdown:.2%}",
        f"  Trades:          {result.num_trades}",
        f"  Win Rate:        {result.win_rate:.0%}",
        f"  Profit Factor:   {result.profit_factor:.2f}",
        f"  Benchmark (B&H): {result.benchmark_return:+.2%} "
        f"(CAGR {result.benchmark_cagr:+.2%})",
        f"  Edge vs B&H:     {edge:+.2%}",
    ]
    return "\n".join(lines)
