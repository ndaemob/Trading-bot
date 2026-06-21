"""A simple, long-only historical backtester.

Replays the strategy across historical data to give a rough sense of how it
*would have* behaved. This is a teaching/MVP backtester: it is all-in,
long-only, ignores fees, slippage and dividends, and assumes fills at the
day's closing price.

**Past performance does not guarantee future results.** Nothing here executes
a real trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import config, strategy


@dataclass
class Trade:
    """A single completed round-trip (buy then sell)."""

    entry_date: object
    entry_price: float
    exit_date: object
    exit_price: float

    @property
    def return_pct(self) -> float:
        """Fractional return of the trade (e.g. ``0.10`` for +10%)."""
        if self.entry_price <= 0:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price

    @property
    def is_win(self) -> bool:
        """``True`` if the trade closed at a profit."""
        return self.exit_price > self.entry_price


@dataclass
class BacktestResult:
    """Aggregate outcome of a backtest run.

    Attributes:
        ticker: Symbol tested.
        initial_capital: Starting hypothetical capital.
        final_value: Ending portfolio value (cash + open position).
        total_return: Fractional return over the whole period.
        num_trades: Number of completed round-trip trades.
        win_rate: Fraction of trades that were profitable (0-1).
        max_drawdown: Largest peak-to-trough equity drop (0-1, positive).
        trades: The individual completed trades.
        equity_curve: Portfolio value over time.
    """

    ticker: str
    initial_capital: float
    final_value: float
    total_return: float
    num_trades: int
    win_rate: float
    max_drawdown: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series | None = None


def _max_drawdown(equity: pd.Series) -> float:
    """Return the maximum peak-to-trough drawdown of an equity curve.

    Args:
        equity: Series of portfolio values over time.

    Returns:
        Drawdown as a positive fraction (``0.25`` == a 25% drop). ``0.0`` if
        the curve never falls or is empty.
    """
    if equity is None or equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdowns = (equity - running_max) / running_max
    worst = drawdowns.min()
    return float(abs(worst)) if worst < 0 else 0.0


def run_backtest(
    df: pd.DataFrame,
    ticker: str = "",
    initial_capital: float = config.DEFAULT_PORTFOLIO_VALUE,
) -> BacktestResult:
    """Run the strategy over ``df`` and report performance.

    Trading rules:

    * Go all-in (long) on a ``BUY`` signal when flat.
    * Liquidate fully on a ``SELL`` signal when in a position.
    * ``HOLD`` / ``WATCH`` do nothing.
    * Any open position is marked-to-market at the final close.

    Args:
        df: DataFrame processed by :func:`indicators.add_indicators`.
        ticker: Optional symbol for reporting.
        initial_capital: Starting capital (default 10,000).

    Returns:
        A :class:`BacktestResult` summarising the run.

    Raises:
        ValueError: If ``df`` is empty or missing required columns.
    """
    if df is None or df.empty:
        raise ValueError("Cannot backtest on empty data.")
    if "Close" not in df.columns:
        raise ValueError("Backtest requires a 'Close' column.")

    signals = strategy.signal_series(df)

    cash: float = float(initial_capital)
    shares: float = 0.0
    entry_price: float = 0.0
    entry_date: object = None

    trades: list[Trade] = []
    equity_points: list[float] = []

    for date, row in df.iterrows():
        price = float(row["Close"])
        signal = signals.loc[date]

        if signal == strategy.BUY and shares == 0 and price > 0:
            shares = cash / price  # fractional shares keep the MVP simple
            cash = 0.0
            entry_price = price
            entry_date = date
        elif signal == strategy.SELL and shares > 0:
            cash = shares * price
            trades.append(
                Trade(
                    entry_date=entry_date,
                    entry_price=entry_price,
                    exit_date=date,
                    exit_price=price,
                )
            )
            shares = 0.0
            entry_price = 0.0
            entry_date = None

        equity_points.append(cash + shares * price)

    equity_curve = pd.Series(equity_points, index=df.index, name="equity")
    final_value = float(equity_curve.iloc[-1])

    wins = sum(1 for t in trades if t.is_win)
    num_trades = len(trades)
    win_rate = (wins / num_trades) if num_trades else 0.0
    total_return = (final_value - initial_capital) / initial_capital

    return BacktestResult(
        ticker=ticker.upper() if ticker else "",
        initial_capital=float(initial_capital),
        final_value=round(final_value, 2),
        total_return=total_return,
        num_trades=num_trades,
        win_rate=win_rate,
        max_drawdown=_max_drawdown(equity_curve),
        trades=trades,
        equity_curve=equity_curve,
    )


def format_result(result: BacktestResult) -> str:
    """Render a :class:`BacktestResult` as a readable multi-line string."""
    lines = [
        f"Backtest: {result.ticker or 'N/A'}",
        f"  Initial Capital: {result.initial_capital:,.2f}",
        f"  Final Value:     {result.final_value:,.2f}",
        f"  Total Return:    {result.total_return:+.2%}",
        f"  Trades:          {result.num_trades}",
        f"  Win Rate:        {result.win_rate:.0%}",
        f"  Max Drawdown:    {result.max_drawdown:.2%}",
    ]
    return "\n".join(lines)
