"""A simple, long-only historical backtester.

Replays the strategy across historical data to give a rough sense of how it
*would have* behaved. This is a teaching/MVP backtester: it is long-only and
assumes fills at the day's close, but it now models **commission** and
**slippage** and can size positions either all-in or via the project's
risk rules.

**Past performance does not guarantee future results.** Nothing here executes
a real trade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from . import config, risk, strategy

SizingMode = Literal["all_in", "risk_based"]


@dataclass
class Trade:
    """A single completed round-trip (buy then sell), net of costs.

    ``entry_cost`` is the cash paid to open (including commission/slippage) and
    ``exit_proceeds`` is the cash received to close (net of commission/slippage),
    so profit and return are reported *after* trading costs.
    """

    entry_date: object
    entry_price: float
    exit_date: object
    exit_price: float
    shares: float
    entry_cost: float
    exit_proceeds: float

    @property
    def profit(self) -> float:
        """Net cash profit of the trade after costs."""
        return self.exit_proceeds - self.entry_cost

    @property
    def return_pct(self) -> float:
        """Fractional return on capital deployed (e.g. ``0.10`` for +10%)."""
        if self.entry_cost <= 0:
            return 0.0
        return self.profit / self.entry_cost

    @property
    def is_win(self) -> bool:
        """``True`` if the trade closed at a net profit."""
        return self.profit > 0


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


def _stop_for_entry(fill_price: float, atr: float) -> float:
    """Derive a stop-loss for risk-based sizing from ATR, with a safe fallback.

    Uses an ATR-based stop when ATR is sensible, otherwise an 8% stop. The
    result is always strictly between 0 and ``fill_price``.
    """
    stop = fill_price - config.STOP_LOSS_ATR * atr
    if math.isnan(atr) or not (0.0 < stop < fill_price):
        stop = fill_price * 0.92
    return stop


def _shares_to_buy(
    cash: float,
    fill_price: float,
    atr: float,
    sizing: SizingMode,
    commission_pct: float,
    risk_per_trade: float,
    max_allocation: float,
) -> float:
    """Decide how many shares to buy under the chosen sizing mode."""
    if sizing == "risk_based":
        try:
            sized = risk.calculate_position_size(
                portfolio_value=cash,
                entry_price=fill_price,
                stop_loss=_stop_for_entry(fill_price, atr),
                max_risk_per_trade=risk_per_trade,
                max_allocation=max_allocation,
            )
        except ValueError:
            return 0.0
        return float(sized.shares)
    # All-in: spend all cash, leaving room for commission.
    return cash / (fill_price * (1 + commission_pct))


def run_backtest(
    df: pd.DataFrame,
    ticker: str = "",
    initial_capital: float = config.DEFAULT_PORTFOLIO_VALUE,
    *,
    commission_pct: float = 0.0,
    slippage_pct: float = 0.0,
    position_sizing: SizingMode = "all_in",
    risk_per_trade: float = config.MAX_RISK_PER_TRADE,
    max_allocation: float = config.MAX_ALLOCATION_PER_STOCK,
) -> BacktestResult:
    """Run the strategy over ``df`` and report performance.

    Trading rules:

    * Buy on a ``BUY`` signal when flat (size per ``position_sizing``).
    * Liquidate fully on a ``SELL`` signal when in a position.
    * ``HOLD`` / ``WATCH`` do nothing.
    * Any open position is marked-to-market at the final close.

    Costs: each fill is moved adversely by ``slippage_pct`` and charged
    ``commission_pct`` of notional.

    Args:
        df: DataFrame processed by :func:`indicators.add_indicators`.
        ticker: Optional symbol for reporting.
        initial_capital: Starting capital (default 10,000).
        commission_pct: Commission per trade as a fraction of notional.
        slippage_pct: Adverse slippage applied to each fill.
        position_sizing: ``"all_in"`` or ``"risk_based"``.
        risk_per_trade: Risk budget per trade when sizing by risk.
        max_allocation: Allocation cap per trade when sizing by risk.

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
    has_atr = "ATR" in df.columns

    cash: float = float(initial_capital)
    shares: float = 0.0
    entry_price: float = 0.0
    entry_cost: float = 0.0
    entry_date: object = None

    trades: list[Trade] = []
    equity_points: list[float] = []

    for date, row in df.iterrows():
        price = float(row["Close"])
        signal = signals.loc[date]

        if signal == strategy.BUY and shares == 0 and price > 0:
            fill = price * (1 + slippage_pct)
            atr = float(row["ATR"]) if has_atr else float("nan")
            qty = _shares_to_buy(
                cash,
                fill,
                atr,
                position_sizing,
                commission_pct,
                risk_per_trade,
                max_allocation,
            )
            cost = qty * fill
            total = cost + cost * commission_pct
            if qty > 0 and total <= cash + 1e-9:
                cash -= total
                shares = qty
                entry_price = fill
                entry_cost = total
                entry_date = date
        elif signal == strategy.SELL and shares > 0:
            fill = price * (1 - slippage_pct)
            proceeds = shares * fill
            net = proceeds - proceeds * commission_pct
            cash += net
            trades.append(
                Trade(
                    entry_date=entry_date,
                    entry_price=entry_price,
                    exit_date=date,
                    exit_price=fill,
                    shares=shares,
                    entry_cost=entry_cost,
                    exit_proceeds=net,
                )
            )
            shares = 0.0
            entry_price = 0.0
            entry_cost = 0.0
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
