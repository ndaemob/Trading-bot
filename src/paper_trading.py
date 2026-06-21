"""Paper-trading: a persistent, fully simulated portfolio.

This lets the bot "act" on its signals **without any real execution**. State
(cash, positions, trade history) is persisted to a JSON file so a simulated
account can evolve across runs.

Nothing here connects to a broker. It is a sandbox for evaluating the strategy.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .strategy import BUY, SELL, Signal


@dataclass
class Position:
    """An open simulated position in a single ticker."""

    ticker: str
    shares: float
    avg_price: float


@dataclass
class PaperTrade:
    """A single simulated fill recorded in the portfolio history."""

    timestamp: str
    action: str
    ticker: str
    shares: float
    price: float


@dataclass
class PaperPortfolio:
    """A simulated portfolio: cash, open positions, and trade history."""

    initial_capital: float = config.DEFAULT_PORTFOLIO_VALUE
    cash: float = config.DEFAULT_PORTFOLIO_VALUE
    positions: dict[str, Position] = field(default_factory=dict)
    history: list[PaperTrade] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #
    def buy(self, ticker: str, shares: float, price: float) -> None:
        """Simulate buying ``shares`` of ``ticker`` at ``price``.

        Raises:
            ValueError: For non-positive inputs or insufficient cash
                (leverage is never allowed).
        """
        ticker = ticker.upper()
        if shares <= 0 or price <= 0:
            raise ValueError("shares and price must be positive.")
        cost = shares * price
        if cost > self.cash + 1e-9:
            raise ValueError(
                f"Insufficient cash for {ticker}: need {cost:.2f}, "
                f"have {self.cash:.2f} (no leverage)."
            )

        self.cash -= cost
        if ticker in self.positions:
            pos = self.positions[ticker]
            total_shares = pos.shares + shares
            pos.avg_price = (pos.shares * pos.avg_price + shares * price) / total_shares
            pos.shares = total_shares
        else:
            self.positions[ticker] = Position(ticker, shares, price)
        self._record("BUY", ticker, shares, price)

    def sell(self, ticker: str, shares: float, price: float) -> None:
        """Simulate selling ``shares`` of ``ticker`` at ``price``.

        Raises:
            ValueError: For non-positive inputs or selling more than held.
        """
        ticker = ticker.upper()
        if shares <= 0 or price <= 0:
            raise ValueError("shares and price must be positive.")
        pos = self.positions.get(ticker)
        if pos is None or shares > pos.shares + 1e-9:
            held = pos.shares if pos else 0.0
            raise ValueError(f"Cannot sell {shares} of {ticker}; only {held} held.")

        self.cash += shares * price
        pos.shares -= shares
        if pos.shares <= 1e-9:
            del self.positions[ticker]
        self._record("SELL", ticker, shares, price)

    def _record(self, action: str, ticker: str, shares: float, price: float) -> None:
        """Append a fill to the trade history."""
        self.history.append(
            PaperTrade(
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                action=action,
                ticker=ticker,
                shares=round(shares, 6),
                price=round(price, 4),
            )
        )

    # ------------------------------------------------------------------ #
    # Valuation & serialisation
    # ------------------------------------------------------------------ #
    def market_value(self, prices: dict[str, float]) -> float:
        """Total value = cash + positions marked at the supplied ``prices``.

        Positions without a supplied price fall back to their average cost.
        """
        value = self.cash
        for ticker, pos in self.positions.items():
            value += pos.shares * prices.get(ticker, pos.avg_price)
        return value

    def to_dict(self) -> dict:
        """Serialise to a plain JSON-friendly dictionary."""
        return {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "positions": {t: asdict(p) for t, p in self.positions.items()},
            "history": [asdict(h) for h in self.history],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PaperPortfolio":
        """Rebuild a portfolio from :meth:`to_dict` output."""
        positions = {t: Position(**p) for t, p in data.get("positions", {}).items()}
        history = [PaperTrade(**h) for h in data.get("history", [])]
        return cls(
            initial_capital=data.get("initial_capital", config.DEFAULT_PORTFOLIO_VALUE),
            cash=data.get("cash", config.DEFAULT_PORTFOLIO_VALUE),
            positions=positions,
            history=history,
        )


def load_portfolio(
    path: str = config.PAPER_PORTFOLIO_FILE,
    initial_capital: float = config.DEFAULT_PORTFOLIO_VALUE,
) -> PaperPortfolio:
    """Load a portfolio from ``path``, or create a fresh one if absent/corrupt."""
    file = Path(path)
    if file.exists():
        try:
            return PaperPortfolio.from_dict(json.loads(file.read_text()))
        except (json.JSONDecodeError, TypeError, ValueError):
            # A corrupt file should not crash the run; start fresh instead.
            pass
    return PaperPortfolio(initial_capital=initial_capital, cash=initial_capital)


def save_portfolio(
    portfolio: PaperPortfolio,
    path: str = config.PAPER_PORTFOLIO_FILE,
) -> None:
    """Persist ``portfolio`` to ``path`` as pretty-printed JSON."""
    file = Path(path)
    if file.parent != Path(""):
        file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(portfolio.to_dict(), indent=2))


def execute_signal(
    portfolio: PaperPortfolio,
    signal: Signal,
    shares: float,
) -> str:
    """Apply a signal to the paper portfolio and return a description.

    * ``BUY``: open a position of up to ``shares`` (capped by available cash)
      if none is currently held for the ticker.
    * ``SELL``: liquidate the entire held position, if any.
    * ``WATCH`` / ``HOLD``: no action.

    Args:
        portfolio: The portfolio to mutate.
        signal: The generated signal.
        shares: Desired share count (e.g. from risk-based sizing).

    Returns:
        A human-readable description of the simulated action.
    """
    ticker = signal.ticker.upper()
    price = signal.latest_close

    if signal.signal == BUY:
        if ticker in portfolio.positions:
            return f"{ticker}: already holding; no new buy."
        if price <= 0:
            return f"{ticker}: invalid price; skipped."
        affordable = math.floor(portfolio.cash / price)
        qty = min(int(shares), affordable)
        if qty <= 0:
            return f"{ticker}: insufficient cash to buy; skipped."
        portfolio.buy(ticker, qty, price)
        return f"{ticker}: BUY {qty} @ {price:.2f} (paper)."

    if signal.signal == SELL:
        pos = portfolio.positions.get(ticker)
        if pos is None:
            return f"{ticker}: no position to sell."
        held = pos.shares
        portfolio.sell(ticker, held, price)
        return f"{ticker}: SELL {held:g} @ {price:.2f} (paper)."

    return f"{ticker}: {signal.signal} — no paper action."
