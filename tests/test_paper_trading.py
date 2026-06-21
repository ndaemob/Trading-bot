"""Tests for the simulated paper-trading portfolio (fully offline)."""

from __future__ import annotations

import pytest

from src.paper_trading import (
    PaperPortfolio,
    execute_signal,
    load_portfolio,
    save_portfolio,
)
from src.strategy import Signal


def _signal(label: str, ticker: str = "AAPL", close: float = 100.0) -> Signal:
    return Signal(
        ticker=ticker,
        signal=label,
        confidence=50,
        latest_close=close,
        stop_loss=90.0,
        entry_zone=(95.0, 99.0),
    )


def test_new_portfolio_starts_with_cash():
    p = PaperPortfolio(initial_capital=10_000, cash=10_000)
    assert p.cash == 10_000
    assert p.positions == {}


def test_buy_reduces_cash_and_adds_position():
    p = PaperPortfolio(initial_capital=10_000, cash=10_000)
    p.buy("AAPL", 10, 100)
    assert p.cash == 9_000
    assert p.positions["AAPL"].shares == 10


def test_buy_insufficient_cash_raises():
    p = PaperPortfolio(initial_capital=100, cash=100)
    with pytest.raises(ValueError):
        p.buy("AAPL", 10, 100)


def test_average_price_updates_on_additional_buy():
    p = PaperPortfolio(initial_capital=10_000, cash=10_000)
    p.buy("AAPL", 10, 100)
    p.buy("AAPL", 10, 200)
    assert p.positions["AAPL"].shares == 20
    assert p.positions["AAPL"].avg_price == pytest.approx(150)


def test_sell_adds_cash_and_removes_position():
    p = PaperPortfolio(initial_capital=10_000, cash=10_000)
    p.buy("AAPL", 10, 100)
    p.sell("AAPL", 10, 120)
    assert "AAPL" not in p.positions
    assert p.cash == pytest.approx(10_000 - 1_000 + 1_200)


def test_oversell_raises():
    p = PaperPortfolio(initial_capital=10_000, cash=10_000)
    p.buy("AAPL", 5, 100)
    with pytest.raises(ValueError):
        p.sell("AAPL", 10, 100)


def test_market_value_marks_positions():
    p = PaperPortfolio(initial_capital=10_000, cash=10_000)
    p.buy("AAPL", 10, 100)  # cash now 9,000
    assert p.market_value({"AAPL": 150}) == pytest.approx(9_000 + 10 * 150)


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "pf.json"
    p = PaperPortfolio(initial_capital=10_000, cash=10_000)
    p.buy("AAPL", 10, 100)
    save_portfolio(p, str(path))
    loaded = load_portfolio(str(path))
    assert loaded.cash == pytest.approx(p.cash)
    assert loaded.positions["AAPL"].shares == 10


def test_load_missing_file_returns_fresh(tmp_path):
    loaded = load_portfolio(str(tmp_path / "absent.json"), initial_capital=5_000)
    assert loaded.cash == 5_000
    assert loaded.positions == {}


def test_execute_signal_buy_opens_position():
    p = PaperPortfolio(initial_capital=10_000, cash=10_000)
    msg = execute_signal(p, _signal("BUY"), shares=10)
    assert "BUY" in msg
    assert p.positions["AAPL"].shares == 10


def test_execute_signal_sell_closes_position():
    p = PaperPortfolio(initial_capital=10_000, cash=10_000)
    p.buy("AAPL", 10, 100)
    msg = execute_signal(p, _signal("SELL"), shares=0)
    assert "SELL" in msg
    assert "AAPL" not in p.positions


def test_execute_signal_hold_is_noop():
    p = PaperPortfolio(initial_capital=10_000, cash=10_000)
    msg = execute_signal(p, _signal("HOLD"), shares=10)
    assert "no paper action" in msg
    assert p.positions == {}


def test_execute_signal_buy_capped_by_cash():
    p = PaperPortfolio(initial_capital=150, cash=150)
    execute_signal(p, _signal("BUY", close=100.0), shares=10)
    assert p.positions["AAPL"].shares == 1  # only one share affordable
