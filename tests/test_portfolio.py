"""Tests for the multi-ticker portfolio backtester (fully offline)."""

from __future__ import annotations

import numpy as np
import pytest

from src import indicators, portfolio

from .conftest import make_ohlcv


def _df(close: np.ndarray, start: str = "2021-01-01"):
    return indicators.add_indicators(make_ohlcv(close, start=start))


def _two_tickers() -> dict:
    rise = np.linspace(50, 160, 300) + 6 * np.sin(np.arange(300) / 7.0)
    fall = np.linspace(160, 90, 100) + 6 * np.sin(np.arange(300, 400) / 7.0)
    a = np.concatenate([rise, fall])
    b = np.linspace(40, 120, 400) + 5 * np.sin(np.arange(400) / 9.0)
    return {"AAA": _df(a), "BBB": _df(b)}


def test_portfolio_runs_and_reports():
    res = portfolio.run_portfolio_backtest(_two_tickers(), initial_capital=10_000)
    assert res.final_value > 0
    assert 0.0 <= res.win_rate <= 1.0
    assert res.max_drawdown >= 0.0
    assert res.equity_curve is not None and len(res.equity_curve) > 0
    assert res.benchmark_curve is not None
    assert set(res.tickers) == {"AAA", "BBB"}


def test_portfolio_never_uses_leverage():
    res = portfolio.run_portfolio_backtest(_two_tickers(), initial_capital=10_000)
    # No leverage => equity can never collapse below zero.
    assert res.equity_curve is not None
    assert float(res.equity_curve.min()) > 0


def test_portfolio_empty_raises():
    with pytest.raises(ValueError):
        portfolio.run_portfolio_backtest({})


def test_portfolio_insufficient_overlap_raises():
    a = _df(np.linspace(50, 60, 250), start="2018-01-01")
    b = _df(np.linspace(50, 60, 250), start="2023-01-01")
    with pytest.raises(ValueError):
        portfolio.run_portfolio_backtest({"A": a, "B": b})


def test_portfolio_max_positions_one():
    res = portfolio.run_portfolio_backtest(_two_tickers(), initial_capital=10_000, max_positions=1)
    assert res.final_value > 0


def test_portfolio_has_benchmark():
    res = portfolio.run_portfolio_backtest(_two_tickers(), initial_capital=10_000)
    # Benchmark return is a finite number derived from buy-and-hold.
    assert isinstance(res.benchmark_return, float)
    assert res.benchmark_curve is not None and len(res.benchmark_curve) == len(res.equity_curve)
