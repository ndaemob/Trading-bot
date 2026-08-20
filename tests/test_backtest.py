"""Tests for the historical backtester (fully offline)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest import BacktestResult, run_backtest


def test_run_backtest_basic(trade_df):
    result = run_backtest(trade_df, "T")
    assert isinstance(result, BacktestResult)
    assert result.num_trades >= 1
    assert 0.0 <= result.win_rate <= 1.0
    assert result.max_drawdown >= 0.0
    assert result.equity_curve is not None
    assert len(result.equity_curve) == len(trade_df)


def test_empty_raises():
    with pytest.raises(ValueError):
        run_backtest(pd.DataFrame(), "T")


def test_missing_close_raises():
    df = pd.DataFrame({"SMA_50": [1.0], "SMA_200": [1.0], "RSI": [50.0]})
    with pytest.raises(ValueError):
        run_backtest(df, "T")


def test_costs_reduce_final_value(trade_df):
    free = run_backtest(trade_df, "T", commission_pct=0.0, slippage_pct=0.0)
    costly = run_backtest(trade_df, "T", commission_pct=0.01, slippage_pct=0.01)
    assert costly.final_value < free.final_value


def test_risk_based_sizing_runs(trade_df):
    result = run_backtest(trade_df, "T", position_sizing="risk_based")
    assert result.final_value > 0
    assert result.num_trades >= 1


def test_trade_profit_consistency(trade_df):
    result = run_backtest(trade_df, "T")
    for trade in result.trades:
        assert trade.profit == pytest.approx(trade.exit_proceeds - trade.entry_cost)
        assert trade.is_win == (trade.profit > 0)


def test_win_rate_matches_trades(trade_df):
    result = run_backtest(trade_df, "T")
    wins = sum(1 for t in result.trades if t.is_win)
    assert result.win_rate == pytest.approx(wins / result.num_trades)
