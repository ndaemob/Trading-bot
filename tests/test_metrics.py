"""Tests for performance metrics (fully offline)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import metrics


def test_daily_returns_length_and_empty():
    eq = pd.Series([100.0, 110.0, 121.0])
    assert len(metrics.daily_returns(eq)) == 2
    assert metrics.daily_returns(pd.Series([100.0])).empty


def test_total_return():
    assert metrics.total_return(pd.Series([100.0, 150.0])) == pytest.approx(0.5)


def test_cagr_two_years():
    # 21% total over exactly two years -> ~10% CAGR.
    eq = pd.Series(np.linspace(100.0, 121.0, 505))
    assert metrics.cagr(eq) == pytest.approx(0.10, abs=1e-3)


def test_max_drawdown_known_case():
    eq = pd.Series([100.0, 120.0, 90.0, 130.0])
    assert metrics.max_drawdown(eq) == pytest.approx(0.25)


def test_max_drawdown_monotonic_is_zero():
    assert metrics.max_drawdown(pd.Series([100.0, 110.0, 120.0])) == 0.0


def test_sharpe_zero_for_constant_equity():
    eq = pd.Series([100.0] * 10)
    assert metrics.sharpe_ratio(metrics.daily_returns(eq)) == 0.0


def test_sharpe_positive_for_steady_uptrend():
    eq = pd.Series(np.linspace(100.0, 200.0, 252))
    assert metrics.sharpe_ratio(metrics.daily_returns(eq)) > 0


def test_sortino_handles_no_downside():
    eq = pd.Series(np.linspace(100.0, 200.0, 50))  # strictly up -> no downside
    assert metrics.sortino_ratio(metrics.daily_returns(eq)) == 0.0


def test_profit_factor_cases():
    assert metrics.profit_factor([10.0, -5.0, 20.0]) == pytest.approx(6.0)
    assert metrics.profit_factor([10.0, 20.0]) == 999.0  # winners only
    assert metrics.profit_factor([]) == 0.0


def test_win_rate():
    assert metrics.win_rate([10.0, -5.0, 20.0]) == pytest.approx(2 / 3)
    assert metrics.win_rate([]) == 0.0


def test_summary_keys():
    summ = metrics.summary(pd.Series(np.linspace(100.0, 130.0, 300)), [10.0, -2.0])
    for key in ("total_return", "cagr", "sharpe", "sortino", "max_drawdown"):
        assert key in summ
