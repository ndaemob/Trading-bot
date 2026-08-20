"""Tests for chart rendering (headless, fully offline)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import charts
from src.backtest import run_backtest


def test_plot_price_creates_file(uptrend_df, tmp_path):
    path = charts.plot_price(uptrend_df, "TEST", output_dir=str(tmp_path))
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0


def test_plot_price_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        charts.plot_price(pd.DataFrame(), "X", output_dir=str(tmp_path))


def test_plot_backtest_creates_file(trade_df, tmp_path):
    result = run_backtest(trade_df, "TEST")
    path = charts.plot_backtest(result, output_dir=str(tmp_path))
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0
