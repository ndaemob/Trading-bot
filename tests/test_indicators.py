"""Tests for technical-indicator calculations (fully offline)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators import add_indicators, latest_indicators

from .conftest import make_ohlcv

INDICATOR_COLUMNS = [
    "RSI",
    "SMA_20",
    "SMA_50",
    "SMA_200",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "ATR",
]


def sample_df() -> pd.DataFrame:
    return make_ohlcv(np.linspace(50, 150, 250))


def test_add_indicators_adds_all_columns():
    df = add_indicators(sample_df())
    for col in INDICATOR_COLUMNS:
        assert col in df.columns


def test_add_indicators_empty_raises():
    with pytest.raises(ValueError):
        add_indicators(pd.DataFrame())


def test_add_indicators_missing_columns_raises():
    with pytest.raises(ValueError):
        add_indicators(pd.DataFrame({"Close": [1, 2, 3]}))


def test_rsi_within_bounds():
    df = add_indicators(sample_df())
    rsi = df["RSI"].dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()


def test_add_indicators_does_not_mutate_input():
    base = sample_df()
    before = list(base.columns)
    add_indicators(base)
    assert list(base.columns) == before


def test_latest_indicators_returns_floats():
    df = add_indicators(sample_df())
    latest = latest_indicators(df)
    assert "RSI" in latest
    assert isinstance(latest["Close"], float)


def test_latest_indicators_empty_raises():
    with pytest.raises(ValueError):
        latest_indicators(pd.DataFrame())
