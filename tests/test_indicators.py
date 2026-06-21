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


EXTENDED_COLUMNS = [
    "EMA_12",
    "EMA_26",
    "BB_High",
    "BB_Low",
    "BB_Mid",
    "BB_Pct",
    "ADX",
    "ADX_Pos",
    "ADX_Neg",
    "Stoch_K",
    "Stoch_D",
    "OBV",
    "Vol_SMA_20",
]


def test_add_indicators_adds_all_columns():
    df = add_indicators(sample_df())
    for col in INDICATOR_COLUMNS:
        assert col in df.columns


def test_add_indicators_adds_extended_columns():
    df = add_indicators(sample_df())
    for col in EXTENDED_COLUMNS:
        assert col in df.columns


def test_volume_indicators_skipped_without_volume():
    df = sample_df().drop(columns=["Volume"])
    out = add_indicators(df)
    assert "OBV" not in out.columns
    assert "ADX" in out.columns  # non-volume indicators still present


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
