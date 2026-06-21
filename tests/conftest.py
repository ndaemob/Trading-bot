"""Shared pytest fixtures and synthetic-data helpers.

All tests run fully offline: they build deterministic OHLCV frames in memory so
no network access or live market data is ever required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import indicators


def make_ohlcv(close: np.ndarray, start: str = "2021-01-01") -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from an array of closing prices.

    High/Low are derived as a small band around Close so ATR is well-defined.

    Args:
        close: 1-D array of closing prices.
        start: First date in the (business-day) index.

    Returns:
        A DataFrame with Open/High/Low/Close/Volume and a DatetimeIndex.
    """
    close = np.asarray(close, dtype="float64")
    n = len(close)
    index = pd.date_range(start=start, periods=n, freq="B")
    high = close * 1.01
    low = close * 0.99
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = np.full(n, 1_000_000.0)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


@pytest.fixture
def uptrend_df() -> pd.DataFrame:
    """A 300-bar realistic uptrend with indicators attached.

    Upward drift plus a small oscillation gives a mix of up and down days, so
    RSI settles into a healthy band (~mid-50s) rather than saturating at 100 as
    a perfectly straight line would. Result: SMA50 > SMA200 and a BUY-able RSI.
    """
    x = np.arange(300)
    close = np.linspace(50, 150, 300) + 6 * np.sin(x / 7.0)
    return indicators.add_indicators(make_ohlcv(close))


@pytest.fixture
def downtrend_df() -> pd.DataFrame:
    """A 300-bar realistic downtrend with indicators attached (SMA50 < SMA200)."""
    x = np.arange(300)
    close = np.linspace(150, 50, 300) + 6 * np.sin(x / 7.0)
    return indicators.add_indicators(make_ohlcv(close))


@pytest.fixture
def choppy_df() -> pd.DataFrame:
    """A 300-bar oscillating series with indicators attached."""
    x = np.arange(300)
    close = 100 + 10 * np.sin(x / 10.0)
    return indicators.add_indicators(make_ohlcv(close))


@pytest.fixture
def trade_df() -> pd.DataFrame:
    """A 400-bar rise-then-fall series that triggers a full buy→sell round-trip.

    The long rise establishes SMA50 > SMA200 with a healthy RSI (→ BUY); the
    later decline flips SMA50 below SMA200 (→ SELL), so a backtest records at
    least one completed trade.
    """
    rise = np.linspace(50, 160, 300) + 6 * np.sin(np.arange(300) / 7.0)
    fall = np.linspace(160, 90, 100) + 6 * np.sin(np.arange(300, 400) / 7.0)
    close = np.concatenate([rise, fall])
    return indicators.add_indicators(make_ohlcv(close))
