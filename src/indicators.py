"""Technical-indicator calculations.

Wraps the :mod:`ta` library to enrich an OHLCV DataFrame with the handful of
indicators the strategy needs: RSI, three SMAs, MACD and ATR.

The functions here are pure: they take a DataFrame and return a new DataFrame,
never mutating their input and never touching the network.
"""

from __future__ import annotations

import pandas as pd

from . import config


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` enriched with technical indicators.

    Adds the following columns:

    * ``RSI`` — Relative Strength Index (default 14-period).
    * ``SMA_20`` / ``SMA_50`` / ``SMA_200`` — Simple Moving Averages.
    * ``MACD`` / ``MACD_Signal`` / ``MACD_Hist`` — Moving Average
      Convergence Divergence and its signal line / histogram.
    * ``ATR`` — Average True Range (volatility, used for stops/entries).

    Args:
        df: OHLCV DataFrame with at least ``High``, ``Low`` and ``Close``.

    Returns:
        A new DataFrame with the original columns plus the indicators above.

    Raises:
        ValueError: If required price columns are missing or the frame is empty.
    """
    if df is None or df.empty:
        raise ValueError("Cannot compute indicators on empty data.")

    required = {"High", "Low", "Close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for indicators: {sorted(missing)}")

    # Imported lazily so unit tests can patch/skip if 'ta' is unavailable.
    from ta.momentum import RSIIndicator
    from ta.trend import MACD, SMAIndicator
    from ta.volatility import AverageTrueRange

    out = df.copy()

    # ``ta`` expects 1-D float Series; squeeze() guards against (n, 1) frames.
    close = out["Close"].astype("float64").squeeze()
    high = out["High"].astype("float64").squeeze()
    low = out["Low"].astype("float64").squeeze()

    out["RSI"] = RSIIndicator(close=close, window=config.RSI_PERIOD).rsi()

    out["SMA_20"] = SMAIndicator(close=close, window=config.SMA_SHORT).sma_indicator()
    out["SMA_50"] = SMAIndicator(close=close, window=config.SMA_MEDIUM).sma_indicator()
    out["SMA_200"] = SMAIndicator(close=close, window=config.SMA_LONG).sma_indicator()

    macd = MACD(
        close=close,
        window_fast=config.MACD_FAST,
        window_slow=config.MACD_SLOW,
        window_sign=config.MACD_SIGNAL,
    )
    out["MACD"] = macd.macd()
    out["MACD_Signal"] = macd.macd_signal()
    out["MACD_Hist"] = macd.macd_diff()

    out["ATR"] = AverageTrueRange(
        high=high, low=low, close=close, window=config.ATR_PERIOD
    ).average_true_range()

    return out


def latest_indicators(df: pd.DataFrame) -> dict[str, float]:
    """Return the most recent row of indicator values as a plain dict.

    Convenience helper for the strategy layer. Any NaN values (e.g. SMA_200
    before 200 bars exist) are converted to ``float('nan')`` so callers can
    test them with :func:`math.isnan`.

    Args:
        df: A DataFrame already processed by :func:`add_indicators`.

    Returns:
        Mapping of column name to the latest float value.

    Raises:
        ValueError: If the frame is empty.
    """
    if df is None or df.empty:
        raise ValueError("No data available to read latest indicators from.")

    last = df.iloc[-1]
    return {col: float(last[col]) for col in df.columns}
