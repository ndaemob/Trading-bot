"""Historical market-data loading via :mod:`yfinance`.

This module is the *only* place the bot reaches out to the network. It never
places orders or touches a brokerage — it just downloads read-only price
history and hands back a clean :class:`pandas.DataFrame`.
"""

from __future__ import annotations

import pandas as pd

from . import config


class DataLoadError(RuntimeError):
    """Raised when market data cannot be downloaded or is unusable."""


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with predictable, capitalised OHLCV column names.

    ``yfinance`` may return either a flat or a :class:`~pandas.MultiIndex`
    column layout (the latter when several tickers are requested). We collapse
    any MultiIndex to its first level so downstream code can always rely on
    ``Open``/``High``/``Low``/``Close``/``Volume``.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def load_data(
    ticker: str,
    period: str = config.DEFAULT_PERIOD,
    interval: str = config.DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """Download historical OHLCV data for a single ``ticker``.

    Args:
        ticker: Stock symbol, e.g. ``"AAPL"``.
        period: yfinance period string (default: 2 years).
        interval: yfinance candle interval (default: daily).

    Returns:
        A DataFrame indexed by date with ``Open``, ``High``, ``Low``,
        ``Close`` and ``Volume`` columns.

    Raises:
        DataLoadError: If the symbol is invalid, the download fails, or the
            returned data is empty.
    """
    if not ticker or not ticker.strip():
        raise DataLoadError("Ticker symbol must be a non-empty string.")

    ticker = ticker.strip().upper()

    # Imported lazily so that unit tests (which feed synthetic data) do not
    # require the network or the yfinance package to be installed.
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DataLoadError(
            "yfinance is not installed. Run 'pip install -r requirements.txt'."
        ) from exc

    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
    except Exception as exc:  # noqa: BLE001 - surface any network/API failure
        raise DataLoadError(f"Failed to download data for {ticker}: {exc}") from exc

    if df is None or df.empty:
        raise DataLoadError(
            f"No data returned for {ticker}. The symbol may be invalid or "
            f"delisted, or the data provider may be unavailable."
        )

    df = _normalise_columns(df)
    validate_data(df, ticker)
    return df


def validate_data(df: pd.DataFrame, ticker: str = "") -> None:
    """Validate that ``df`` looks like usable OHLCV data.

    Args:
        df: Candidate price DataFrame.
        ticker: Optional symbol, used only to produce clearer error messages.

    Raises:
        DataLoadError: If the frame is empty or missing required columns.
    """
    label = f" for {ticker}" if ticker else ""

    if df is None or df.empty:
        raise DataLoadError(f"Received empty data{label}.")

    required = {"Open", "High", "Low", "Close"}
    missing = required.difference(df.columns)
    if missing:
        raise DataLoadError(
            f"Data{label} is missing required columns: {sorted(missing)}."
        )

    if df["Close"].dropna().empty:
        raise DataLoadError(f"Data{label} contains no usable closing prices.")
