"""Historical market-data loading via :mod:`yfinance`.

This module is the *only* place the bot reaches out to the network. It never
places orders or touches a brokerage — it just downloads read-only price
history and hands back a clean :class:`pandas.DataFrame`.

For robustness it adds:

* **Retries** with exponential backoff around the download, and
* an optional **same-day CSV cache** so repeated runs avoid re-downloading.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

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


def _download_raw(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Perform a single raw download from yfinance.

    Isolated into its own function so tests can monkeypatch it without any
    network access, and so the retry logic in :func:`load_data` stays readable.

    Raises:
        DataLoadError: If yfinance is unavailable or the request errors.
    """
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
    return _normalise_columns(df)


def _cache_path(cache_dir: str, ticker: str, period: str, interval: str) -> Path:
    """Return the CSV cache path for a given request."""
    safe = f"{ticker}_{period}_{interval}".replace("/", "-")
    return Path(cache_dir) / f"{safe}.csv"


def _is_fresh(path: Path) -> bool:
    """``True`` if ``path`` exists and was modified on the current UTC day."""
    if not path.exists():
        return False
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return modified.date() == datetime.now(timezone.utc).date()


def _read_cache(path: Path) -> pd.DataFrame:
    """Load a cached CSV, restoring the datetime index."""
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    """Persist ``df`` to ``path`` as CSV, creating parent dirs as needed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
    except OSError:
        # Caching is a best-effort optimisation; never fail the load over it.
        pass


def load_data(
    ticker: str,
    period: str = config.DEFAULT_PERIOD,
    interval: str = config.DEFAULT_INTERVAL,
    *,
    use_cache: bool = config.USE_CACHE,
    cache_dir: str = config.CACHE_DIR,
    max_retries: int = config.MAX_RETRIES,
    retry_backoff: float = config.RETRY_BACKOFF,
) -> pd.DataFrame:
    """Download historical OHLCV data for a single ``ticker``.

    Args:
        ticker: Stock symbol, e.g. ``"AAPL"``.
        period: yfinance period string (default: 2 years).
        interval: yfinance candle interval (default: daily).
        use_cache: Reuse a same-day local CSV cache when available.
        cache_dir: Directory holding the CSV cache.
        max_retries: Number of download attempts before giving up.
        retry_backoff: Base seconds for exponential backoff between retries.

    Returns:
        A DataFrame indexed by date with ``Open``, ``High``, ``Low``,
        ``Close`` and ``Volume`` columns.

    Raises:
        DataLoadError: If the symbol is invalid, the download fails after all
            retries, or the returned data is empty.
    """
    if not ticker or not ticker.strip():
        raise DataLoadError("Ticker symbol must be a non-empty string.")

    ticker = ticker.strip().upper()
    path = _cache_path(cache_dir, ticker, period, interval)

    if use_cache and _is_fresh(path):
        try:
            cached = _read_cache(path)
            validate_data(cached, ticker)
            return cached
        except (DataLoadError, ValueError, OSError):
            # A corrupt cache should never block a fresh download.
            pass

    last_error: DataLoadError | None = None
    for attempt in range(1, max(1, max_retries) + 1):
        try:
            df = _download_raw(ticker, period, interval)
            validate_data(df, ticker)
            if use_cache:
                _write_cache(path, df)
            return df
        except DataLoadError as exc:
            last_error = exc
            if attempt < max_retries:
                # Exponential backoff: 2s, 4s, 8s, ...
                time.sleep(retry_backoff * (2 ** (attempt - 1)))

    assert last_error is not None  # loop runs at least once
    raise last_error


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
        raise DataLoadError(f"Data{label} is missing required columns: {sorted(missing)}.")

    if df["Close"].dropna().empty:
        raise DataLoadError(f"Data{label} contains no usable closing prices.")
