"""Chart rendering with matplotlib.

Generates PNG charts for visual inspection: a price panel with moving averages
and an RSI sub-panel, plus an equity-curve panel for backtests.

A non-interactive backend (``Agg``) is selected so charts render to files on
headless machines and in CI without a display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402  (intentional: after backend set)
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .backtest import BacktestResult  # noqa: E402


def _ensure_dir(path: str) -> Path:
    """Create ``path`` if needed and return it as a :class:`Path`."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def plot_price(
    df: pd.DataFrame,
    ticker: str,
    output_dir: str = config.OUTPUT_DIR,
) -> str:
    """Render a price + moving-average chart with an RSI sub-panel.

    Args:
        df: DataFrame processed by :func:`indicators.add_indicators`.
        ticker: Symbol, used in the title and filename.
        output_dir: Directory to write the PNG into.

    Returns:
        The path to the written PNG file.

    Raises:
        ValueError: If ``df`` is empty or lacks a ``Close`` column.
    """
    if df is None or df.empty or "Close" not in df.columns:
        raise ValueError("plot_price requires a non-empty frame with 'Close'.")

    directory = _ensure_dir(output_dir)
    fig, (ax_price, ax_rsi) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_price.plot(df.index, df["Close"], label="Close", linewidth=1.2)
    for col, style in (("SMA_20", "--"), ("SMA_50", "-."), ("SMA_200", ":")):
        if col in df.columns:
            ax_price.plot(df.index, df[col], style, label=col, linewidth=1.0)
    ax_price.set_title(f"{ticker.upper()} — Price & Moving Averages")
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left")
    ax_price.grid(True, alpha=0.3)

    if "RSI" in df.columns:
        ax_rsi.plot(df.index, df["RSI"], color="purple", linewidth=1.0, label="RSI")
        ax_rsi.axhline(70, color="red", linestyle="--", alpha=0.5)
        ax_rsi.axhline(30, color="green", linestyle="--", alpha=0.5)
        ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.set_xlabel("Date")
    ax_rsi.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = directory / f"{ticker.upper()}_price.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return str(out_path)


def plot_backtest(
    result: BacktestResult,
    output_dir: str = config.OUTPUT_DIR,
) -> str:
    """Render the equity curve of a backtest result.

    Args:
        result: A :class:`~src.backtest.BacktestResult` with an equity curve.
        output_dir: Directory to write the PNG into.

    Returns:
        The path to the written PNG file.

    Raises:
        ValueError: If the result has no equity curve to plot.
    """
    if result.equity_curve is None or result.equity_curve.empty:
        raise ValueError("BacktestResult has no equity curve to plot.")

    directory = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(result.equity_curve.index, result.equity_curve, label="Equity", linewidth=1.2)
    ax.axhline(
        result.initial_capital,
        color="grey",
        linestyle="--",
        alpha=0.6,
        label="Initial Capital",
    )
    ax.set_title(
        f"{result.ticker or 'Backtest'} — Equity Curve "
        f"(Total Return {result.total_return:+.1%})"
    )
    ax.set_ylabel("Portfolio Value")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = directory / f"{result.ticker or 'backtest'}_equity.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return str(out_path)
