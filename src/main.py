"""Command-line entry point for the AI Trading Analyst Bot.

Usage examples::

    python -m src.main                      # analyse the default tickers
    python -m src.main AAPL MSFT            # analyse specific tickers
    python -m src.main NVDA --backtest      # also run a historical backtest
    python -m src.main AAPL --period 5y --portfolio 25000

The bot only *analyses* — it never connects to a broker or places a trade.
"""

from __future__ import annotations

import argparse
import sys

from . import backtest, config, data_loader, indicators, risk, strategy
from .strategy import Signal

DISCLAIMER = (
    "DISCLAIMER: Educational tool only. Not financial advice. "
    "No real trades are executed."
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="ai-trading-bot",
        description="Safe, data-driven stock analysis. Never executes trades.",
        epilog=DISCLAIMER,
    )
    parser.add_argument(
        "tickers",
        nargs="*",
        default=config.DEFAULT_TICKERS,
        help=f"Ticker symbols to analyse (default: {', '.join(config.DEFAULT_TICKERS)}).",
    )
    parser.add_argument(
        "--period", default=config.DEFAULT_PERIOD,
        help=f"History window, e.g. 1y/2y/5y (default: {config.DEFAULT_PERIOD}).",
    )
    parser.add_argument(
        "--interval", default=config.DEFAULT_INTERVAL,
        help=f"Candle interval (default: {config.DEFAULT_INTERVAL}).",
    )
    parser.add_argument(
        "--portfolio", type=float, default=config.DEFAULT_PORTFOLIO_VALUE,
        help=f"Hypothetical portfolio value (default: {config.DEFAULT_PORTFOLIO_VALUE:.0f}).",
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Also run a simple historical backtest for each ticker.",
    )
    return parser


def format_signal(sig: Signal, sizing: risk.PositionSize | None = None) -> str:
    """Render a :class:`~src.strategy.Signal` as a readable report block."""
    lines: list[str] = []
    lines.append(f"Ticker: {sig.ticker}")
    lines.append(f"Signal: {sig.signal}")
    lines.append(f"Confidence: {sig.confidence}")
    lines.append(f"Latest Close: {sig.latest_close:.2f}")

    lines.append("Reasons:")
    for reason in sig.reasons or ["No notable signals"]:
        lines.append(f"- {reason}")

    lines.append("Risks:")
    for risk_item in sig.risks or ["None noted"]:
        lines.append(f"- {risk_item}")

    if sig.entry_zone is not None:
        low, high = sig.entry_zone
        lines.append("Suggested Entry Zone:")
        lines.append(f"{low:.2f} - {high:.2f}")

    if sig.stop_loss is not None:
        lines.append("Suggested Stop Loss:")
        lines.append(f"{sig.stop_loss:.2f}")

    if sizing is not None:
        lines.append("Suggested Position (risk-managed):")
        lines.append(
            f"{sizing.shares} shares (~{sizing.allocation_pct:.1%} of portfolio, "
            f"risking {sizing.risk_pct:.2%})"
        )
        for warning in sizing.warnings:
            lines.append(f"! {warning}")

    return "\n".join(lines)


def analyse_ticker(
    ticker: str,
    period: str,
    interval: str,
    portfolio_value: float,
    do_backtest: bool,
) -> bool:
    """Analyse a single ticker and print the result.

    Returns:
        ``True`` on success, ``False`` if the ticker could not be analysed.
    """
    try:
        df = data_loader.load_data(ticker, period=period, interval=interval)
        df = indicators.add_indicators(df)
        sig = strategy.generate_signal(df, ticker)
    except (data_loader.DataLoadError, ValueError) as exc:
        print(f"\nTicker: {ticker.upper()}")
        print(f"  Could not analyse: {exc}")
        return False

    # Only size a position when there is a tradeable long setup with a stop.
    sizing: risk.PositionSize | None = None
    if sig.signal in (strategy.BUY, strategy.WATCH) and sig.stop_loss:
        entry = sig.entry_zone[1] if sig.entry_zone else sig.latest_close
        try:
            sizing = risk.calculate_position_size(
                portfolio_value=portfolio_value,
                entry_price=entry,
                stop_loss=sig.stop_loss,
                ticker=ticker,
            )
        except ValueError:
            sizing = None

    print()
    print(format_signal(sig, sizing))

    if do_backtest:
        try:
            result = backtest.run_backtest(df, ticker, initial_capital=portfolio_value)
            print()
            print(backtest.format_result(result))
        except ValueError as exc:
            print(f"  Backtest unavailable: {exc}")

    return True


def main(argv: list[str] | None = None) -> int:
    """Program entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    tickers = [t.upper() for t in (args.tickers or config.DEFAULT_TICKERS)]

    print("=" * 60)
    print("AI Trading Analyst Bot — analysis only, no trades executed")
    print("=" * 60)

    analysed = 0
    for ticker in tickers:
        if analyse_ticker(
            ticker, args.period, args.interval, args.portfolio, args.backtest
        ):
            analysed += 1
        print("-" * 60)

    print(DISCLAIMER)
    # Non-zero exit only if *nothing* could be analysed (e.g. no network).
    return 0 if analysed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
