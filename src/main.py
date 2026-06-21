"""Command-line entry point for the AI Trading Analyst Bot.

Usage examples::

    python -m src.main                      # analyse the default tickers
    python -m src.main AAPL MSFT            # analyse specific tickers
    python -m src.main NVDA --backtest      # also run a historical backtest
    python -m src.main NVDA --chart         # also save price/RSI charts
    python -m src.main AAPL --paper         # apply signals to a paper portfolio
    python -m src.main AAPL --period 5y --portfolio 25000

The bot only *analyses* and *simulates* — it never connects to a broker or
places a real trade.
"""

from __future__ import annotations

import argparse
import sys

from . import (
    backtest,
    charts,
    config,
    data_loader,
    indicators,
    paper_trading,
    risk,
    strategy,
)
from .strategy import Signal

DISCLAIMER = (
    "DISCLAIMER: Educational tool only. Not financial advice. " "No real trades are executed."
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
        "--period",
        default=config.DEFAULT_PERIOD,
        help=f"History window, e.g. 1y/2y/5y (default: {config.DEFAULT_PERIOD}).",
    )
    parser.add_argument(
        "--interval",
        default=config.DEFAULT_INTERVAL,
        help=f"Candle interval (default: {config.DEFAULT_INTERVAL}).",
    )
    parser.add_argument(
        "--portfolio",
        type=float,
        default=config.DEFAULT_PORTFOLIO_VALUE,
        help=f"Hypothetical portfolio value (default: {config.DEFAULT_PORTFOLIO_VALUE:.0f}).",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Also run a simple historical backtest for each ticker.",
    )
    parser.add_argument(
        "--risk-sizing",
        action="store_true",
        help="Size backtest positions via risk rules instead of all-in.",
    )
    parser.add_argument(
        "--chart",
        action="store_true",
        help="Save price/RSI (and equity, with --backtest) charts to the output dir.",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Apply each signal to a persistent simulated paper portfolio.",
    )
    parser.add_argument(
        "--paper-file",
        default=config.PAPER_PORTFOLIO_FILE,
        help="Path to the paper-portfolio JSON file.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the local data cache and always download fresh.",
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


def _size_position(sig: Signal, portfolio_value: float) -> risk.PositionSize | None:
    """Compute a risk-managed position size for a tradeable long signal."""
    if sig.signal not in (strategy.BUY, strategy.WATCH) or not sig.stop_loss:
        return None
    entry = sig.entry_zone[1] if sig.entry_zone else sig.latest_close
    try:
        return risk.calculate_position_size(
            portfolio_value=portfolio_value,
            entry_price=entry,
            stop_loss=sig.stop_loss,
            ticker=sig.ticker,
        )
    except ValueError:
        return None


def analyse_ticker(
    ticker: str,
    args: argparse.Namespace,
    portfolio: paper_trading.PaperPortfolio | None,
) -> bool:
    """Analyse a single ticker and print the result.

    Returns:
        ``True`` on success, ``False`` if the ticker could not be analysed.
    """
    try:
        df = data_loader.load_data(
            ticker,
            period=args.period,
            interval=args.interval,
            use_cache=not args.no_cache,
        )
        df = indicators.add_indicators(df)
        sig = strategy.generate_signal(df, ticker)
    except (data_loader.DataLoadError, ValueError) as exc:
        print(f"\nTicker: {ticker.upper()}")
        print(f"  Could not analyse: {exc}")
        return False

    sizing = _size_position(sig, args.portfolio)

    print()
    print(format_signal(sig, sizing))

    if args.chart:
        try:
            path = charts.plot_price(df, ticker)
            print(f"Chart saved: {path}")
        except (ValueError, OSError) as exc:
            print(f"  Chart unavailable: {exc}")

    if portfolio is not None:
        shares = float(sizing.shares) if sizing else 0.0
        print(paper_trading.execute_signal(portfolio, sig, shares))

    if args.backtest:
        try:
            result = backtest.run_backtest(
                df,
                ticker,
                initial_capital=args.portfolio,
                commission_pct=config.COMMISSION_PCT,
                slippage_pct=config.SLIPPAGE_PCT,
                position_sizing="risk_based" if args.risk_sizing else "all_in",
            )
            print()
            print(backtest.format_result(result))
            if args.chart:
                try:
                    print(f"Chart saved: {charts.plot_backtest(result)}")
                except (ValueError, OSError) as exc:
                    print(f"  Equity chart unavailable: {exc}")
        except ValueError as exc:
            print(f"  Backtest unavailable: {exc}")

    return True


def _print_portfolio(portfolio: paper_trading.PaperPortfolio) -> None:
    """Print a summary of the simulated paper portfolio."""
    print("-" * 60)
    print("Paper Portfolio (simulated — no real trades):")
    print(f"  Cash: {portfolio.cash:,.2f}")
    if portfolio.positions:
        print("  Positions:")
        for pos in portfolio.positions.values():
            print(f"    {pos.ticker}: {pos.shares:g} @ avg {pos.avg_price:.2f}")
    else:
        print("  Positions: none")
    value = portfolio.market_value({})
    print(f"  Approx. value (positions at avg cost): {value:,.2f}")


def main(argv: list[str] | None = None) -> int:
    """Program entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    tickers = [t.upper() for t in (args.tickers or config.DEFAULT_TICKERS)]

    portfolio: paper_trading.PaperPortfolio | None = None
    if args.paper:
        portfolio = paper_trading.load_portfolio(args.paper_file, initial_capital=args.portfolio)

    print("=" * 60)
    print("AI Trading Analyst Bot — analysis only, no trades executed")
    print("=" * 60)

    analysed = 0
    for ticker in tickers:
        if analyse_ticker(ticker, args, portfolio):
            analysed += 1
        print("-" * 60)

    if portfolio is not None:
        paper_trading.save_portfolio(portfolio, args.paper_file)
        _print_portfolio(portfolio)

    print(DISCLAIMER)
    # Non-zero exit only if *nothing* could be analysed (e.g. no network).
    return 0 if analysed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
