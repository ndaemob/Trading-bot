"""Central configuration for the AI Trading Analyst Bot.

All tunable constants live here so the rest of the codebase stays free of
magic numbers and the behaviour of the bot can be adjusted in a single place.

Nothing in this module performs any I/O or trading action. It only holds
plain, well-documented values.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
DEFAULT_TICKERS: list[str] = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]
"""Tickers analysed when the user does not supply their own."""

DEFAULT_PERIOD: str = "2y"
"""Default look-back window passed to yfinance (e.g. ``1y``, ``2y``, ``5y``)."""

DEFAULT_INTERVAL: str = "1d"
"""Default candle interval. Daily data keeps the MVP simple and robust."""

# --------------------------------------------------------------------------- #
# Indicator parameters
# --------------------------------------------------------------------------- #
RSI_PERIOD: int = 14
SMA_SHORT: int = 20
SMA_MEDIUM: int = 50
SMA_LONG: int = 200
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
ATR_PERIOD: int = 14

# --------------------------------------------------------------------------- #
# Strategy thresholds
# --------------------------------------------------------------------------- #
RSI_BUY_LOW: float = 40.0
"""Lower bound of the healthy RSI band that supports a BUY."""

RSI_BUY_HIGH: float = 70.0
"""Upper bound of the healthy RSI band. Above this we only WATCH."""

RSI_OVERBOUGHT: float = 80.0
"""RSI above this is treated as extremely overbought -> SELL."""

# --------------------------------------------------------------------------- #
# Entry zone / stop-loss sizing (expressed as multiples of ATR)
# --------------------------------------------------------------------------- #
ENTRY_ZONE_LOW_ATR: float = 1.5
"""Bottom of the suggested entry zone = close - this many ATRs."""

ENTRY_ZONE_HIGH_ATR: float = 0.5
"""Top of the suggested entry zone = close - this many ATRs."""

STOP_LOSS_ATR: float = 2.0
"""Suggested stop-loss = close - this many ATRs."""

# --------------------------------------------------------------------------- #
# Risk management
# --------------------------------------------------------------------------- #
DEFAULT_PORTFOLIO_VALUE: float = 10_000.0
"""Hypothetical starting capital used for sizing and backtests."""

MAX_RISK_PER_TRADE: float = 0.02
"""Never risk more than 2% of the portfolio on a single trade."""

MAX_ALLOCATION_PER_STOCK: float = 0.20
"""Never allocate more than 20% of the portfolio to a single stock."""

# Leverage is intentionally disabled and is never recommended by this bot.
ALLOW_LEVERAGE: bool = False

# --------------------------------------------------------------------------- #
# Backtest realism — costs applied to simulated fills
# --------------------------------------------------------------------------- #
COMMISSION_PCT: float = 0.0005
"""Commission charged per trade as a fraction of notional (0.05%)."""

SLIPPAGE_PCT: float = 0.0005
"""Adverse slippage applied to each simulated fill (0.05%)."""

# --------------------------------------------------------------------------- #
# Data loading robustness
# --------------------------------------------------------------------------- #
USE_CACHE: bool = True
"""Reuse a same-day local cache instead of re-downloading when possible."""

CACHE_DIR: str = "data_cache"
"""Directory for cached CSV downloads (git-ignored)."""

MAX_RETRIES: int = 3
"""How many times to attempt a download before giving up."""

RETRY_BACKOFF: float = 2.0
"""Base seconds for exponential backoff between download retries."""

# --------------------------------------------------------------------------- #
# Paper trading & output
# --------------------------------------------------------------------------- #
PAPER_PORTFOLIO_FILE: str = "paper_portfolio.json"
"""Default JSON file persisting the simulated paper-trading portfolio."""

OUTPUT_DIR: str = "output"
"""Directory for generated charts and exports (git-ignored)."""

# --------------------------------------------------------------------------- #
# Extended indicators
# --------------------------------------------------------------------------- #
EMA_FAST: int = 12
EMA_SLOW: int = 26
BB_PERIOD: int = 20
BB_STD: float = 2.0
ADX_PERIOD: int = 14
STOCH_PERIOD: int = 14
STOCH_SMOOTH: int = 3
VOLUME_SMA: int = 20

# Trend is considered "strong" (high conviction) above this ADX level.
ADX_TREND_STRENGTH: float = 25.0

# --------------------------------------------------------------------------- #
# Performance metrics
# --------------------------------------------------------------------------- #
TRADING_DAYS_PER_YEAR: int = 252
RISK_FREE_RATE: float = 0.0
"""Annual risk-free rate used in Sharpe/Sortino (kept simple at 0)."""

# --------------------------------------------------------------------------- #
# Multi-factor signal scoring — weights must sum to 1.0
# --------------------------------------------------------------------------- #
FACTOR_WEIGHTS: dict[str, float] = {
    "trend": 0.35,
    "momentum": 0.30,
    "participation": 0.15,
    "quality": 0.20,
}
