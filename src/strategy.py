"""Signal-generation strategy.

Turns indicator values into a single discrete recommendation
(``BUY`` / ``HOLD`` / ``WATCH`` / ``SELL``) together with a confidence score,
human-readable reasons, risks, and suggested entry / stop levels.

The recommendations are *probabilistic and educational*. They are not financial
advice and the bot never executes a trade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from . import config

# The only labels the strategy is ever allowed to emit.
BUY = "BUY"
HOLD = "HOLD"
WATCH = "WATCH"
SELL = "SELL"
VALID_SIGNALS: tuple[str, ...] = (BUY, HOLD, WATCH, SELL)


@dataclass
class Signal:
    """A structured trading signal for a single ticker.

    Attributes:
        ticker: The analysed symbol.
        signal: One of :data:`VALID_SIGNALS`.
        confidence: Integer 0-100 expressing conviction in the signal.
        latest_close: Most recent closing price.
        reasons: Bullet-point rationale supporting the signal.
        risks: Bullet-point caveats the user should keep in mind.
        stop_loss: Suggested protective stop-loss price (``None`` if N/A).
        entry_zone: ``(low, high)`` suggested accumulation band (``None`` if N/A).
    """

    ticker: str
    signal: str
    confidence: int
    latest_close: float
    reasons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    stop_loss: float | None = None
    entry_zone: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.signal not in VALID_SIGNALS:
            raise ValueError(f"Invalid signal {self.signal!r}; expected one of {VALID_SIGNALS}.")
        # Confidence is always clamped to the documented 0-100 range.
        self.confidence = int(max(0, min(100, self.confidence)))


def classify(sma50: float, sma200: float, rsi: float) -> str:
    """Map raw indicator values to a discrete signal label.

    This is the heart of the strategy and is deliberately tiny and pure so it
    can be reused by the backtester and exercised directly in tests.

    Decision order (first match wins):

    1. **SELL** if the trend is down (``SMA50 < SMA200``) *or* RSI is extremely
       overbought (``> 80``).
    2. **BUY** if the trend is up *and* RSI sits in the healthy 40-70 band.
    3. **WATCH** if the trend is up but RSI is elevated (70-80).
    4. **HOLD** otherwise.

    Args:
        sma50: 50-period simple moving average.
        sma200: 200-period simple moving average.
        rsi: Relative Strength Index (0-100).

    Returns:
        One of :data:`VALID_SIGNALS`. Returns ``HOLD`` if any input is NaN,
        since we cannot responsibly act on incomplete data.
    """
    if any(math.isnan(x) for x in (sma50, sma200, rsi)):
        return HOLD

    trend_up = sma50 > sma200

    if not trend_up or rsi > config.RSI_OVERBOUGHT:
        return SELL
    if config.RSI_BUY_LOW <= rsi <= config.RSI_BUY_HIGH:
        return BUY
    if rsi > config.RSI_BUY_HIGH:
        return WATCH
    return HOLD


def _confidence(signal: str, sma50: float, sma200: float, rsi: float, macd_hist: float) -> int:
    """Heuristic 0-100 confidence score for a signal.

    Combines trend separation (how far SMA50 is above/below SMA200), how
    comfortably RSI sits in its ideal band, and MACD momentum. The result is
    intentionally bounded and never expresses certainty.
    """
    # Trend strength as a percentage gap between the moving averages.
    trend_gap = 0.0
    if sma200 and not math.isnan(sma200):
        trend_gap = (sma50 - sma200) / sma200

    # Scale a ~10% gap to a full strength contribution.
    trend_strength = max(-1.0, min(1.0, trend_gap / 0.10))
    momentum = 1.0 if (not math.isnan(macd_hist) and macd_hist > 0) else -1.0

    if signal == BUY:
        # Reward a healthy uptrend and an RSI near the middle of the band.
        rsi_center = 1.0 - abs(rsi - 55.0) / 15.0  # 1.0 at RSI 55, 0 at 40/70
        score = 50 + 25 * trend_strength + 15 * max(0.0, rsi_center) + 10 * momentum
    elif signal == WATCH:
        # Positive trend but stretched: moderate conviction, capped.
        score = 45 + 20 * trend_strength + 5 * momentum
    elif signal == SELL:
        # Stronger conviction the deeper the downtrend / the higher the RSI.
        overbought = max(0.0, (rsi - config.RSI_BUY_HIGH) / 30.0)
        score = 50 + 25 * (-trend_strength) + 20 * overbought - 5 * momentum
    else:  # HOLD
        score = 40

    return int(max(0, min(100, round(score))))


def _levels(close: float, atr: float) -> tuple[float | None, tuple[float, float] | None]:
    """Compute a suggested stop-loss and entry zone from price and ATR.

    Returns ``(None, None)`` when ATR is unavailable so we never emit
    nonsensical levels.
    """
    if math.isnan(atr) or atr <= 0 or math.isnan(close):
        return None, None

    stop_loss = round(close - config.STOP_LOSS_ATR * atr, 2)
    entry_low = round(close - config.ENTRY_ZONE_LOW_ATR * atr, 2)
    entry_high = round(close - config.ENTRY_ZONE_HIGH_ATR * atr, 2)
    # Guard against negative prices on very volatile / low-priced names.
    stop_loss = max(0.0, stop_loss)
    entry_low = max(0.0, entry_low)
    entry_high = max(0.0, entry_high)
    return stop_loss, (entry_low, entry_high)


def generate_signal(df: pd.DataFrame, ticker: str) -> Signal:
    """Produce a :class:`Signal` from a DataFrame of indicators.

    Args:
        df: DataFrame already processed by
            :func:`indicators.add_indicators`. Must contain ``Close``,
            ``SMA_50``, ``SMA_200``, ``RSI`` and (ideally) ``ATR``.
        ticker: Symbol being analysed (echoed back in the result).

    Returns:
        A fully populated :class:`Signal`.

    Raises:
        ValueError: If ``df`` is empty or missing required indicator columns.
    """
    if df is None or df.empty:
        raise ValueError(f"No data to generate a signal for {ticker}.")

    required = {"Close", "SMA_50", "SMA_200", "RSI"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing indicator columns: {sorted(missing)}")

    last = df.iloc[-1]
    close = float(last["Close"])
    sma50 = float(last["SMA_50"])
    sma200 = float(last["SMA_200"])
    rsi = float(last["RSI"])
    atr = float(last["ATR"]) if "ATR" in df.columns else float("nan")
    macd_hist = float(last["MACD_Hist"]) if "MACD_Hist" in df.columns else float("nan")

    signal = classify(sma50, sma200, rsi)
    confidence = _confidence(signal, sma50, sma200, rsi, macd_hist)
    stop_loss, entry_zone = _levels(close, atr)

    reasons, risks = _explain(signal, sma50, sma200, rsi, macd_hist)

    # SELL is an exit signal — entry levels would be misleading, so drop them.
    if signal == SELL:
        entry_zone = None

    return Signal(
        ticker=ticker.upper(),
        signal=signal,
        confidence=confidence,
        latest_close=round(close, 2),
        reasons=reasons,
        risks=risks,
        stop_loss=stop_loss,
        entry_zone=entry_zone,
    )


def _explain(
    signal: str, sma50: float, sma200: float, rsi: float, macd_hist: float
) -> tuple[list[str], list[str]]:
    """Build human-readable ``(reasons, risks)`` lists for a signal."""
    reasons: list[str] = []
    risks: list[str] = []

    trend_up = not math.isnan(sma50) and not math.isnan(sma200) and sma50 > sma200
    if trend_up:
        reasons.append("SMA50 above SMA200 (long-term uptrend)")
    elif not math.isnan(sma50) and not math.isnan(sma200):
        reasons.append("SMA50 below SMA200 (long-term downtrend)")

    if not math.isnan(rsi):
        if rsi > config.RSI_OVERBOUGHT:
            reasons.append(f"RSI very high at {rsi:.0f} (extremely overbought)")
            risks.append("Extremely overbought; sharp pullback risk")
        elif rsi > config.RSI_BUY_HIGH:
            reasons.append(f"RSI elevated at {rsi:.0f}")
            risks.append("Momentum may be overheated")
        elif rsi < config.RSI_BUY_LOW:
            reasons.append(f"RSI weak at {rsi:.0f}")
            risks.append("Weak momentum; trend may not be supported")
        else:
            reasons.append(f"RSI healthy at {rsi:.0f}")

    if not math.isnan(macd_hist):
        if macd_hist > 0:
            reasons.append("MACD histogram positive (upward momentum)")
        else:
            reasons.append("MACD histogram negative (downward momentum)")
            risks.append("MACD momentum is fading or negative")

    if signal == SELL:
        risks.append("Signal favours exiting or avoiding new long exposure")
    if signal == WATCH:
        risks.append("Wait for a healthier entry; do not chase strength")
    if signal == HOLD:
        risks.append("No clear edge; conditions are mixed")

    # Universal caveat — every signal is probabilistic, not a guarantee.
    risks.append("Not financial advice; signals are probabilistic only")
    return reasons, risks


def signal_series(df: pd.DataFrame) -> pd.Series:
    """Compute the strategy label for *every* row of ``df``.

    Used by the backtester to replay the strategy through history. Rows where
    the required indicators are still warming up resolve to ``HOLD`` via
    :func:`classify`.

    Args:
        df: DataFrame processed by :func:`indicators.add_indicators`.

    Returns:
        A ``pandas.Series`` of signal labels aligned to ``df``'s index.
    """
    required = {"SMA_50", "SMA_200", "RSI"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing indicator columns: {sorted(missing)}")

    labels = [
        classify(float(row.SMA_50), float(row.SMA_200), float(row.RSI))
        for row in df.itertuples(index=False)
    ]
    return pd.Series(labels, index=df.index, name="signal")
