"""Signal-generation strategy.

Turns indicator values into a single discrete recommendation
(``BUY`` / ``HOLD`` / ``WATCH`` / ``SELL``) together with a confidence score,
human-readable reasons, risks, and suggested entry / stop levels.

The discrete label comes from a tiny, auditable rule (:func:`classify`). The
*confidence* behind it comes from a transparent **multi-factor model** that
blends trend, momentum, participation (volume) and quality (over-extension)
into a single directional score — every factor is exposed on the resulting
:class:`Signal` so the reasoning is fully inspectable.

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
        score: Directional multi-factor score in ``[-1, 1]`` (bullish positive).
        factors: Per-factor sub-scores in ``[-1, 1]`` (the score's breakdown).
    """

    ticker: str
    signal: str
    confidence: int
    latest_close: float
    reasons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    stop_loss: float | None = None
    entry_zone: tuple[float, float] | None = None
    score: float = 0.0
    factors: dict[str, float] = field(default_factory=dict)

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


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clamp ``x`` to ``[lo, hi]`` (NaN becomes 0.0)."""
    if math.isnan(x):
        return 0.0
    return max(lo, min(hi, x))


def _val(values: dict[str, float], key: str) -> float:
    """Read ``key`` from a values dict, defaulting to NaN when absent."""
    v = values.get(key, float("nan"))
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def factor_scores(values: dict[str, float]) -> dict[str, float]:
    """Compute four bullish-positive factor scores in ``[-1, 1]``.

    Factors (each ``-1`` very bearish … ``+1`` very bullish):

    * ``trend`` — SMA50 vs SMA200 and SMA20 vs SMA50, amplified by ADX strength.
    * ``momentum`` — MACD histogram (scaled by ATR), RSI and Stochastic.
    * ``participation`` — volume vs its average, signed by momentum direction.
    * ``quality`` — penalises over-extension (high RSI/Stochastic, price above
      the upper Bollinger band); higher means "more room to run".

    Missing indicators degrade gracefully to a neutral ``0.0`` contribution.
    """
    sma20 = _val(values, "SMA_20")
    sma50 = _val(values, "SMA_50")
    sma200 = _val(values, "SMA_200")
    rsi = _val(values, "RSI")
    macd_hist = _val(values, "MACD_Hist")
    atr = _val(values, "ATR")
    close = _val(values, "Close")
    adx = _val(values, "ADX")
    stoch_k = _val(values, "Stoch_K")
    bb_pct = _val(values, "BB_Pct")
    volume = _val(values, "Volume")
    vol_sma = _val(values, "Vol_SMA_20")

    # --- trend -------------------------------------------------------------
    long_trend = 0.0
    if not math.isnan(sma50) and not math.isnan(sma200) and sma200 != 0:
        long_trend = _clamp(((sma50 - sma200) / sma200) / 0.08)
    med_trend = 0.0
    if not math.isnan(sma20) and not math.isnan(sma50) and sma50 != 0:
        med_trend = _clamp(((sma20 - sma50) / sma50) / 0.05)
    raw_trend = 0.6 * long_trend + 0.4 * med_trend
    adx_mult = 1.0
    if not math.isnan(adx):
        adx_mult = _clamp(adx / config.ADX_TREND_STRENGTH, 0.5, 1.2)
    trend = _clamp(raw_trend * adx_mult)

    # --- momentum ----------------------------------------------------------
    parts: list[float] = []
    if not math.isnan(macd_hist):
        scale = (0.5 * atr) if (not math.isnan(atr) and atr > 0) else max(close, 1.0) * 0.01
        parts.append(_clamp(macd_hist / scale))
    if not math.isnan(rsi):
        parts.append(_clamp((rsi - 50) / 15) if rsi <= 65 else _clamp((65 - rsi) / 15))
    if not math.isnan(stoch_k):
        parts.append(_clamp((stoch_k - 50) / 40))
    momentum = _clamp(sum(parts) / len(parts)) if parts else 0.0

    # --- participation (volume confirms direction) -------------------------
    participation = 0.0
    if not math.isnan(volume) and not math.isnan(vol_sma) and vol_sma > 0:
        vol_ratio = _clamp(volume / vol_sma - 1.0)
        direction = 1.0 if momentum >= 0 else -1.0
        participation = _clamp(vol_ratio * direction)

    # --- quality (lower when over-extended) --------------------------------
    if all(math.isnan(x) for x in (rsi, stoch_k, bb_pct)):
        quality = 0.0  # nothing to assess -> neutral
    else:
        quality = 0.3  # baseline: "not over-extended"
        if not math.isnan(rsi):
            quality -= 1.2 * max(0.0, (rsi - 70) / 30)
        if not math.isnan(stoch_k):
            quality -= 0.6 * max(0.0, (stoch_k - 80) / 20)
        if not math.isnan(bb_pct):
            quality -= 0.8 * max(0.0, bb_pct - 1.0)
        quality = _clamp(quality)

    return {
        "trend": round(trend, 3),
        "momentum": round(momentum, 3),
        "participation": round(participation, 3),
        "quality": round(quality, 3),
    }


def directional_score(factors: dict[str, float]) -> float:
    """Weighted blend of factor scores into one directional score in ``[-1, 1]``."""
    score = sum(config.FACTOR_WEIGHTS.get(name, 0.0) * value for name, value in factors.items())
    return _clamp(score)


def _confidence(signal: str, score: float) -> int:
    """Map the directional ``score`` to a 0-100 confidence for the given signal.

    Confidence rises as the multi-factor score aligns with the signal's
    direction (bullish for BUY/WATCH, bearish for SELL). It never expresses
    certainty.
    """
    if signal == BUY:
        raw = 50 + 50 * score
    elif signal == WATCH:
        raw = 45 + 35 * score
    elif signal == SELL:
        raw = 50 + 50 * (-score)
    else:  # HOLD — conviction is low precisely because the picture is mixed.
        raw = 45 - 25 * abs(score)
    return int(max(0, min(100, round(raw))))


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


def _row_values(df: pd.DataFrame) -> dict[str, float]:
    """Extract the latest row of ``df`` as a plain ``{column: float}`` dict."""
    last = df.iloc[-1]
    return {col: _val({col: last[col]}, col) for col in df.columns}


def generate_signal(df: pd.DataFrame, ticker: str) -> Signal:
    """Produce a :class:`Signal` from a DataFrame of indicators.

    Args:
        df: DataFrame already processed by
            :func:`indicators.add_indicators`. Must contain ``Close``,
            ``SMA_50``, ``SMA_200``, ``RSI`` and (ideally) the extended set.
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

    values = _row_values(df)
    close = _val(values, "Close")
    sma50 = _val(values, "SMA_50")
    sma200 = _val(values, "SMA_200")
    rsi = _val(values, "RSI")
    atr = _val(values, "ATR")

    signal = classify(sma50, sma200, rsi)
    factors = factor_scores(values)
    score = directional_score(factors)
    confidence = _confidence(signal, score)
    stop_loss, entry_zone = _levels(close, atr)
    reasons, risks = _explain(signal, values)

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
        score=round(score, 3),
        factors=factors,
    )


def _explain(signal: str, values: dict[str, float]) -> tuple[list[str], list[str]]:
    """Build human-readable ``(reasons, risks)`` lists from indicator values."""
    reasons: list[str] = []
    risks: list[str] = []

    sma50 = _val(values, "SMA_50")
    sma200 = _val(values, "SMA_200")
    rsi = _val(values, "RSI")
    macd_hist = _val(values, "MACD_Hist")
    adx = _val(values, "ADX")
    stoch_k = _val(values, "Stoch_K")
    bb_pct = _val(values, "BB_Pct")
    volume = _val(values, "Volume")
    vol_sma = _val(values, "Vol_SMA_20")

    if not math.isnan(sma50) and not math.isnan(sma200):
        if sma50 > sma200:
            reasons.append("SMA50 above SMA200 (long-term uptrend)")
        else:
            reasons.append("SMA50 below SMA200 (long-term downtrend)")

    if not math.isnan(adx):
        if adx >= config.ADX_TREND_STRENGTH:
            reasons.append(f"Strong trend (ADX {adx:.0f})")
        else:
            reasons.append(f"Weak or ranging trend (ADX {adx:.0f})")
            risks.append("Trend strength is low; signals less reliable in chop")

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

    if not math.isnan(stoch_k):
        if stoch_k > 80:
            risks.append(f"Stochastic overbought ({stoch_k:.0f})")
        elif stoch_k < 20:
            reasons.append(f"Stochastic oversold ({stoch_k:.0f}); possible bounce")

    if not math.isnan(bb_pct):
        if bb_pct > 1.0:
            risks.append("Price above upper Bollinger band (stretched)")
        elif bb_pct < 0.0:
            reasons.append("Price below lower Bollinger band (oversold)")

    if not math.isnan(volume) and not math.isnan(vol_sma) and vol_sma > 0:
        if volume > 1.2 * vol_sma:
            reasons.append("Above-average volume confirms the move")
        elif volume < 0.8 * vol_sma:
            risks.append("Below-average volume; weak conviction in the move")

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
