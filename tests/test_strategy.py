"""Tests for the signal-generation strategy."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from src import strategy
from src.strategy import VALID_SIGNALS, Signal, classify, generate_signal, signal_series

from .conftest import make_ohlcv


# --------------------------------------------------------------------------- #
# classify(): only ever emits the four allowed labels
# --------------------------------------------------------------------------- #
def test_classify_only_returns_valid_signals():
    """Across a wide grid of inputs, classify() stays within the allowed set."""
    smas = [40, 50, 60, 100, 150]
    rsis = [10, 30, 41, 55, 69, 71, 79, 81, 95]
    for sma50, sma200, rsi in itertools.product(smas, smas, rsis):
        assert classify(sma50, sma200, rsi) in VALID_SIGNALS


def test_classify_buy_branch():
    """Uptrend + healthy RSI -> BUY."""
    assert classify(sma50=110, sma200=100, rsi=55) == strategy.BUY


def test_classify_sell_on_downtrend():
    """Downtrend -> SELL regardless of RSI."""
    assert classify(sma50=90, sma200=100, rsi=55) == strategy.SELL


def test_classify_sell_on_extreme_overbought():
    """RSI above 80 -> SELL even in an uptrend."""
    assert classify(sma50=110, sma200=100, rsi=85) == strategy.SELL


def test_classify_watch_branch():
    """Uptrend but RSI elevated (70-80) -> WATCH."""
    assert classify(sma50=110, sma200=100, rsi=75) == strategy.WATCH


def test_classify_hold_branch():
    """Uptrend but weak RSI (<40) -> HOLD."""
    assert classify(sma50=110, sma200=100, rsi=30) == strategy.HOLD


def test_classify_nan_is_safe_hold():
    """NaN inputs (warm-up period) resolve to a safe HOLD."""
    assert classify(sma50=float("nan"), sma200=100, rsi=55) == strategy.HOLD


# --------------------------------------------------------------------------- #
# generate_signal(): structured, valid, bounded output
# --------------------------------------------------------------------------- #
def test_generate_signal_returns_valid_signal(uptrend_df):
    sig = generate_signal(uptrend_df, "TEST")
    assert isinstance(sig, Signal)
    assert sig.signal in VALID_SIGNALS
    assert sig.ticker == "TEST"


def test_confidence_is_bounded(uptrend_df, downtrend_df, choppy_df):
    """Confidence is always an int in [0, 100]."""
    for df in (uptrend_df, downtrend_df, choppy_df):
        sig = generate_signal(df, "TEST")
        assert isinstance(sig.confidence, int)
        assert 0 <= sig.confidence <= 100


def test_signal_has_reasons_and_risks(uptrend_df):
    sig = generate_signal(uptrend_df, "TEST")
    assert sig.reasons, "Every signal should explain itself"
    assert sig.risks, "Every signal should disclose risks"
    # The universal probabilistic caveat must always be present.
    assert any("probabilistic" in r.lower() for r in sig.risks)


def test_signal_has_bounded_factor_breakdown(uptrend_df, downtrend_df, choppy_df):
    """Every signal exposes the four factors and a score, all in [-1, 1]."""
    for df in (uptrend_df, downtrend_df, choppy_df):
        sig = generate_signal(df, "TEST")
        assert set(sig.factors) >= {"trend", "momentum", "participation", "quality"}
        for value in sig.factors.values():
            assert -1.0 <= value <= 1.0
        assert -1.0 <= sig.score <= 1.0


def test_factor_scores_neutral_on_empty_values():
    """With no indicator values, all factors are a neutral 0.0."""
    factors = strategy.factor_scores({})
    assert factors == {"trend": 0.0, "momentum": 0.0, "participation": 0.0, "quality": 0.0}


def test_uptrend_is_not_sell(uptrend_df):
    """A clean uptrend should never be classified as SELL."""
    sig = generate_signal(uptrend_df, "TEST")
    assert sig.signal in (strategy.BUY, strategy.WATCH, strategy.HOLD)


def test_downtrend_is_sell(downtrend_df):
    """A clean downtrend should be classified as SELL."""
    sig = generate_signal(downtrend_df, "TEST")
    assert sig.signal == strategy.SELL
    # Exit signals do not advertise an entry zone.
    assert sig.entry_zone is None


def test_invalid_signal_label_rejected():
    """The Signal dataclass refuses labels outside the allowed set."""
    with pytest.raises(ValueError):
        Signal(ticker="X", signal="MOON", confidence=50, latest_close=10.0)


def test_confidence_clamped_by_dataclass():
    """Out-of-range confidence is clamped to [0, 100]."""
    assert Signal("X", strategy.HOLD, 250, 10.0).confidence == 100
    assert Signal("X", strategy.HOLD, -5, 10.0).confidence == 0


# --------------------------------------------------------------------------- #
# Empty / malformed data is handled safely
# --------------------------------------------------------------------------- #
def test_generate_signal_empty_data_raises():
    with pytest.raises(ValueError):
        generate_signal(pd.DataFrame(), "EMPTY")


def test_generate_signal_missing_columns_raises():
    df = pd.DataFrame({"Close": [1, 2, 3]})
    with pytest.raises(ValueError):
        generate_signal(df, "PARTIAL")


# --------------------------------------------------------------------------- #
# signal_series(): used by the backtester
# --------------------------------------------------------------------------- #
def test_signal_series_only_valid_labels(choppy_df):
    series = signal_series(choppy_df)
    assert len(series) == len(choppy_df)
    assert set(series.unique()).issubset(set(VALID_SIGNALS))


def test_signal_series_warmup_is_hold():
    """Early rows (indicators still NaN) classify as HOLD."""
    df = make_ohlcv(np.linspace(50, 60, 10))
    # No indicators added -> add minimal NaN columns to exercise warm-up.
    df["SMA_50"] = float("nan")
    df["SMA_200"] = float("nan")
    df["RSI"] = float("nan")
    series = signal_series(df)
    assert (series == strategy.HOLD).all()
