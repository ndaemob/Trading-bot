"""Tests for data loading, caching and retry logic (fully offline)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data_loader
from src.data_loader import DataLoadError, load_data, validate_data

from .conftest import make_ohlcv


def small_df() -> pd.DataFrame:
    """A tiny valid OHLCV frame for injection in place of a real download."""
    return make_ohlcv(np.linspace(100, 110, 30))


def test_validate_empty_raises():
    with pytest.raises(DataLoadError):
        validate_data(pd.DataFrame(), "X")


def test_validate_missing_columns_raises():
    with pytest.raises(DataLoadError):
        validate_data(pd.DataFrame({"Close": [1, 2]}), "X")


def test_empty_ticker_raises():
    with pytest.raises(DataLoadError):
        load_data("   ")


def test_load_data_success_and_caches(tmp_path, monkeypatch):
    df = small_df()
    calls = {"n": 0}

    def fake(ticker, period, interval):
        calls["n"] += 1
        return df

    monkeypatch.setattr(data_loader, "_download_raw", fake)
    out = load_data("aapl", cache_dir=str(tmp_path))
    assert not out.empty
    assert calls["n"] == 1
    assert list(tmp_path.iterdir()), "a cache file should have been written"


def test_cache_is_reused(tmp_path, monkeypatch):
    df = small_df()
    monkeypatch.setattr(data_loader, "_download_raw", lambda t, p, i: df)
    load_data("aapl", cache_dir=str(tmp_path))  # populates cache

    def boom(t, p, i):
        raise DataLoadError("network down")

    monkeypatch.setattr(data_loader, "_download_raw", boom)
    out = load_data("aapl", cache_dir=str(tmp_path))  # served from cache
    assert not out.empty


def test_no_cache_bypasses_cache(tmp_path, monkeypatch):
    df = small_df()
    monkeypatch.setattr(data_loader, "_download_raw", lambda t, p, i: df)
    load_data("aapl", cache_dir=str(tmp_path))

    calls = {"n": 0}

    def fake(t, p, i):
        calls["n"] += 1
        return df

    monkeypatch.setattr(data_loader, "_download_raw", fake)
    load_data("aapl", cache_dir=str(tmp_path), use_cache=False)
    assert calls["n"] == 1


def test_retry_then_success(tmp_path, monkeypatch):
    df = small_df()
    state = {"n": 0}

    def flaky(t, p, i):
        state["n"] += 1
        if state["n"] < 3:
            raise DataLoadError("transient")
        return df

    monkeypatch.setattr(data_loader, "_download_raw", flaky)
    out = load_data("aapl", cache_dir=str(tmp_path), max_retries=3, retry_backoff=0.0)
    assert not out.empty
    assert state["n"] == 3


def test_retry_exhausted_raises(tmp_path, monkeypatch):
    def boom(t, p, i):
        raise DataLoadError("provider down")

    monkeypatch.setattr(data_loader, "_download_raw", boom)
    with pytest.raises(DataLoadError):
        load_data("aapl", cache_dir=str(tmp_path), max_retries=2, retry_backoff=0.0)
