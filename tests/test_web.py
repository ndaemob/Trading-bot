"""Tests for the Flask web dashboard (fully offline via demo mode)."""

from __future__ import annotations

import pytest

from src import data_loader, web
from src.web import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_index_is_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"AI Trading" in resp.data


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_analyze_demo_returns_valid_signals(client):
    resp = client.post(
        "/api/analyze", json={"tickers": ["AAPL", "MSFT"], "demo": True, "period": "1y"}
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["results"]) == 2
    for res in data["results"]:
        assert res["ok"] is True
        assert res["signal"] in ("BUY", "HOLD", "WATCH", "SELL")
        assert 0 <= res["confidence"] <= 100
        assert set(res["factors"]) >= {"trend", "momentum", "participation", "quality"}
        assert res["narrative"]
        assert isinstance(res["history"], list) and len(res["history"]) > 1
        assert {"t", "c", "s20", "s50"} <= set(res["history"][0])


def test_analyze_defaults_to_five_tickers(client):
    data = client.post("/api/analyze", json={"demo": True}).get_json()
    assert len(data["results"]) == 5


def test_analyze_is_deterministic_in_demo(client):
    a = client.post("/api/analyze", json={"tickers": ["NVDA"], "demo": True}).get_json()
    b = client.post("/api/analyze", json={"tickers": ["NVDA"], "demo": True}).get_json()
    assert a["results"][0]["signal"] == b["results"][0]["signal"]
    assert a["results"][0]["confidence"] == b["results"][0]["confidence"]


def test_backtest_demo(client):
    data = client.post(
        "/api/backtest", json={"tickers": ["NVDA"], "demo": True, "period": "2y"}
    ).get_json()
    assert data["ok"] is True
    assert "metrics" in data and "sharpe" in data["metrics"]
    assert isinstance(data["equity_curve"], list) and data["equity_curve"]


def test_portfolio_demo(client):
    data = client.post(
        "/api/portfolio",
        json={"tickers": ["AAPL", "MSFT", "NVDA"], "demo": True, "period": "2y"},
    ).get_json()
    assert data["ok"] is True
    assert data["equity_curve"] and data["benchmark_curve"]
    assert "benchmark_return" in data["metrics"]


def test_analyze_handles_load_error_gracefully(client, monkeypatch):
    """A live-data failure becomes a structured ok=False result, not a 500."""

    def boom(*args, **kwargs):
        raise data_loader.DataLoadError("network blocked")

    monkeypatch.setattr(web.data_loader, "load_data", boom)
    data = client.post("/api/analyze", json={"tickers": ["AAPL"], "demo": False}).get_json()
    assert data["results"][0]["ok"] is False
    assert "network blocked" in data["results"][0]["error"]
