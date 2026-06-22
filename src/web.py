"""Local web dashboard for the AI Trading Analyst Bot.

A lightweight Flask server that serves a sleek single-page dashboard and a small
JSON API on top of the existing analysis engine. It is **analysis only** — it
never connects to a broker or places a real trade.

Run it::

    python -m src.web                 # then open http://127.0.0.1:5000
    ai-trading-bot-web --port 8000    # after `pip install -e .`

A **demo mode** (toggle in the UI / ``"demo": true`` in the API) generates
deterministic synthetic data so the dashboard works fully offline, with no
network access required.
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from . import (
    backtest,
    config,
    data_loader,
    indicators,
    portfolio,
    risk,
    strategy,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "web_static"
MAX_TICKERS = 12

DISCLAIMER = "Educational tool only. Not financial advice. No real trades are executed."


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def _parse_tickers(value: Any) -> list[str]:
    """Normalise a tickers payload (list or delimited string) to a clean list."""
    if isinstance(value, str):
        raw = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple)):
        raw = [str(v) for v in value]
    else:
        raw = list(config.DEFAULT_TICKERS)
    seen: list[str] = []
    for item in raw:
        sym = item.strip().upper()
        if sym and sym not in seen:
            seen.append(sym)
    return seen[:MAX_TICKERS] or list(config.DEFAULT_TICKERS)


def _period_to_days(period: str) -> int:
    """Approximate the number of trading days in a yfinance period string."""
    p = (period or config.DEFAULT_PERIOD).strip().lower()
    try:
        if p.endswith("mo"):
            return min(int(float(p[:-2]) * 21), 2520)
        if p.endswith("y"):
            return min(int(float(p[:-1]) * 252), 2520)
        if p.endswith("d"):
            return min(int(p[:-1]), 2520)
    except ValueError:
        pass
    return 504


def _demo_frame(ticker: str, period: str) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV data for offline/demo use."""
    n = max(260, _period_to_days(period))
    seed = int(hashlib.md5(ticker.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(seed)

    drift = rng.uniform(-0.0002, 0.0009)
    vol = rng.uniform(0.012, 0.028)
    shocks = rng.normal(drift, vol, n)
    start_price = rng.uniform(25, 320)
    close = np.maximum(start_price * np.cumprod(1.0 + shocks), 1.0)

    intraday = rng.uniform(0.005, 0.02, n)
    index = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": np.concatenate([[close[0]], close[:-1]]),
            "High": close * (1 + intraday),
            "Low": close * (1 - intraday),
            "Close": close,
            "Volume": rng.uniform(1e6, 2e7, n),
        },
        index=index,
    )


def _load_frame(ticker: str, period: str, demo: bool) -> pd.DataFrame:
    """Load (or synthesise) a price frame and attach indicators."""
    raw = _demo_frame(ticker, period) if demo else data_loader.load_data(ticker, period=period)
    return indicators.add_indicators(raw)


def _curve(series: pd.Series | None, max_points: int = 220) -> list[dict[str, Any]]:
    """Down-sample an equity curve to a compact list of ``{t, v}`` points."""
    if series is None or len(series) == 0:
        return []
    step = max(1, len(series) // max_points)
    sampled = series.iloc[::step]
    return [{"t": idx.strftime("%Y-%m-%d"), "v": round(float(v), 2)} for idx, v in sampled.items()]


def _num_or_none(value: Any) -> float | None:
    """Coerce a value to a rounded float, or ``None`` when missing/NaN."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, 2)


def _price_history(df: pd.DataFrame, max_points: int = 140) -> list[dict[str, Any]]:
    """Down-sample close + SMA20/SMA50 for the per-card mini price chart."""
    if df is None or len(df) == 0:
        return []
    step = max(1, len(df) // max_points)
    sampled = df.iloc[::step]
    out: list[dict[str, Any]] = []
    for idx, row in sampled.iterrows():
        out.append(
            {
                "t": idx.strftime("%Y-%m-%d"),
                "c": _num_or_none(row.get("Close")),
                "s20": _num_or_none(row.get("SMA_20")),
                "s50": _num_or_none(row.get("SMA_50")),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Narrative + payload builders
# --------------------------------------------------------------------------- #
def _narrative(sig: strategy.Signal) -> str:
    """Compose a short, readable analyst-style summary from a signal."""
    t = sig.ticker
    lead = sig.reasons[0].lower() if sig.reasons else "mixed conditions"
    if sig.signal == strategy.BUY:
        text = f"{t} looks constructive — {lead}, with a confidence of {sig.confidence}/100."
        if sig.entry_zone:
            text += f" A measured entry sits around {sig.entry_zone[0]:.2f}–{sig.entry_zone[1]:.2f}"
        if sig.stop_loss:
            text += f", protected by a stop near {sig.stop_loss:.2f}."
    elif sig.signal == strategy.WATCH:
        text = (
            f"{t} is strong but stretched — {lead}. Confidence {sig.confidence}/100. "
            "Patience is warranted; wait for a healthier pullback rather than chasing."
        )
    elif sig.signal == strategy.SELL:
        text = (
            f"{t} is unattractive for new longs — {lead}. Confidence {sig.confidence}/100. "
            "The model favours stepping aside or reducing exposure."
        )
    else:
        text = (
            f"{t} shows no clear edge right now — {lead}. Confidence {sig.confidence}/100. "
            "Holding and reassessing is the disciplined choice."
        )
    return text


def _position_payload(sig: strategy.Signal, portfolio_value: float) -> dict[str, Any] | None:
    """Risk-managed sizing for a tradeable long signal, as a JSON dict."""
    if sig.signal not in (strategy.BUY, strategy.WATCH) or not sig.stop_loss:
        return None
    entry = sig.entry_zone[1] if sig.entry_zone else sig.latest_close
    try:
        sized = risk.calculate_position_size(
            portfolio_value=portfolio_value,
            entry_price=entry,
            stop_loss=sig.stop_loss,
            ticker=sig.ticker,
        )
    except ValueError:
        return None
    return {
        "shares": sized.shares,
        "position_value": sized.position_value,
        "allocation_pct": round(sized.allocation_pct, 4),
        "risk_pct": round(sized.risk_pct, 4),
        "warnings": sized.warnings,
    }


def _signal_payload(sig: strategy.Signal, portfolio_value: float) -> dict[str, Any]:
    """Serialise a :class:`~src.strategy.Signal` for the API."""
    return {
        "ticker": sig.ticker,
        "signal": sig.signal,
        "confidence": sig.confidence,
        "score": sig.score,
        "latest_close": sig.latest_close,
        "factors": sig.factors,
        "reasons": sig.reasons,
        "risks": sig.risks,
        "entry_zone": list(sig.entry_zone) if sig.entry_zone else None,
        "stop_loss": sig.stop_loss,
        "narrative": _narrative(sig),
        "position": _position_payload(sig, portfolio_value),
    }


def _analyze_one(ticker: str, period: str, portfolio_value: float, demo: bool) -> dict[str, Any]:
    """Analyse a single ticker, returning a JSON-ready result or an error dict."""
    try:
        df = _load_frame(ticker, period, demo)
        sig = strategy.generate_signal(df, ticker)
    except (data_loader.DataLoadError, ValueError) as exc:
        return {"ticker": ticker.upper(), "ok": False, "error": str(exc)}
    payload = _signal_payload(sig, portfolio_value)
    payload["history"] = _price_history(df)
    payload["ok"] = True
    return payload


# --------------------------------------------------------------------------- #
# Flask app
# --------------------------------------------------------------------------- #
def create_app() -> Flask:
    """Build and configure the Flask application."""
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

    @app.get("/")
    def index() -> Any:
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/api/health")
    def health() -> Any:
        return jsonify({"status": "ok", "disclaimer": DISCLAIMER})

    @app.post("/api/analyze")
    def analyze() -> Any:
        payload = request.get_json(silent=True) or {}
        tickers = _parse_tickers(payload.get("tickers"))
        period = str(payload.get("period", config.DEFAULT_PERIOD))
        demo = bool(payload.get("demo", False))
        try:
            portfolio_value = float(payload.get("portfolio", config.DEFAULT_PORTFOLIO_VALUE))
        except (TypeError, ValueError):
            portfolio_value = config.DEFAULT_PORTFOLIO_VALUE

        results = [_analyze_one(t, period, portfolio_value, demo) for t in tickers]
        return jsonify({"results": results, "disclaimer": DISCLAIMER})

    @app.post("/api/backtest")
    def backtest_route() -> Any:
        payload = request.get_json(silent=True) or {}
        tickers = _parse_tickers(payload.get("tickers"))
        ticker = tickers[0]
        period = str(payload.get("period", config.DEFAULT_PERIOD))
        demo = bool(payload.get("demo", False))
        try:
            capital = float(payload.get("portfolio", config.DEFAULT_PORTFOLIO_VALUE))
        except (TypeError, ValueError):
            capital = config.DEFAULT_PORTFOLIO_VALUE

        try:
            df = _load_frame(ticker, period, demo)
            result = backtest.run_backtest(
                df,
                ticker,
                initial_capital=capital,
                commission_pct=config.COMMISSION_PCT,
                slippage_pct=config.SLIPPAGE_PCT,
            )
        except (data_loader.DataLoadError, ValueError) as exc:
            return jsonify({"ok": False, "ticker": ticker, "error": str(exc)}), 200

        return jsonify(
            {
                "ok": True,
                "ticker": result.ticker,
                "metrics": {
                    "total_return": result.total_return,
                    "cagr": result.cagr,
                    "sharpe": result.sharpe,
                    "sortino": result.sortino,
                    "volatility": result.volatility,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor,
                    "num_trades": result.num_trades,
                    "final_value": result.final_value,
                },
                "equity_curve": _curve(result.equity_curve),
            }
        )

    @app.post("/api/portfolio")
    def portfolio_route() -> Any:
        payload = request.get_json(silent=True) or {}
        tickers = _parse_tickers(payload.get("tickers"))
        period = str(payload.get("period", config.DEFAULT_PERIOD))
        demo = bool(payload.get("demo", False))
        try:
            capital = float(payload.get("portfolio", config.DEFAULT_PORTFOLIO_VALUE))
        except (TypeError, ValueError):
            capital = config.DEFAULT_PORTFOLIO_VALUE

        data: dict[str, pd.DataFrame] = {}
        errors: dict[str, str] = {}
        for t in tickers:
            try:
                data[t] = _load_frame(t, period, demo)
            except (data_loader.DataLoadError, ValueError) as exc:
                errors[t] = str(exc)

        try:
            result = portfolio.run_portfolio_backtest(
                data,
                initial_capital=capital,
                commission_pct=config.COMMISSION_PCT,
                slippage_pct=config.SLIPPAGE_PCT,
            )
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc), "errors": errors}), 200

        return jsonify(
            {
                "ok": True,
                "tickers": result.tickers,
                "errors": errors,
                "metrics": {
                    "total_return": result.total_return,
                    "cagr": result.cagr,
                    "sharpe": result.sharpe,
                    "sortino": result.sortino,
                    "volatility": result.volatility,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor,
                    "num_trades": result.num_trades,
                    "final_value": result.final_value,
                    "benchmark_return": result.benchmark_return,
                    "benchmark_cagr": result.benchmark_cagr,
                },
                "equity_curve": _curve(result.equity_curve),
                "benchmark_curve": _curve(result.benchmark_curve),
            }
        )

    return app


def main() -> None:
    """Console entry point: start the local dashboard server."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="AI Trading Analyst — local web dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=5000, help="Bind port (default: 5000).")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode.")
    args = parser.parse_args()

    app = create_app()
    print("=" * 60)
    print("  AI Trading Analyst — dashboard (analysis only, no trades)")
    print(f"  Open  ->  http://{args.host}:{args.port}")
    print("=" * 60)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":  # pragma: no cover
    main()
