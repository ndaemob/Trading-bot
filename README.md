# AI Trading Analyst Bot

[![CI](https://github.com/ndaemob/Trading-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/ndaemob/Trading-bot/actions/workflows/ci.yml)

A safe, data-driven **stock analysis** bot. It downloads historical price data,
computes technical indicators, and produces probabilistic **signals**
(`BUY` / `HOLD` / `WATCH` / `SELL`) with confidence scores, suggested entry
zones, stop-losses, and risk-managed position sizes. It can also **backtest**,
render **charts**, and run a **simulated paper-trading** account.

> ⚠️ **This is not financial advice. No real trades are ever executed.**

---

## What this bot does

- 📈 **Loads** up to 2 years of daily historical data from Yahoo Finance (via `yfinance`), with retries and a same-day local cache.
- 🧮 **Computes** technical indicators: RSI, SMA (20/50/200), MACD, and ATR.
- 🚦 **Generates** a single discrete signal per ticker with confidence (0–100), human-readable **reasons** and **risks**, latest close, suggested **entry zone** and **stop-loss**.
- 🛡️ **Sizes positions** under strict risk rules: max **2%** risk per trade, max **20%** allocation per stock, and **never** leverage.
- 🔁 **Backtests** the strategy over history (total return, # trades, win rate, max drawdown), modelling **commission** and **slippage**, with all-in or risk-based sizing.
- 📊 **Charts** price + moving averages + RSI, and the backtest equity curve (PNG).
- 🧪 **Paper-trades**: applies signals to a persistent, fully **simulated** portfolio (JSON-backed) — no broker, no real orders.

## What this bot does **not** do

- ❌ It does **not** connect to any broker or place real/live orders.
- ❌ It does **not** perform automatic or live trading.
- ❌ It does **not** require or store API keys or credentials.
- ❌ It does **not** use leverage or margin.
- ❌ It does **not** guarantee profits — every output is **probabilistic** and for **educational** purposes only.

---

## Project structure

```
ai-trading-bot/
├── README.md
├── LICENSE
├── Makefile              # make install / format / lint / typecheck / test / check
├── pyproject.toml        # packaging + ruff/black/mypy/pytest config
├── requirements.txt
├── .gitignore
├── .github/workflows/
│   └── ci.yml            # lint + type-check + tests on push/PR (Py 3.11 & 3.12)
├── src/
│   ├── __init__.py
│   ├── main.py           # CLI entry point
│   ├── data_loader.py    # yfinance download + validation + retry + cache
│   ├── indicators.py     # RSI, SMA, MACD, ATR
│   ├── strategy.py       # signal generation logic
│   ├── risk.py           # position sizing & risk limits
│   ├── backtest.py       # historical backtest (fees, slippage, sizing modes)
│   ├── charts.py         # matplotlib price & equity charts
│   ├── paper_trading.py  # persistent simulated portfolio
│   └── config.py         # tunable constants
└── tests/
    ├── conftest.py       # offline synthetic-data fixtures
    ├── test_strategy.py
    ├── test_risk.py
    ├── test_data_loader.py
    ├── test_indicators.py
    ├── test_backtest.py
    ├── test_paper_trading.py
    └── test_charts.py
```

---

## Installation

Requires **Python 3.11+**.

```bash
git clone https://github.com/ndaemob/Trading-bot.git
cd Trading-bot

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements.txt      # runtime only
# or, for development (tests + linters + type checker):
pip install -e ".[dev]"
```

---

## How to run

Analyse the default tickers (`AAPL, MSFT, NVDA, TSLA, GOOGL`):

```bash
python -m src.main
# or, after `pip install -e .`, use the console command:
ai-trading-bot
```

Common options:

```bash
python -m src.main AAPL MSFT NVDA          # specific tickers
python -m src.main NVDA --backtest         # add a historical backtest
python -m src.main NVDA --backtest --risk-sizing   # size by risk rules, not all-in
python -m src.main NVDA --chart            # save price/RSI (+ equity) charts to output/
python -m src.main AAPL --paper            # apply signals to a simulated portfolio
python -m src.main AAPL --period 5y --portfolio 25000
python -m src.main AAPL --no-cache         # bypass the local data cache
python -m src.main --help                  # full option list
```

| Flag | Effect |
| --- | --- |
| `--backtest` | Run a historical backtest per ticker |
| `--risk-sizing` | Size backtest entries via the 2%/20% risk rules (default: all-in) |
| `--chart` | Save price/RSI and equity-curve PNGs to `output/` |
| `--paper` | Apply signals to a persistent simulated paper portfolio |
| `--paper-file PATH` | Where to store the paper portfolio (default `paper_portfolio.json`) |
| `--period` / `--interval` | History window / candle size |
| `--portfolio` | Hypothetical portfolio value for sizing & backtests |
| `--no-cache` | Always download fresh instead of using the same-day cache |

### Example output

```
Ticker: NVDA
Signal: WATCH
Confidence: 72
Latest Close: 142.30
Reasons:
- SMA50 above SMA200 (long-term uptrend)
- RSI elevated at 74
Risks:
- Momentum may be overheated
- Not financial advice; signals are probabilistic only
Suggested Entry Zone:
135.00 - 140.00
Suggested Stop Loss:
128.00
```

> Network access to Yahoo Finance is required at runtime to download prices.
> The test suite, by contrast, runs **fully offline**.

---

## Development & testing

```bash
make install      # pip install -e ".[dev]"
make format       # black .
make lint         # ruff check .
make typecheck    # mypy src
make test         # pytest
make check        # everything CI runs (lint + type-check + tests + format check)
```

Without `make`, run the tools directly: `pytest`, `ruff check .`, `black --check .`, `mypy src`.

The tests use deterministic, in-memory synthetic price data, so they need
**no network connection** and **no live market data**. They verify, among other
things, that:

- the strategy only ever emits `BUY`, `HOLD`, `WATCH`, or `SELL`;
- position sizing never exceeds the 2% per-trade risk rule;
- an allocation warning appears above the 20% per-stock limit;
- trading costs reduce backtest returns and trade P/L is computed net of fees;
- the paper portfolio never goes negative (no leverage) and round-trips through save/load;
- empty or malformed data is handled safely (clear errors, no crashes).

CI (GitHub Actions) runs ruff, black, mypy and pytest on Python 3.11 and 3.12
for every push and pull request.

---

## How the strategy decides

| Condition                                              | Signal  |
|--------------------------------------------------------|---------|
| `SMA50 < SMA200` **or** `RSI > 80`                     | `SELL`  |
| `SMA50 > SMA200` **and** `40 ≤ RSI ≤ 70`              | `BUY`   |
| `SMA50 > SMA200` **and** `RSI > 70` (and `≤ 80`)      | `WATCH` |
| anything else (e.g. weak uptrend, RSI < 40)            | `HOLD`  |

All thresholds live in [`src/config.py`](src/config.py) and can be tuned in one
place. The first match (top-to-bottom) wins, so the rules never conflict.

---

## Extending the bot

The code is intentionally small, modular, and typed so you can build on it:

- Add indicators in `indicators.py`; tweak thresholds in `config.py`.
- Replace signal logic in `strategy.py` (`classify()` is the core).
- Improve the backtester (more cost models, multi-position portfolios) in `backtest.py`.
- Grow the paper-trading sandbox (`paper_trading.py`) toward a richer simulation — **still without real execution**.

---

## Disclaimer

This software is provided for **educational and informational purposes only**.
It is **not financial advice**, not an offer or solicitation, and not a
recommendation to buy or sell any security. **No real trades are executed.**
Markets are risky; past performance does not guarantee future results. Always do
your own research and consult a licensed financial professional before investing.
The authors accept no liability for any losses incurred.
