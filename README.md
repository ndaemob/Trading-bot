# AI Trading Analyst Bot

A safe, data-driven **stock analysis** bot. It downloads historical price data,
computes technical indicators, and produces probabilistic **signals**
(`BUY` / `HOLD` / `WATCH` / `SELL`) with confidence scores, suggested entry
zones, stop-losses, and risk-managed position sizes.

> ⚠️ **This is not financial advice. No real trades are ever executed.**

---

## What this bot does

- 📈 **Loads** up to 2 years of daily historical data from Yahoo Finance (via `yfinance`).
- 🧮 **Computes** technical indicators: RSI, SMA (20/50/200), MACD, and ATR.
- 🚦 **Generates** a single discrete signal per ticker with:
  - confidence score (0–100),
  - human-readable **reasons** and **risks**,
  - latest close price,
  - suggested **entry zone** and **stop-loss**.
- 🛡️ **Sizes positions** under strict risk rules:
  - max **2%** of the portfolio risked per trade,
  - max **20%** of the portfolio allocated to any one stock,
  - **never** uses leverage.
- 🔁 **Backtests** the strategy over history (total return, number of trades,
  win rate, max drawdown).

## What this bot does **not** do

- ❌ It does **not** connect to any broker or place real/live orders.
- ❌ It does **not** perform automatic or live trading.
- ❌ It does **not** require or store API keys or credentials.
- ❌ It does **not** use leverage or margin.
- ❌ It does **not** guarantee profits — every output is **probabilistic** and
  for **educational** purposes only.

---

## Project structure

```
ai-trading-bot/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── main.py          # CLI entry point
│   ├── data_loader.py   # yfinance download + validation
│   ├── indicators.py    # RSI, SMA, MACD, ATR
│   ├── strategy.py      # signal generation logic
│   ├── risk.py          # position sizing & risk limits
│   ├── backtest.py      # simple historical backtest
│   └── config.py        # tunable constants
└── tests/
    ├── conftest.py      # offline synthetic-data fixtures
    ├── test_strategy.py
    └── test_risk.py
```

---

## Installation

Requires **Python 3.11+**.

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd ai-trading-bot

# 2. (Recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to run

Analyse the default tickers (`AAPL, MSFT, NVDA, TSLA, GOOGL`):

```bash
python -m src.main
```

Analyse specific tickers:

```bash
python -m src.main AAPL MSFT NVDA
```

Useful options:

```bash
python -m src.main NVDA --backtest              # also run a historical backtest
python -m src.main AAPL --period 5y             # 5 years of history
python -m src.main TSLA --portfolio 25000       # size against a $25k portfolio
python -m src.main --help                        # full option list
```

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

## How to run the tests

```bash
pytest                 # run everything
pytest -v              # verbose
pytest tests/test_risk.py
```

The tests use deterministic, in-memory synthetic price data, so they need
**no network connection** and **no live market data**.

They verify, among other things, that:

- the strategy only ever emits `BUY`, `HOLD`, `WATCH`, or `SELL`;
- position sizing never exceeds the 2% per-trade risk rule;
- an allocation warning appears above the 20% per-stock limit;
- empty or malformed data is handled safely (clear errors, no crashes).

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

- Add indicators in `indicators.py`.
- Tweak thresholds in `config.py`.
- Add or replace signal logic in `strategy.py` (`classify()` is the core).
- Improve the backtester (fees, slippage, position sizing) in `backtest.py`.
- Add paper-trading on top of the signals — **without** real execution.

---

## Disclaimer

This software is provided for **educational and informational purposes only**.
It is **not financial advice**, not an offer or solicitation, and not a
recommendation to buy or sell any security. **No real trades are executed.**
Markets are risky; past performance does not guarantee future results. Always do
your own research and consult a licensed financial professional before investing.
The authors accept no liability for any losses incurred.
