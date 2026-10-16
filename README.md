# EDGAR Mispricing Pipeline

Can public information predict quarterly earnings outcomes better than prediction markets do? This project extracts features from SEC EDGAR earnings filings with an LLM, combines them with EPS history, and compares the resulting forecasts against live and historical prices of Polymarket's "Will X beat quarterly earnings?" contracts.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Results

Full write-up with methodology, calibration tables, caveats and engineering notes: **[REPORT.md](REPORT.md)**.

**1. LLM features vs. the price reaction to earnings** (`src/modeling/train.py`)
270 earnings releases (20 large caps, 3 years). Claude (Haiku 5.5) extracts sentiment, guidance, tone, revenue-surprise and risk-flag features from each 8-K earnings press release.

| Features | Walk-forward ROC AUC (162 test events) |
|---|---|
| Text only | 0.50 |
| EPS surprise only | 0.54 |
| EPS surprise + text | 0.54 |

Press-release sentiment carries no out-of-sample signal for the next-day abnormal return; EPS surprise vs. consensus is the only significant predictor (Spearman ρ = 0.16, p = 0.007).

**2. Next-quarter EPS beat model** (`src/modeling/beat_model.py`)
Trained on 8,628 releases from the 406 companies Polymarket has listed earnings markets for. Chronological holdout (2,157 releases): Brier 0.169 vs. 0.185 for the base rate, ROC AUC 0.69.

**3. Model vs. Polymarket backtest** (`src/modeling/backtest.py`)
1,256 resolved earnings contracts (Oct 2025 – Oct 2026). Market price = last trade before the release day; model = walk-forward, trained only on outcomes public before each contract's quarter.

| Forecast | Brier ↓ | Log loss ↓ | ROC AUC ↑ |
|---|---|---|---|
| **Polymarket** | **0.143** | **0.444** | **0.800** |
| Beat model | 0.174 | 0.529 | 0.663 |
| Base rate | 0.188 | 0.564 | 0.500 |

The market is well calibrated and clearly beats a model built on EPS history alone; blending the model into the market price doesn't improve it. No simulated strategy shows a significant edge (best: +1.7¢ per $1 contract, 95% CI [−0.4¢, +3.8¢]), and live order books show a median bid/ask spread of 29¢ — far larger than any edge.

**4. Live order books** (`src/ingestion/orderbook_collector.py`)
Hourly snapshots of every open earnings contract's bid/ask and depth, collected by GitHub Actions onto the `orderbook-data` branch, for execution-realistic testing as contracts resolve.

## How it works

```
SEC EDGAR 8-K/10-Q ──► Claude extraction ──┐
Yahoo EPS vs. consensus ───────────────────┼──► price-reaction model (train.py)
FRED macro indicators ─────────────────────┘

Yahoo EPS history (406 cos.) ──► next-quarter beat model (beat_model.py)
                                          │
Polymarket open contracts ────────────────┼──► live signals (pipeline.py) ──► Streamlit
Polymarket resolved contracts + CLOB prices ─► backtest (backtest.py) ───────► Streamlit
```

- **Ingestion** (`src/ingestion/`): EDGAR filings (8-K earnings exhibits, 10-Q/10-K) with rate-limit pacing and retries; FRED indicators; Kalshi and Polymarket contracts via their public APIs (no keys needed); hourly Polymarket order-book snapshots.
- **Processing** (`src/processing/`): Claude structured extraction with an enforced JSON schema and an on-disk cache (a filing is never paid for twice); market-adjusted post-earnings returns; EPS surprise matching.
- **Modeling** (`src/modeling/`): XGBoost with Platt-scaled probabilities, chronological holdouts, walk-forward backtesting.
- **Signals**: each open contract is matched to the company's latest report before the release date; already-released contracts and Kalshi's KPI contracts (which ask a different question) are excluded.

## Usage

```bash
pip install -r requirements.txt
cp .env.example .env              # ANTHROPIC_API_KEY, FRED_API_KEY, EDGAR_USER_AGENT; USE_FIXTURES=false for live data

python -m src.pipeline            # live signals vs. open Polymarket contracts (no Claude calls)
python -m src.modeling.beat_model # retrain the beat model (free: Yahoo data only)
python -m src.modeling.backtest   # model vs. market on resolved contracts (free)
python -m src.ingestion.orderbook_collector  # one order-book snapshot (runs hourly in GitHub Actions)
python -m src.modeling.train --tickers AAPL MSFT ... --lookback-days 1095 --cached-only
                                  # price-reaction model; --cached-only never calls Claude
streamlit run src/dashboard/app.py
pytest -q
```

With `USE_FIXTURES=true` (the default) the pipeline and dashboard run on bundled sample data with no API keys.

## Project structure

```
src/
├── ingestion/     edgar_client.py, fred_client.py, contract_client.py, orderbook_collector.py
├── processing/    transcript_extractor.py, price_labels.py, eps_surprise.py, macro_features.py
├── modeling/      train.py, beat_model.py, backtest.py, scorer.py, signal.py, model.py
├── dashboard/     app.py
└── pipeline.py
tests/             unit tests (run in CI on every push)
data/fixtures/     sample data for keyless runs
```

## License

MIT
