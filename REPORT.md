# Findings Report

**Question:** Can public information (SEC filings parsed by an LLM, EPS history, macro data) forecast quarterly earnings outcomes better than prediction markets?

**Short answer:** No — not with these inputs. Polymarket's earnings contracts are well calibrated and clearly beat every model built here. The LLM-extracted text features carried no measurable out-of-sample signal. The useful outputs are a rigorous, leak-free evaluation framework, a confirmed (small) EPS-surprise signal, and evidence that any remaining market inefficiency is smaller than the bid/ask spread.

All numbers below are reproducible with the commands in [Reproducing](#reproducing). Total LLM spend for the whole project: ~$1.80.

---

## 1. Headline results

| # | Finding | Evidence | Confidence |
|---|---|---|---|
| 1 | **Polymarket beats the model** on earnings beat/miss forecasts | Brier 0.143 vs 0.174, ROC AUC 0.80 vs 0.66, 1,256 resolved contracts | High |
| 2 | **The model adds nothing on top of the market** | Logistic blend of market + model: Brier 0.1364 vs market alone 0.1364 (out-of-sample half) | High |
| 3 | **LLM text features don't predict the price reaction** to earnings | Walk-forward AUC 0.50 (text only); no feature significant | High |
| 4 | **EPS surprise vs. consensus does** (modestly) | Spearman ρ = +0.16, p = 0.007; walk-forward AUC 0.54 | Medium |
| 5 | **Beats are predictable from beat history** | Beat model: Brier 0.169 vs 0.185 base rate, AUC 0.69 on 2,157 held-out releases | High |
| 6 | **Two candidate inefficiencies: non-GAAP beats underpriced; late moves overshoot** | +3.8¢ (BH q = 0.04) and +8.1¢ before spreads; the second may be bid-ask bounce | Low — exploratory; pre-registered out-of-sample test running |
| 7 | **No tradable edge yet** | Best strategy (buy YES on every contract) +1.7¢/contract, 95% CI [−0.4¢, +3.8¢], before a median 29¢ bid/ask spread | High |

---

## 2. Model vs. market backtest

**Setup** (`src/modeling/backtest.py`)
- 1,256 resolved Polymarket "Will X beat quarterly earnings?" contracts, Oct 2025 – Oct 2026, 387 companies.
- **Market probability:** last traded YES price before 00:00 UTC on the release date (the prior US evening — before both pre-market and after-close announcements).
- **Model probability:** next-quarter beat model, **walk-forward** — retrained each calendar quarter using only outcomes public before that quarter. No contract is scored by a model that saw its result.
- **Outcome:** how the contract actually resolved.

**Accuracy on the same 1,256 contracts**

| Forecast | Brier ↓ | Log loss ↓ | ROC AUC ↑ |
|---|---|---|---|
| **Polymarket** | **0.143** | **0.444** | **0.800** |
| Beat model | 0.174 | 0.529 | 0.663 |
| Base rate (~75%) | 0.188 | 0.564 | 0.500 |

**Calibration — the market is well calibrated**

| Market price bucket | Contracts | Avg price | Actual beat rate |
|---|---|---|---|
| 0–20% | 40 | 10% | 15% |
| 20–40% | 113 | 31% | 35% |
| 40–60% | 136 | 51% | 51% |
| 60–80% | 306 | 72% | 75% |
| 80–100% | 661 | 89% | 90% |

The actual beat rate sits slightly *above* the price in most buckets (overall 74.8% vs. an average price of 73.2%), but the gap is not statistically significant.

**Trading simulation** (buy the side the model favors when it disagrees with the market by more than a threshold; $1 payout contracts, no fees or spread)

| Threshold | Trades | P&L per trade | 95% CI |
|---|---|---|---|
| 10 pts | 679 | +1.1¢ | [−1.6¢, +3.9¢] |
| 20 pts | 257 | +0.9¢ | [−4.3¢, +6.0¢] |
| 10 pts, buy-YES side only | 300 | +3.0¢ | [−2.0¢, +8.2¢] |
| Buy YES on every contract | 1,256 | +1.7¢ | [−0.4¢, +3.8¢] |

None of these is a demonstrated edge: every interval includes zero, and the figures use last-trade prices, not executable ones. Live order-book snapshots show a **median bid/ask spread of 29¢** across open earnings contracts — many sit at placeholder quotes (bid ≈ 0.20, ask ≈ 0.90) until about a week before release — and only a handful trade at 1–5¢ spreads. Crossing that spread costs far more than the 1–3¢ the simulated strategies earn.

An earlier run on 1,111 contracts showed buy-YES-everything at +2.2¢ with a CI just touching zero. A slug-parsing bug had excluded every contract with a negative EPS consensus (loss-making companies); with those 145 contracts restored, the effect shrank to +1.7¢ and lost significance. This is the main reason small backtest edges shouldn't be trusted until they survive the full sample and realistic costs.

---

## 2a. How efficiently does Polymarket price beats?

`src/research/beat_pricing.py`, same 1,256 contracts and cached price histories.

**Calibration regression** — logit P(beat) = a + b · logit(price). A calibrated market has a = 0, b = 1.

| | Estimate | 95% CI | p |
|---|---|---|---|
| a (bias toward beats) | +0.142 | [−0.039, +0.323] | 0.13 |
| b (slope) | 0.970 | [0.839, 1.102] | 0.66 (vs. 1) |
| Joint test a = 0, b = 1 | | | 0.26 |

Overall the market is calibrated: no significant bias, and prices are neither too extreme nor too timid.

**Slices** (17 tested; Benjamini-Hochberg correction). Only one survives:

| Slice | n | Avg price | Beat rate | Edge | 95% CI | BH q |
|---|---|---|---|---|---|---|
| **Non-GAAP EPS contracts** | 792 | 77.7% | 81.4% | **+3.8¢** | [+1.3¢, +6.2¢] | **0.04** |
| GAAP EPS contracts | 464 | 65.5% | 63.6% | −1.9¢ | [−5.7¢, +1.9¢] | 0.56 |

The non-GAAP edge holds with each company weighted equally (+3.5¢, 239 companies) and is positive in every price bucket, but it is uneven over time (+8.9¢ and +5.7¢ in two quarters, ≈ 0 in the other two). A plausible mechanism: companies have discretion over non-GAAP EPS and tend to manage it to just clear consensus.

**Price dynamics** — the market learns as the release approaches (contracts priced at every horizon, n = 1,168):

| Days before release | 5 | 3 | 1 | Release eve |
|---|---|---|---|---|
| Brier | 0.157 | 0.147 | 0.144 | 0.140 |
| ROC AUC | 0.741 | 0.781 | 0.786 | 0.803 |

Late moves are informative (61% of ≥ 5-point moves go toward the eventual outcome) but **overshoot**: regressing the outcome on the final price plus the 3-day drift gives a drift coefficient of −0.34 (95% CI [−0.58, −0.10], p = 0.006). Betting against ≥ 10-point moves earned +8.1¢ per contract (n = 309, CI [+3.4¢, +12.9¢]) before spreads. The effect is weaker in the highest-volume third (+4.3¢, CI includes 0), which fits an alternative explanation: **bid-ask bounce** in last-trade prices, not genuine overreaction. Last-trade data can't separate the two.

**Both leads were found by searching the data**, so they are being tested out of sample at executable prices under a fixed, pre-committed plan: [PREREGISTRATION.md](PREREGISTRATION.md) (contracts releasing Oct 20 – Dec 31, 2026; analysis on Jan 5, 2027).

---

## 3. What the LLM features did and didn't do

**Setup** (`src/modeling/train.py`): 270 earnings events (20 large caps, 3 years, one per company-quarter). Claude Haiku 5.5 extracts 8 structured fields from each 8-K earnings press release (sentiment, guidance, tone, revenue surprise, analyst sentiment, risk flags), plus quarter-over-quarter changes. Label: did the stock beat SPY over the ±1 trading day window around the filing?

**Per-feature correlation with the label** (none of the text features is significant)

| Feature | Spearman ρ | p-value |
|---|---|---|
| eps_surprise_pct | **+0.164** | **0.007** |
| eps_beat | +0.129 | 0.034 |
| sentiment_change (QoQ) | +0.088 | 0.15 |
| risk_flag_count | +0.071 | 0.24 |
| sentiment_score | +0.033 | 0.59 |
| forward_guidance_score | +0.032 | 0.60 |
| all macro features | \|ρ\| < 0.07 | > 0.27 |

**Walk-forward ROC AUC (162 test events, 5 windows)**

| Features | Logistic | XGBoost |
|---|---|---|
| Text only | 0.50 | 0.47 |
| Text + QoQ change | 0.48 | 0.52 |
| EPS only | 0.54 | — |
| EPS + text | 0.54 | — |

**Why the text didn't work:** press releases are written to sound good — 80%+ of extractions were "bullish" — so tone barely varies between good and bad quarters, and the market reacts to the *numbers vs. expectations* within minutes. The extraction itself worked well (sensible outputs like "margin boosted by one-time tariff refunds"); there just isn't predictive signal in press-release tone that EPS surprise doesn't already capture. This matches the academic literature, and the pipeline confirmed it on live data rather than assuming it.

**Small-sample warning:** a single 68-event holdout swung between AUC 0.56 and 0.39 across feature sets — pure noise. This is why the conclusions above rest on walk-forward evaluation over 162 events, not single holdouts.

---

## 4. Next-quarter beat model

`src/modeling/beat_model.py` — trained on the companies Polymarket actually lists (406 companies, 8,628 releases), using only EPS history: last surprise, last beat, 4-quarter beat rate, 4-quarter mean surprise, beat streak.

- Chronological holdout (2,157 releases, Feb 2025 – Jul 2026): **Brier 0.169 vs 0.185** base rate, **AUC 0.69**.
- **Population matters:** an earlier version trained on 20 mega-caps (85% beat rate) would have flagged nearly every Polymarket contract as "underpriced" — mid-caps and regional banks beat far less often. Training on the contract universe removed that bias.
- It beats the base rate but not the market (§2): the market already knows the beat history *plus* guidance, analyst revisions, sector news, etc.

---

## 5. Engineering findings worth knowing

Problems found and fixed along the way — each would have silently corrupted results:

| Problem | Impact if unfixed | Fix |
|---|---|---|
| 8-K pipeline read the **cover page**, not the press release (Exhibit 99.1) | Half the "transcripts" were XBRL boilerplate; LLM scored them "neutral" | Parse the filing index, pull exhibit documents |
| Each earnings event counted **twice** (8-K and 10-Q filed days apart) | Inflated sample size, leaky evaluation | Dedupe to one event per company-quarter |
| Training rows sorted by ticker, not date | Time-series validation was meaningless | Chronological sort |
| SEC throttling **silently dropped whole companies** (GOOGL, META) | Biased, non-reproducible training set | Retry with exponential backoff |
| Polymarket `endDate` is a resolution deadline, not the release date | Contracts matched to the wrong earnings event | Parse release date from the market slug |
| Open contracts whose earnings already happened | "Mispricings" that are just unsettled outcomes | Drop contracts past their release date |
| LLM responses with thinking blocks / markdown fences | **0 of 44** extractions succeeded on the first paid run | Structured outputs (enforced JSON schema) |
| Backtest model could see the outcomes it was graded on | Inflated model accuracy | Walk-forward retraining per quarter |
| Slug parser rejected **negative** EPS thresholds (`…-neg0pt34`) | Every loss-making company silently dropped from backtest and live signals (145 contracts); inflated the apparent edge | Parse the optional `neg` prefix |
| Order-book `last_trade_price` often reports the NO side (0.04 on a 0.92/0.93 YES book) | Inverted prices in any analysis built on it | Use best bid/ask; field kept only for reference |
| Real API keys pasted into tracked `.env.example` | Keys published on commit | Caught before commit; reset to placeholders |

Cost controls: extraction cache keyed by (model, filing), `--cached-only` training flag, and a switch from Sonnet to Haiku cut per-run extraction cost ~20×.

---

## 6. Limitations

- **One year of contracts.** Polymarket's earnings markets only go back to late 2025; 1,256 contracts is enough to rank forecasts but thin for trading conclusions.
- **Last-trade prices**, not executable bid/ask, in the historical backtest. Polymarket keeps no order-book history, so executable prices exist only from the start of live collection (Oct 16, 2026).
- **Consensus mismatch.** Polymarket's threshold is the consensus at market creation (sometimes GAAP, sometimes non-GAAP); the model learns from Yahoo's consensus. Some disagreement is definitional, not informational.
- **LLM study covers 20 large caps.** Text features might matter more for smaller, less-covered companies — untested (budget).
- **Price-reaction label is noisy** (±1 day abnormal return vs. SPY, no factor model).

---

## 7. What would be worth doing next

1. **Execution-realistic testing.** Order-book snapshots of every open earnings contract (every 30 minutes) are now collected automatically (`src/ingestion/orderbook_collector.py`, GitHub Actions → `orderbook-data` branch). As contracts resolve, re-test at executable prices, measure how spreads and prices evolve into the release, and paper-trade a pre-registered strategy restricted to tight-spread markets.
2. **Use the LLM on what the market doesn't price well:** earnings-call *Q&A* transcripts (not press releases), or guidance language for small caps with little analyst coverage.
3. **Kalshi/Polymarket "mention" markets** ("Will the CEO say X on the call?") — a direct fit for transcript NLP.

---

## Reproducing

```bash
python -m src.modeling.beat_model     # beat model (free; Yahoo data)
python -m src.modeling.backtest       # §2 (free; Polymarket + Yahoo)
python -m src.research.beat_pricing   # §2a (run after the backtest)
python -m src.modeling.train --tickers AAPL MSFT GOOGL AMZN META NVDA TSLA JPM V JNJ WMT XOM PG MA HD CVX KO PEP ABBV COST \
    --lookback-days 1095 --cached-only  # §3 (free with the extraction cache; otherwise ~$0.25 on Haiku)
python -m src.pipeline                # live signals vs. open contracts (free)
streamlit run src/dashboard/app.py    # Signals + Backtest pages
```

Per-contract backtest results: `data/processed/backtest_results.parquet`. Training set: `data/processed/training_set.parquet`.
