# Findings Report

**Question:** Can public information (SEC filings parsed by an LLM, EPS history, macro data) forecast quarterly earnings outcomes better than prediction markets?

**Short answer:** No — not with these inputs. Polymarket's earnings contracts are well calibrated and clearly beat every model built here. The LLM-extracted text features carried no measurable out-of-sample signal. The useful outputs are a rigorous, leak-free evaluation framework, a confirmed (small) EPS-surprise signal, and one borderline market inefficiency worth testing with more data.

All numbers below are reproducible with the commands in [Reproducing](#reproducing). Total LLM spend for the whole project: ~$1.80.

---

## 1. Headline results

| # | Finding | Evidence | Confidence |
|---|---|---|---|
| 1 | **Polymarket beats the model** on earnings beat/miss forecasts | Brier 0.137 vs 0.163, ROC AUC 0.79 vs 0.64, 1,111 resolved contracts | High |
| 2 | **The model adds nothing on top of the market** | Logistic blend of market + model: Brier 0.1312 vs market alone 0.1307 (out-of-sample half) | High |
| 3 | **LLM text features don't predict the price reaction** to earnings | Walk-forward AUC 0.50 (text only); no feature significant | High |
| 4 | **EPS surprise vs. consensus does** (modestly) | Spearman ρ = +0.16, p = 0.007; walk-forward AUC 0.54 | Medium |
| 5 | **Beats are predictable from beat history** | Beat model: Brier 0.169 vs 0.185 base rate, AUC 0.69 on 2,157 held-out releases | High |
| 6 | **Polymarket may slightly underprice beats** | Actual beat rate 77.8% vs avg price 75.6%; buy-YES-everything +2.2¢/contract, 95% CI [+0.0¢, +4.4¢] | Low — borderline, pre-spread |

---

## 2. Model vs. market backtest

**Setup** (`src/modeling/backtest.py`)
- 1,111 resolved Polymarket "Will X beat quarterly earnings?" contracts, Oct 2025 – Oct 2026, 363 companies.
- **Market probability:** last traded YES price before 00:00 UTC on the release date (the prior US evening — before both pre-market and after-close announcements).
- **Model probability:** next-quarter beat model, **walk-forward** — retrained each calendar quarter using only outcomes public before that quarter. No contract is scored by a model that saw its result.
- **Outcome:** how the contract actually resolved.

**Accuracy on the same 1,111 contracts**

| Forecast | Brier ↓ | Log loss ↓ | ROC AUC ↑ |
|---|---|---|---|
| **Polymarket** | **0.137** | **0.430** | **0.786** |
| Beat model | 0.163 | 0.505 | 0.641 |
| Base rate (~75%) | 0.173 | 0.531 | 0.500 |

**Calibration — the market is well calibrated**

| Market price bucket | Contracts | Avg price | Actual beat rate |
|---|---|---|---|
| 0–20% | 25 | 10% | 16% |
| 20–40% | 87 | 31% | 38% |
| 40–60% | 101 | 51% | 53% |
| 60–80% | 263 | 72% | 76% |
| 80–100% | 635 | 90% | 91% |

Note that the actual beat rate sits slightly *above* the price in every bucket — the source of finding #6.

**Trading simulation** (buy the side the model favors when it disagrees with the market by more than a threshold; $1 payout contracts, no fees or spread)

| Threshold | Trades | P&L per trade | 95% CI |
|---|---|---|---|
| 10 pts | 574 | +2.5¢ | [−0.5¢, +5.5¢] |
| 20 pts | 206 | +3.4¢ | [−2.4¢, +9.2¢] |
| 10 pts, buy-YES side only | 235 | +5.8¢ | [−0.1¢, +11.8¢] |
| Buy YES on every contract | 1,111 | +2.2¢ | [+0.0¢, +4.4¢] |

None of these is a demonstrated edge: every interval touches zero, and real execution would pay the bid/ask spread. These markets are thin (median volume ≈ $6,000; 44% under $5,000), so spreads of several cents are common — enough to erase all of the above. The buy-YES results all point the same way (beats slightly underpriced), which makes it the one lead worth re-testing as more contracts resolve, ideally with order-book prices instead of last trades.

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
| Real API keys pasted into tracked `.env.example` | Keys published on commit | Caught before commit; reset to placeholders |

Cost controls: extraction cache keyed by (model, filing), `--cached-only` training flag, and a switch from Sonnet to Haiku cut per-run extraction cost ~20×.

---

## 6. Limitations

- **One year of contracts.** Polymarket's earnings markets only go back to late 2025; 1,111 contracts is enough to rank forecasts but thin for trading conclusions.
- **Last-trade prices**, not executable bid/ask. Real fills would be worse.
- **Consensus mismatch.** Polymarket's threshold is the consensus at market creation (sometimes GAAP, sometimes non-GAAP); the model learns from Yahoo's consensus. Some disagreement is definitional, not informational.
- **LLM study covers 20 large caps.** Text features might matter more for smaller, less-covered companies — untested (budget).
- **Price-reaction label is noisy** (±1 day abnormal return vs. SPY, no factor model).

---

## 7. What would be worth doing next

1. **Re-test the beat-underpricing lead** as more contracts resolve, using CLOB order-book prices (best ask) instead of last trades. This is the only result that could become a real edge.
2. **Use the LLM on what the market doesn't price well:** earnings-call *Q&A* transcripts (not press releases), or guidance language for small caps with little analyst coverage.
3. **Kalshi/Polymarket "mention" markets** ("Will the CEO say X on the call?") — a direct fit for transcript NLP.

---

## Reproducing

```bash
python -m src.modeling.beat_model     # beat model (free; Yahoo data)
python -m src.modeling.backtest       # §2 (free; Polymarket + Yahoo)
python -m src.modeling.train --tickers AAPL MSFT GOOGL AMZN META NVDA TSLA JPM V JNJ WMT XOM PG MA HD CVX KO PEP ABBV COST \
    --lookback-days 1095 --cached-only  # §3 (free with the extraction cache; otherwise ~$0.25 on Haiku)
python -m src.pipeline                # live signals vs. open contracts (free)
streamlit run src/dashboard/app.py    # Signals + Backtest pages
```

Per-contract backtest results: `data/processed/backtest_results.parquet`. Training set: `data/processed/training_set.parquet`.
