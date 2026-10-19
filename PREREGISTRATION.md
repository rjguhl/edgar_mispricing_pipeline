# Pre-registration: Polymarket earnings-contract inefficiencies

**Registered:** 2026-10-19 (the commit adding this file is the timestamp of record). **Analysis date:** 2027-01-05. Hypotheses, rules, sample and decision criteria below are fixed before any of the evaluation data exists; the evaluation code will implement exactly this, and any deviation will be reported as such.

## Background

Exploratory analysis of 1,256 resolved contracts (Oct 2025 – Oct 2026, last-trade prices; see `src/research/beat_pricing.py` and REPORT.md §2a) found the market well calibrated overall, with two candidate inefficiencies:

- **H1 — non-GAAP beats are underpriced.** Non-GAAP contracts resolved YES 3.8¢ more often than priced (CI +1.3¢ to +6.2¢; the only slice significant after Benjamini-Hochberg correction). Plausible mechanism: management discretion over non-GAAP EPS makes beats more likely than consensus implies.
- **H2 — late price moves overshoot.** Betting against ≥ 10-point moves over the final 3 days earned +8.1¢ per contract (CI +3.4¢ to +12.9¢). Alternative explanation: bid-ask bounce in last-trade prices. Mid prices from order-book snapshots can tell these apart.

Both were found by searching the data, so they need an out-of-sample test at executable prices before being believed.

## Data

- Order-book snapshots every 30 minutes (`src/ingestion/orderbook_collector.py`, `orderbook-data` branch).
- **Sample:** every Polymarket "beat quarterly earnings" contract whose release date (from its slug) is between **2026-10-20 and 2026-12-31** (the first release date whose final snapshot postdates this registration), that resolves to exactly YES or NO.
- **Final snapshot:** the last snapshot before 00:00 UTC on the release date. **3-day snapshot:** the last snapshot before 00:00 UTC three days earlier. A snapshot more than 12 hours older than its cutoff counts as missing (collection gaps).
- **Tradable:** the final snapshot has both a bid and an ask and a spread ≤ 0.10.
- Contracts missing a required snapshot are excluded and counted.

## Rules (one $1-payout contract per signal; no sizing, no discretion)

**H1.** For each tradable non-GAAP contract: buy YES at the final snapshot's best ask. P&L = outcome − ask.

**H2.** For each tradable contract whose mid moved ≥ 0.10 between the 3-day and final snapshots (both snapshots need a bid and an ask):
- moved up → buy NO at 1 − best bid; P&L = (1 − outcome) − (1 − bid)
- moved down → buy YES at best ask; P&L = outcome − ask

## Primary tests

For each hypothesis: one-sided t-test that mean per-contract P&L > 0, at α = 0.025 (Bonferroni for two hypotheses). Report mean P&L, a 95% bootstrap CI, n, and the number of contracts excluded at each filter.

**Decision:** a hypothesis is *confirmed* only if its test rejects at α = 0.025 **and** its mean P&L is positive after paying the spread (which the rules above already do). Otherwise it is reported as not confirmed, regardless of how close it came.

## Secondary (descriptive, no decision attached)

- H1 and H2 measured at mid instead of executable prices (does the inefficiency exist even if it can't be traded?).
- Calibration regression on final-snapshot mids.
- Spread and depth distributions; how spreads change in the final week.
- H2 using last-trade prices on the same contracts, to measure how much of the original effect was bid-ask bounce.
