"""Backtest: beat model vs. Polymarket on resolved earnings contracts.

For every resolved "Will X beat quarterly earnings?" market:

    market_prob  last traded YES price before the release day starts
                 (00:00 UTC = prior US evening, so it's pre-announcement
                 for both pre-market and after-close reporters)
    model_prob   beat model's P(beat) from the company's EPS history
                 strictly before the release, using a model trained only
                 on outcomes that were public before the contract's
                 quarter (walk-forward; retrained each calendar quarter)
    outcome      how the market actually resolved (1 = beat)

Reports Brier score / log loss / ROC AUC for market, model, and a
base-rate forecast on the same contracts, a calibration table, and a
naive trading simulation (buy the side the model favors when it disagrees
with the market by more than a threshold; $1 payout contracts, ignoring
fees and bid/ask spread — real fills would be worse).

Run:
    python -m src.modeling.backtest
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.ingestion.contract_client import (
    ContractClient,
    _eps_terms_from_slug,
    _release_date_from_slug,
    _ticker_from_question,
)
from src.modeling.beat_model import (
    BEAT_FEATURES,
    MIN_TRAINING_SAMPLES,
    build_beat_dataset,
    load_eps_history,
    release_features,
)
from src.modeling.model import MispricingModel

logger = logging.getLogger(__name__)

CLOB_PRICES_URL = "https://clob.polymarket.com/prices-history"
PRICE_LOOKBACK_DAYS = 10  # history window requested before the release
PRICE_FIDELITY_MINUTES = 60
CLOB_REQUEST_INTERVAL_SECONDS = 0.2
PRICE_CACHE_DIR = Path("data/processed/polymarket_prices")
RESULTS_PATH = Path("data/processed/backtest_results.parquet")

MIN_PRIOR_RELEASE_GAP_DAYS = 14  # model inputs must predate the release by this much
TRADE_THRESHOLD = 0.10
PROB_CLIP = (0.01, 0.99)  # for log loss


# ----------------------------------------------------------------------
# Data assembly
# ----------------------------------------------------------------------


def resolved_contracts() -> pd.DataFrame:
    """[contract_id, ticker, release_date, eps_basis, eps_threshold, outcome,
    yes_token, volume] for
    resolved earnings markets with a parseable release date."""
    rows = []
    for m in ContractClient(use_fixtures=False).fetch_polymarket_earnings_markets(closed=True):
        release = _release_date_from_slug(m.get("slug", ""))
        try:
            outcome_prices = [float(p) for p in json.loads(m.get("outcomePrices") or "[]")]
            yes_token = json.loads(m.get("clobTokenIds") or "[]")[0]
        except (ValueError, IndexError, TypeError):
            continue
        # Resolved markets settle at exactly 1/0; skip anything else (voided/disputed).
        if not release or sorted(outcome_prices) != [0.0, 1.0]:
            continue
        basis, threshold = _eps_terms_from_slug(m.get("slug", ""))
        rows.append(
            {
                "contract_id": str(m.get("id")),
                "ticker": _ticker_from_question(m.get("question", "")),
                "release_date": pd.Timestamp(release),
                "eps_basis": basis,
                "eps_threshold": threshold,
                "outcome": int(outcome_prices[0] == 1.0),
                "yes_token": yes_token,
                "volume": float(m.get("volumeNum") or 0.0),
            }
        )
    df = pd.DataFrame(rows)
    logger.info("Resolved earnings contracts: %d", len(df))
    return df


def pre_release_price(contract_id: str, yes_token: str, release_date: pd.Timestamp) -> float | None:
    """Last YES price at or before 00:00 UTC on the release date (cached)."""
    path = PRICE_CACHE_DIR / f"{contract_id}.json"
    if path.exists():
        history = json.loads(path.read_text())
    else:
        end = int(release_date.tz_localize("UTC").timestamp())
        params = {
            "market": yes_token,
            "startTs": end - PRICE_LOOKBACK_DAYS * 86400,
            "endTs": end,
            "fidelity": PRICE_FIDELITY_MINUTES,
        }
        time.sleep(CLOB_REQUEST_INTERVAL_SECONDS)
        try:
            resp = requests.get(CLOB_PRICES_URL, params=params, timeout=30)
            resp.raise_for_status()
            history = resp.json().get("history", [])
        except (requests.RequestException, ValueError) as e:
            logger.warning("Price history unavailable for contract %s: %s", contract_id, e)
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(history))
    return float(history[-1]["p"]) if history else None


def model_inputs_before(ticker: str, release_date: pd.Timestamp) -> pd.Series | None:
    """BEAT_FEATURES as of the company's last release before this one."""
    feats = release_features(load_eps_history(ticker))
    if feats.empty:
        return None
    prior = feats[feats["earnings_date"] <= release_date - pd.Timedelta(days=MIN_PRIOR_RELEASE_GAP_DAYS)]
    return prior.iloc[-1][BEAT_FEATURES] if not prior.empty else None


def assemble(contracts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(contracts.itertuples(index=False), 1):
        price = pre_release_price(c.contract_id, c.yes_token, c.release_date)
        inputs = model_inputs_before(c.ticker, c.release_date) if c.ticker else None
        if price is not None and inputs is not None:
            rows.append({**c._asdict(), "market_prob": price, **inputs.to_dict()})
        if i % 200 == 0:
            logger.info("Assembled %d/%d contracts", i, len(contracts))
    df = pd.DataFrame(rows).drop(columns=["yes_token"])
    logger.info("Contracts with a pre-release price and model inputs: %d/%d", len(df), len(contracts))
    return df.sort_values("release_date").reset_index(drop=True)


def walk_forward_model_probs(bt: pd.DataFrame, training: pd.DataFrame) -> pd.Series:
    """Score each calendar quarter of contracts with a model trained only on
    releases whose next-quarter outcome was public before that quarter."""
    probs = pd.Series(np.nan, index=bt.index)
    for quarter, idx in bt.groupby(bt["release_date"].dt.to_period("Q")).groups.items():
        cutoff = quarter.start_time
        train = training[training["next_earnings_date"] < cutoff]
        if len(train) < MIN_TRAINING_SAMPLES:
            logger.warning("Skipping %s: only %d training rows before it", quarter, len(train))
            continue
        model = MispricingModel()
        model.train(train[BEAT_FEATURES], train["next_beat"].astype(int))
        probs[idx] = model.predict_proba(bt.loc[idx, BEAT_FEATURES])
        base = train["next_beat"].mean()
        bt.loc[idx, "base_rate_prob"] = base
        logger.info("%s: %d contracts, model trained on %d releases", quarter, len(idx), len(train))
    return probs


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------


def score_forecasts(bt: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    y = bt["outcome"]
    rows = []
    for name in ("market_prob", "model_prob", "base_rate_prob"):
        p = bt[name].clip(*PROB_CLIP)
        rows.append(
            {
                "forecast": name.replace("_prob", ""),
                "brier": brier_score_loss(y, p),
                "log_loss": log_loss(y, p),
                "roc_auc": roc_auc_score(y, p) if name != "base_rate_prob" else 0.5,
            }
        )
    return pd.DataFrame(rows)


def calibration_table(bt: pd.DataFrame, column: str, bins=(0, 0.2, 0.4, 0.6, 0.8, 1.0)) -> pd.DataFrame:
    cut = pd.cut(bt[column], bins=bins, include_lowest=True)
    return (
        bt.groupby(cut, observed=True)
        .agg(n=("outcome", "size"), avg_forecast=(column, "mean"), actual_beat_rate=("outcome", "mean"))
        .round(3)
    )


def simulate_trades(bt: pd.DataFrame, threshold: float = TRADE_THRESHOLD) -> dict:
    """Buy YES when model > market + threshold, NO when model < market -
    threshold. Each trade buys one $1-payout contract at the market price."""
    div = bt["model_prob"] - bt["market_prob"]
    yes = div > threshold
    no = div < -threshold
    pnl = pd.Series(0.0, index=bt.index)
    pnl[yes] = bt.loc[yes, "outcome"] - bt.loc[yes, "market_prob"]
    pnl[no] = (1 - bt.loc[no, "outcome"]) - (1 - bt.loc[no, "market_prob"])
    traded = yes | no
    cost = bt.loc[yes, "market_prob"].sum() + (1 - bt.loc[no, "market_prob"]).sum()
    return {
        "threshold": threshold,
        "trades": int(traded.sum()),
        "buy_yes": int(yes.sum()),
        "buy_no": int(no.sum()),
        "win_rate": float((pnl[traded] > 0).mean()) if traded.any() else float("nan"),
        "total_pnl": float(pnl[traded].sum()),
        "pnl_per_trade": float(pnl[traded].mean()) if traded.any() else float("nan"),
        "return_on_cost": float(pnl[traded].sum() / cost) if cost else float("nan"),
    }


def run_backtest() -> pd.DataFrame:
    contracts = resolved_contracts()
    bt = assemble(contracts)
    training = build_beat_dataset(sorted(set(contracts["ticker"]) - {""}))
    bt["model_prob"] = walk_forward_model_probs(bt, training)
    bt = bt.dropna(subset=["model_prob"]).reset_index(drop=True)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    bt.to_parquet(RESULTS_PATH)

    logger.info(
        "Backtest: %d contracts, %s to %s, actual beat rate %.2f",
        len(bt), f"{bt['release_date'].min():%Y-%m-%d}", f"{bt['release_date'].max():%Y-%m-%d}",
        bt["outcome"].mean(),
    )
    print("\nForecast accuracy on the same contracts (lower Brier / log loss is better):")
    print(score_forecasts(bt).round(4).to_string(index=False))
    print("\nCalibration — market:")
    print(calibration_table(bt, "market_prob").to_string())
    print("\nCalibration — model:")
    print(calibration_table(bt, "model_prob").to_string())
    print("\nTrading simulation (no fees/spread):")
    for threshold in (0.10, 0.20, 0.30):
        print(" ", {k: round(v, 3) if isinstance(v, float) else v for k, v in simulate_trades(bt, threshold).items()})
    print(f"\nPer-contract results saved to {RESULTS_PATH}")
    return bt


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    run_backtest()
