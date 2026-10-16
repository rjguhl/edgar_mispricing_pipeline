"""Next-quarter EPS beat model — the forecast compared against market prices.

Polymarket's "Will X beat quarterly earnings?" contracts pay out if the
company's next reported EPS exceeds the consensus estimate. This model
answers the same question from information available before the release:
the company's recent record against consensus.

Why EPS history only (no transcript features): in src.modeling.train the
Claude-extracted text features added no out-of-sample signal on top of EPS
surprise, and dropping them means the model can train on every company
Polymarket has ever listed (~400) for free — no EDGAR downloads, no API
calls — instead of the 20 large caps we have extractions for. Training on
the same population the contracts cover matters: large caps beat ~85% of
the time, the mid caps and regional banks Polymarket lists much less.

Each row is one earnings release t for one company:
    eps_surprise_pct        surprise at t, % vs consensus (clipped)
    eps_beat                1 if t beat consensus
    trailing_beat_rate      share of the last 4 releases (incl. t) that beat
    trailing_mean_surprise  mean clipped surprise over the last 4 releases
    beat_streak             consecutive beats ending at t (capped)
Label: next_beat = 1 if release t+1 beat consensus (t+1 within ~1 quarter).

Run:
    python -m src.modeling.beat_model
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.modeling.model import MispricingModel
from src.modeling.scorer import BEAT_MODEL_PATH
from src.processing.eps_surprise import SURPRISE_CLIP_PCT, TRAILING_QUARTERS, fetch_eps_history

logger = logging.getLogger(__name__)

BEAT_FEATURES = [
    "eps_surprise_pct",
    "eps_beat",
    "trailing_beat_rate",
    "trailing_mean_surprise",
    "beat_streak",
]
MAX_STREAK = 8
NEXT_RELEASE_MAX_GAP_DAYS = 120  # "next quarter" must be within ~1 quarter
HOLDOUT_FRACTION = 0.25
MIN_TRAINING_SAMPLES = 200

# Yahoo earnings history is cached per ticker for a day so retrains and
# pipeline runs don't re-download ~400 histories.
EPS_CACHE_DIR = Path("data/processed/eps_history")
EPS_CACHE_MAX_AGE_SECONDS = 24 * 3600
YAHOO_REQUEST_INTERVAL_SECONDS = 0.3


def load_eps_history(ticker: str) -> pd.DataFrame:
    """fetch_eps_history with a one-day on-disk cache."""
    path = EPS_CACHE_DIR / f"{ticker.replace('.', '-')}.parquet"
    if path.exists() and time.time() - path.stat().st_mtime < EPS_CACHE_MAX_AGE_SECONDS:
        return pd.read_parquet(path)
    time.sleep(YAHOO_REQUEST_INTERVAL_SECONDS)
    history = fetch_eps_history(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(path)
    return history


def release_features(history: pd.DataFrame) -> pd.DataFrame:
    """One row per reported release with BEAT_FEATURES and the next_beat label."""
    if history.empty:
        return pd.DataFrame()
    df = history.sort_values("earnings_date").reset_index(drop=True)
    df["eps_beat"] = (df["reported_eps"] > df["eps_estimate"]).astype(float)
    df["eps_surprise_pct"] = df["eps_surprise_pct"].clip(-SURPRISE_CLIP_PCT, SURPRISE_CLIP_PCT)
    df["trailing_beat_rate"] = df["eps_beat"].rolling(TRAILING_QUARTERS, min_periods=1).mean()
    df["trailing_mean_surprise"] = (
        df["eps_surprise_pct"].rolling(TRAILING_QUARTERS, min_periods=1).mean()
    )
    # Streak: consecutive beats ending at each row (resets on a miss).
    miss_groups = (df["eps_beat"] == 0).cumsum()
    df["beat_streak"] = (
        df.groupby(miss_groups)["eps_beat"].cumsum().clip(upper=MAX_STREAK)
    )

    df["next_beat"] = df["eps_beat"].shift(-1)
    # When the label becomes known — walk-forward backtests train only on
    # rows whose outcome was public before the contract being scored.
    df["next_earnings_date"] = df["earnings_date"].shift(-1)
    gap = df["next_earnings_date"] - df["earnings_date"]
    df.loc[gap > pd.Timedelta(days=NEXT_RELEASE_MAX_GAP_DAYS), "next_beat"] = np.nan
    df[BEAT_FEATURES] = df[BEAT_FEATURES].fillna(0.0)
    return df


def build_beat_dataset(tickers: list[str]) -> pd.DataFrame:
    """Labeled training rows across `tickers`, sorted chronologically."""
    frames = []
    for i, ticker in enumerate(tickers, 1):
        feats = release_features(load_eps_history(ticker))
        if not feats.empty:
            feats.insert(0, "ticker", ticker)
            frames.append(feats)
        if i % 50 == 0:
            logger.info("EPS history loaded for %d/%d tickers", i, len(tickers))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).dropna(subset=["next_beat"])
    return df.sort_values("earnings_date").reset_index(drop=True)


def latest_beat_features(tickers: list[str]) -> pd.DataFrame:
    """Features as of each ticker's most recent reported release — the
    inputs for forecasting its *next* (upcoming) release. Columns:
    ticker, event_date (= that latest release date), BEAT_FEATURES."""
    rows = []
    for ticker in tickers:
        feats = release_features(load_eps_history(ticker))
        if feats.empty:
            continue
        last = feats.iloc[-1]
        rows.append({"ticker": ticker, "event_date": last["earnings_date"], **last[BEAT_FEATURES].to_dict()})
    return pd.DataFrame(rows, columns=["ticker", "event_date"] + BEAT_FEATURES)


def score_upcoming(tickers: list[str], model_path: str = BEAT_MODEL_PATH) -> pd.DataFrame:
    """P(beat at the next release) per ticker, from its latest release.

    Returns [ticker, event_date, model_prob] where event_date is the latest
    reported release — flag_mispricings matches it to the upcoming contract.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"No beat model at {model_path} — train it with `python -m src.modeling.beat_model`."
        )
    feats = latest_beat_features(tickers)
    model = MispricingModel()
    model.load(model_path)
    scored = feats[["ticker", "event_date"]].copy()
    scored["model_prob"] = model.predict_proba(feats[BEAT_FEATURES]) if len(feats) else []
    logger.info("Scored next-quarter beat probability for %d/%d tickers", len(scored), len(tickers))
    return scored


def evaluate_holdout(df: pd.DataFrame) -> dict:
    """Chronological holdout vs. a base-rate baseline (always predict the
    training set's beat rate)."""
    from sklearn.metrics import brier_score_loss, roc_auc_score

    n_test = int(round(len(df) * HOLDOUT_FRACTION))
    train, test = df.iloc[:-n_test], df.iloc[-n_test:]
    model = MispricingModel()
    model.train(train[BEAT_FEATURES], train["next_beat"].astype(int))
    probs = model.predict_proba(test[BEAT_FEATURES])
    y = test["next_beat"].astype(int)
    base = train["next_beat"].mean()

    metrics = {
        "n_train": len(train),
        "n_test": len(test),
        "test_period": f"{test['earnings_date'].min():%Y-%m-%d} to {test['earnings_date'].max():%Y-%m-%d}",
        "base_rate": base,
        "brier": brier_score_loss(y, probs),
        "baseline_brier": brier_score_loss(y, np.full(len(y), base)),
        "roc_auc": roc_auc_score(y, probs),
    }
    logger.info(
        "Beat model holdout (%d train / %d test, %s): Brier %.4f vs base-rate %.4f "
        "(lower is better) | ROC AUC %.3f | train beat rate %.2f",
        metrics["n_train"], metrics["n_test"], metrics["test_period"],
        metrics["brier"], metrics["baseline_brier"], metrics["roc_auc"], base,
    )
    return metrics


def train_beat_model(tickers: list[str], model_path: str = BEAT_MODEL_PATH) -> MispricingModel:
    df = build_beat_dataset(tickers)
    logger.info(
        "Beat dataset: %d releases across %d companies (beat rate %.2f)",
        len(df), df["ticker"].nunique() if not df.empty else 0,
        df["next_beat"].mean() if not df.empty else float("nan"),
    )
    if len(df) < MIN_TRAINING_SAMPLES:
        raise ValueError(f"Only {len(df)} labeled releases — need at least {MIN_TRAINING_SAMPLES}.")

    evaluate_holdout(df)
    model = MispricingModel()
    model.train(df[BEAT_FEATURES], df["next_beat"].astype(int))
    model.save(model_path)
    return model


def default_universe() -> list[str]:
    """Every Polymarket earnings ticker plus the project's S&P list."""
    from src.ingestion.contract_client import ContractClient

    universe = set(ContractClient(use_fixtures=False).fetch_polymarket_earnings_tickers())
    sp_path = Path(Config.SP500_TICKERS_PATH)
    if sp_path.exists():
        universe |= set(pd.read_csv(sp_path)["ticker"])
    return sorted(universe)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", help="Defaults to all Polymarket earnings tickers + S&P list")
    parser.add_argument("--model-path", default=BEAT_MODEL_PATH)
    args = parser.parse_args()
    train_beat_model(args.tickers or default_universe(), args.model_path)
