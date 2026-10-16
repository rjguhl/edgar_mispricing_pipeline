"""Price-reaction model: do LLM-extracted filing features predict the stock's
post-earnings move?

    1. Pull EDGAR earnings filings (8-K Item 2.02 exhibits / 10-Q / 10-K)
       and keep one per company-quarter.
    2. Extract structured features with Claude (cached per filing; with
       --cached-only, uncached filings are skipped instead of paid for).
    3. Label each event: abnormal return vs. SPY over ±1 trading day
       (src.processing.price_labels).
    4. Add EPS surprise vs. consensus and FRED macro features.
    5. Evaluate on a chronological holdout, then fit on all events and save.

Run:
    python -m src.modeling.train --tickers AAPL MSFT GOOGL TSLA NVDA --lookback-days 1095 --cached-only

The model compared against market prices is the separate next-quarter beat
model in src.modeling.beat_model.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from src.config import Config
from src.ingestion.edgar_client import EdgarClient, dedupe_earnings_events
from src.ingestion.fred_client import FredClient
from src.modeling.model import MispricingModel
from src.modeling.scorer import ALL_FEATURES, encode_features
from src.processing.eps_surprise import attach_eps_surprise
from src.processing.macro_features import MacroFeatureBuilder
from src.processing.price_labels import label_events
from src.processing.transcript_extractor import TranscriptExtractor

logger = logging.getLogger(__name__)

MIN_TRAINING_SAMPLES = 30  # TimeSeriesSplit(n_splits=5) needs >=6; this is a
# much higher floor since a smaller real dataset won't generalize meaningfully.
HOLDOUT_FRACTION = 0.25  # most recent quarter of events held out for evaluation
TRAINING_SET_PATH = Path("data/processed/training_set.parquet")


def build_training_set(
    tickers: list[str], lookback_days: int, cached_only: bool = False
) -> pd.DataFrame:
    """Assemble (features, real price-surprise label) rows for training."""
    logger.info("Fetching historical EDGAR filings for %s (lookback=%dd)", tickers, lookback_days)
    filings = EdgarClient(use_fixtures=False).fetch_transcripts(tickers, lookback_days=lookback_days)
    transcripts = dedupe_earnings_events(filings)
    logger.info(
        "Deduped %d filings to %d earnings events (one per company per quarter)",
        len(filings), len(transcripts),
    )
    if not transcripts:
        logger.error("No EDGAR filings found — nothing to train on")
        return pd.DataFrame()

    logger.info("Extracting structured features from %d transcripts via Claude", len(transcripts))
    extracted = TranscriptExtractor().extract_batch(transcripts, cached_only=cached_only)
    events = pd.DataFrame(extracted)

    logger.info("Computing real post-earnings price-surprise labels")
    labeled = label_events(events)
    if labeled.empty:
        logger.error("No events had usable price history — nothing to train on")
        return pd.DataFrame()

    logger.info("Attaching EPS surprise vs. analyst consensus")
    labeled = attach_eps_surprise(labeled)

    logger.info("Fetching FRED macro indicators for the same window")
    macro_records = FredClient().fetch_indicators(lookback_days=lookback_days)
    macro_df = MacroFeatureBuilder().build_features(pd.DataFrame(macro_records))

    feature_df = encode_features(labeled, macro_df)
    feature_df["surprise_label"] = labeled["surprise_label"].astype(int).values
    # Chronological order: the model's TimeSeriesSplit calibration and the
    # holdout evaluation both assume rows are sorted by time.
    return feature_df.sort_values("event_date").reset_index(drop=True)


def evaluate_holdout(
    feature_df: pd.DataFrame,
    test_frac: float = HOLDOUT_FRACTION,
    label_col: str = "surprise_label",
) -> dict | None:
    """Fit on the earliest events, score on the most recent ones.

    A chronological split (not random) so the model is never tested on
    events that happened before ones it trained on. Compares against a
    naive baseline that always predicts the training set's majority class.
    """
    n_test = max(1, int(round(len(feature_df) * test_frac)))
    train, test = feature_df.iloc[:-n_test], feature_df.iloc[-n_test:]
    y_train, y_test = train[label_col].astype(int), test[label_col].astype(int)

    model = MispricingModel()
    try:
        model.train(train[ALL_FEATURES], y_train)
    except ValueError as e:
        logger.warning("Holdout evaluation skipped — training split too small: %s", e)
        return None

    probs = model.predict_proba(test[ALL_FEATURES])
    preds = (probs >= 0.5).astype(int)
    majority = int(y_train.mean() >= 0.5)

    metrics = {
        "n_train": len(train),
        "n_test": len(test),
        "test_period": f"{test['event_date'].min():%Y-%m-%d} to {test['event_date'].max():%Y-%m-%d}",
        "accuracy": accuracy_score(y_test, preds),
        "baseline_accuracy": accuracy_score(y_test, np.full(len(y_test), majority)),
        "brier": brier_score_loss(y_test, probs),
        "baseline_brier": brier_score_loss(y_test, np.full(len(y_test), y_train.mean())),
        # AUC is undefined when the test window has only one outcome class.
        "roc_auc": roc_auc_score(y_test, probs) if y_test.nunique() == 2 else float("nan"),
    }
    logger.info(
        "[%s] Holdout (%d train / %d test, %s): accuracy %.2f vs baseline %.2f | "
        "Brier %.3f vs baseline %.3f (lower is better) | ROC AUC %.2f",
        label_col, metrics["n_train"], metrics["n_test"], metrics["test_period"],
        metrics["accuracy"], metrics["baseline_accuracy"],
        metrics["brier"], metrics["baseline_brier"], metrics["roc_auc"],
    )
    return metrics


def train_and_save(
    feature_df: pd.DataFrame,
    model_path: str = "models/mispricing_model.pkl",
    label_col: str = "surprise_label",
) -> MispricingModel:
    if len(feature_df) < MIN_TRAINING_SAMPLES:
        raise ValueError(
            f"Only {len(feature_df)} labeled samples — need at least "
            f"{MIN_TRAINING_SAMPLES} for a model fit worth trusting. "
            "Widen --lookback-days or add more tickers."
        )

    X = feature_df[ALL_FEATURES]
    y = feature_df[label_col].astype(int)

    model = MispricingModel()
    model.train(X, y)
    model.save(model_path)
    logger.info("Trained on %d real labeled events; saved to %s", len(feature_df), model_path)
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"])
    parser.add_argument("--lookback-days", type=int, default=1095, help="~3 years of history")
    parser.add_argument("--model-path", default="models/mispricing_model.pkl")
    parser.add_argument(
        "--cached-only",
        action="store_true",
        help="Use only cached Claude extractions; never call the API (costs $0).",
    )
    args = parser.parse_args()

    if not Config.ANTHROPIC_API_KEY and not args.cached_only:
        raise SystemExit("ANTHROPIC_API_KEY is not set — required for transcript extraction.")
    if not Config.FRED_API_KEY:
        raise SystemExit("FRED_API_KEY is not set — required for macro features.")

    df = build_training_set(args.tickers, args.lookback_days, cached_only=args.cached_only)
    if not df.empty:
        TRAINING_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(TRAINING_SET_PATH)
        logger.info("Saved training set to %s", TRAINING_SET_PATH)
    # Price reaction to the filing itself (abnormal return > 0). The
    # next-quarter beat model compared against market prices is trained
    # separately by src.modeling.beat_model.
    if len(df) >= MIN_TRAINING_SAMPLES:
        evaluate_holdout(df)
    # Final model is refit on all events (including the holdout window).
    train_and_save(df, args.model_path)

