"""Score new earnings events with the trained MispricingModel.

Two execution paths:

1. **Production**: load a serialized `MispricingModel` from disk and call
   `predict_proba` over a feature matrix.
2. **Demo / cold-start**: when no trained model is available, fall back to
   a transparent heuristic combining transcript sentiment, forward-guidance
   direction, and macro surprise. This keeps the dashboard meaningful before
   we have enough historical labels to train.

Inputs are aligned per (ticker, filing_date). Output is a DataFrame with
columns: ticker, event_date, model_prob, source ("model" | "heuristic").
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Expected feature columns produced upstream by transcript extraction +
# macro-feature build. Kept centralized so the schema is easy to evolve.
TRANSCRIPT_FEATURES = [
    "sentiment_score",
    "forward_guidance_score",
    "tone_score",
]
MACRO_FEATURES = [
    "cpi_pct_change",
    "unemployment_change",
    "yield_curve_level",
    "consumer_sentiment_level",
]
ALL_FEATURES = TRANSCRIPT_FEATURES + MACRO_FEATURES


# Encodings used by both the heuristic and the eventual training pipeline.
GUIDANCE_MAP = {"raised": 1.0, "maintained": 0.0, "lowered": -1.0, "withdrawn": -0.5, "not_mentioned": 0.0}
TONE_MAP = {"confident": 1.0, "neutral": 0.0, "cautious": -0.3, "defensive": -0.7, "evasive": -0.5}


def encode_features(events: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Build a model-ready feature matrix from raw extraction + macro frames.

    Args:
        events: rows from transcript extraction with columns:
            ticker, filing_date, sentiment_score, forward_guidance,
            management_tone
        macro: macro feature DataFrame indexed by observation_date with
            columns matching `MacroFeatureBuilder.build_features` output.

    Returns a DataFrame with one row per event and columns = ALL_FEATURES,
    plus the join keys (ticker, event_date).
    """
    ev = events.copy()
    ev["event_date"] = pd.to_datetime(ev["filing_date"])
    ev["forward_guidance_score"] = (
        ev.get("forward_guidance", "not_mentioned").map(GUIDANCE_MAP).fillna(0.0)
    )
    ev["tone_score"] = ev.get("management_tone", "neutral").map(TONE_MAP).fillna(0.0)
    ev["sentiment_score"] = ev.get("sentiment_score", 0.0).astype(float)

    # As-of join: pick the most recent macro row at or before each event date.
    macro_sorted = macro.sort_index().copy()
    macro_sorted.index = pd.to_datetime(macro_sorted.index)
    macro_aligned = (
        macro_sorted.reindex(macro_sorted.index.union(ev["event_date"].unique()))
        .sort_index()
        .ffill()
        .loc[ev["event_date"].unique()]
        .rename_axis("event_date")
        .reset_index()
    )

    # Map MacroFeatureBuilder column names to the names the model expects.
    rename = {
        "cpiaucsl_pct_change": "cpi_pct_change",
        "unrate_change": "unemployment_change",
        "t10y2y_level": "yield_curve_level",
        "umcsent_level": "consumer_sentiment_level",
    }
    for src, dst in rename.items():
        if src in macro_aligned.columns:
            macro_aligned[dst] = macro_aligned[src]

    keep = ["event_date"] + [c for c in MACRO_FEATURES if c in macro_aligned.columns]
    macro_aligned = macro_aligned[keep]

    feat = ev.merge(macro_aligned, on="event_date", how="left")

    # Ensure every feature column exists (missing macro features → 0).
    for col in ALL_FEATURES:
        if col not in feat.columns:
            feat[col] = 0.0
    feat[ALL_FEATURES] = feat[ALL_FEATURES].fillna(0.0)

    return feat[["ticker", "event_date"] + ALL_FEATURES]


def score(
    feature_df: pd.DataFrame,
    model_path: str | Path = "models/mispricing_model.pkl",
) -> pd.DataFrame:
    """Return calibrated probability of earnings beat per row in `feature_df`.

    Falls back to `_heuristic_proba` when the model file is absent.
    """
    keys = feature_df[["ticker", "event_date"]].copy()
    X = feature_df[ALL_FEATURES]

    path = Path(model_path)
    if path.exists():
        # Lazy-import so the heavy XGBoost / sklearn stack is only required
        # when an actual trained model is present on disk.
        from src.modeling.model import MispricingModel

        model = MispricingModel()
        model.load(str(path))
        probs = model.predict_proba(X)
        keys["model_prob"] = probs
        keys["source"] = "model"
        logger.info("Scored %d events with trained model", len(keys))
    else:
        keys["model_prob"] = _heuristic_proba(X)
        keys["source"] = "heuristic"
        logger.warning(
            "No trained model at %s — falling back to heuristic scoring", path
        )
    return keys


def _heuristic_proba(X: pd.DataFrame) -> np.ndarray:
    """Transparent fallback: weighted sum of standardized signals → sigmoid.

    Weights are hand-picked to be defensible *before* we have data; they
    will be replaced once `MispricingModel.train` is run on labeled history.
    """
    weights = {
        "sentiment_score": 0.8,
        "forward_guidance_score": 0.6,
        "tone_score": 0.3,
        "cpi_pct_change": -0.2,
        "unemployment_change": -0.4,
        "yield_curve_level": 0.15,
        "consumer_sentiment_level": 0.005,  # raw index, small coefficient
    }
    z = np.zeros(len(X))
    for col, w in weights.items():
        if col in X.columns:
            z = z + w * X[col].astype(float).values
    # Center the consumer-sentiment contribution so probs hover near 0.5
    # rather than saturating near 1.0.
    z = z - 0.35
    return 1.0 / (1.0 + np.exp(-z))
