"""Feature encoding and scoring for filing-level (transcript) features.

`encode_features` turns Claude extractions + EPS surprise + FRED macro data
into the model matrix used by src.modeling.train. `score` applies a saved
model, or — in fixture mode, with no model — a transparent hand-weighted
heuristic so the keyless demo still produces sensible probabilities.

Output columns: ticker, event_date, model_prob, source ("model" | "heuristic").
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
    "revenue_surprise_score",
    "analyst_sentiment_score",
    "risk_flag_count",
]
# Quarter-over-quarter change vs. the same company's previous filing.
# Press releases are almost uniformly upbeat, so the *shift* in tone is
# more informative than its level. 0 when there's no prior event.
CHANGE_FEATURES = [
    "sentiment_change",
    "forward_guidance_change",
    "tone_change",
]
# Results vs. analyst consensus (src.processing.eps_surprise).
EPS_FEATURES = [
    "eps_surprise_pct",
    "eps_beat",
    "trailing_beat_rate",
]
MACRO_FEATURES = [
    "cpi_pct_change",
    "unemployment_change",
    "yield_curve_level",
    "consumer_sentiment_level",
]
# P(beat EPS consensus next quarter) — compared against prediction-market
# "beat quarterly earnings" prices. Trained by src.modeling.train.
BEAT_MODEL_PATH = "models/beat_model.pkl"

ALL_FEATURES = TRANSCRIPT_FEATURES + CHANGE_FEATURES + EPS_FEATURES + MACRO_FEATURES


# Categorical extraction fields → numeric scores.
GUIDANCE_MAP = {"raised": 1.0, "maintained": 0.0, "lowered": -1.0, "withdrawn": -0.5, "not_mentioned": 0.0}
TONE_MAP = {"confident": 1.0, "neutral": 0.0, "cautious": -0.3, "defensive": -0.7, "evasive": -0.5}
REVENUE_SURPRISE_MAP = {"beat": 1.0, "inline": 0.0, "miss": -1.0, "unclear": 0.0}
ANALYST_SENTIMENT_MAP = {"positive": 1.0, "mixed": 0.0, "neutral": 0.0, "negative": -1.0}


def _mapped(ev: pd.DataFrame, col: str, mapping: dict, default: str) -> pd.Series:
    """Encode a categorical extraction column; missing column/values → default."""
    raw = ev[col] if col in ev.columns else pd.Series(default, index=ev.index)
    return raw.fillna(default).map(mapping).fillna(0.0)


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
    ev["forward_guidance_score"] = _mapped(ev, "forward_guidance", GUIDANCE_MAP, "not_mentioned")
    ev["tone_score"] = _mapped(ev, "management_tone", TONE_MAP, "neutral")
    ev["sentiment_score"] = (
        ev["sentiment_score"].astype(float) if "sentiment_score" in ev.columns else 0.0
    )
    ev["revenue_surprise_score"] = _mapped(ev, "revenue_surprise_indicator", REVENUE_SURPRISE_MAP, "unclear")
    ev["analyst_sentiment_score"] = _mapped(ev, "analyst_sentiment", ANALYST_SENTIMENT_MAP, "neutral")
    ev["risk_flag_count"] = (
        ev["risk_flags"].apply(lambda f: float(len(f)) if isinstance(f, list) else 0.0)
        if "risk_flags" in ev.columns
        else 0.0
    )

    # Missing EPS match (or column absent) → 0 = "no surprise information".
    for col in EPS_FEATURES:
        ev[col] = ev[col].astype(float).fillna(0.0) if col in ev.columns else 0.0

    # Change vs. the same ticker's previous event (by date). diff() leaves
    # the first event per ticker as NaN → 0 ("no change known").
    ordered = ev.sort_values("event_date")
    for level, change in [
        ("sentiment_score", "sentiment_change"),
        ("forward_guidance_score", "forward_guidance_change"),
        ("tone_score", "tone_change"),
    ]:
        ev[change] = ordered.groupby("ticker")[level].diff().reindex(ev.index).fillna(0.0)

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
    model_path: str | Path | None = "models/mispricing_model.pkl",
) -> pd.DataFrame:
    """Return calibrated probability of earnings beat per row in `feature_df`.

    Falls back to `_heuristic_proba` when the model file is absent.
    """
    keys = feature_df[["ticker", "event_date"]].copy()
    X = feature_df[ALL_FEATURES]

    path = Path(model_path) if model_path is not None else None
    if path is not None and path.exists():
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
    """Fixture-mode fallback: hand-picked weighted sum of signals → sigmoid."""
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
