"""Compare model probabilities to live market prices and emit mispricing signals."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)

# A filing older than ~1 quarter before the release is stale — it's from
# two earnings cycles back and shouldn't be matched to this contract.
MAX_FILING_AGE_DAYS = 120


def flag_mispricings(
    scored: pd.DataFrame,
    contracts: pd.DataFrame,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Join model scores with market contract prices and flag divergences.

    Args:
        scored: DataFrame with columns [ticker, event_date, model_prob].
        contracts: DataFrame with columns
            [ticker, event_date, platform, market_probability, volume].
        threshold: |model_prob - market_prob| > threshold flags a signal.
            Defaults to Config.MISPRICING_THRESHOLD.

    Each contract (event_date = expected earnings release) is matched to
    the ticker's latest scored filing on or before that date.

    Returns a DataFrame keyed by (ticker, event_date, platform) with
    divergence and signal columns. Multiple platforms per event produce
    multiple rows — useful for cross-venue arb visualization in the dashboard.
    """
    if threshold is None:
        threshold = Config.MISPRICING_THRESHOLD

    if scored.empty or contracts.empty:
        logger.warning("Empty input to flag_mispricings (scored=%d, contracts=%d)", len(scored), len(contracts))
        return pd.DataFrame(
            columns=[
                "ticker",
                "event_date",
                "filing_date",
                "platform",
                "model_prob",
                "market_prob",
                "divergence",
                "abs_divergence",
                "signal",
                "volume",
            ]
        )

    # Pair each contract with the company's most recent filing on or before
    # the contract's earnings release date (within MAX_FILING_AGE_DAYS) —
    # the latest information available when the contract is priced.
    s = scored.rename(columns={"event_date": "filing_date"}).copy()
    c = contracts.copy()
    s["filing_date"] = pd.to_datetime(s["filing_date"])
    c["event_date"] = pd.to_datetime(c["event_date"])

    merged = pd.merge_asof(
        c.sort_values("event_date"),
        s.sort_values("filing_date"),
        left_on="event_date",
        right_on="filing_date",
        by="ticker",
        direction="backward",
        tolerance=pd.Timedelta(days=MAX_FILING_AGE_DAYS),
    ).dropna(subset=["model_prob"])

    merged["market_prob"] = merged["market_probability"].astype(float)
    merged["divergence"] = merged["model_prob"] - merged["market_prob"]
    merged["abs_divergence"] = merged["divergence"].abs()
    merged["signal"] = np.where(
        merged["abs_divergence"] > threshold,
        np.where(merged["divergence"] > 0, "underpriced", "overpriced"),
        "no_signal",
    )

    out_cols = [
        "ticker",
        "event_date",
        "filing_date",
        "platform",
        "model_prob",
        "market_prob",
        "divergence",
        "abs_divergence",
        "signal",
        "volume",
    ]
    result = merged[out_cols].sort_values(
        ["abs_divergence"], ascending=False, kind="mergesort"
    )

    n_signals = (result["signal"] != "no_signal").sum()
    logger.info(
        "Flagged %d mispricings out of %d contract rows (threshold=%.2f)",
        n_signals,
        len(result),
        threshold,
    )
    return result


def write_signals(
    signals: pd.DataFrame,
    output_path: str | Path = "data/processed/signals.parquet",
) -> Path:
    """Persist the signals frame for the dashboard to consume.

    We write parquet for the canonical store and a JSON sidecar for quick
    inspection / debugging.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = signals.copy()
    for col in ("event_date", "filing_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")

    try:
        df.to_parquet(out, index=False)
    except Exception as e:  # pyarrow / fastparquet not installed → JSON only
        logger.warning("Parquet write failed (%s); falling back to JSON only", e)

    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(df.to_dict(orient="records"), indent=2))
    logger.info("Wrote %d signals to %s", len(df), out)
    return out
