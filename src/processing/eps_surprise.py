"""EPS surprise vs. analyst consensus, from Yahoo Finance earnings history.

The market reacts to results *relative to expectations*; transcript tone
alone can't capture that. For each earnings event we attach:

    eps_surprise_pct = (reported EPS - consensus estimate) / |estimate| * 100
    eps_beat         = 1 if reported EPS > estimate, else 0
    trailing_beat_rate = share of the last TRAILING_QUARTERS releases (up to
                         and including this one) that beat consensus

Earnings dates come from Yahoo and are matched to EDGAR filing dates
within MATCH_WINDOW_DAYS (the 8-K is normally filed the same day as the
release; 10-Qs a few days later).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MATCH_WINDOW_DAYS = 7
# Surprise % explodes when the estimate is near zero (e.g. +900%); clip so a
# handful of tiny-denominator quarters don't dominate the model.
SURPRISE_CLIP_PCT = 50.0
# Quarters of history to request per ticker (~6 years; covers a 3-year
# lookback with room to spare).
HISTORY_QUARTERS = 25
# Beats are persistent (some companies guide conservatively and beat nearly
# every quarter), so recent beat history is a strong next-quarter prior.
TRAILING_QUARTERS = 4


def fetch_eps_history(ticker: str) -> pd.DataFrame:
    """Return [earnings_date, eps_estimate, reported_eps, eps_surprise_pct]."""
    import yfinance as yf  # imported lazily — optional/heavy dependency

    try:
        raw = yf.Ticker(ticker).get_earnings_dates(limit=HISTORY_QUARTERS)
    except Exception as e:
        logger.warning("EPS history unavailable for %s: %s", ticker, e)
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.rename(
        columns={
            "EPS Estimate": "eps_estimate",
            "Reported EPS": "reported_eps",
            "Surprise(%)": "eps_surprise_pct",
        }
    ).dropna(subset=["eps_estimate", "reported_eps"])
    df["earnings_date"] = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[["earnings_date", "eps_estimate", "reported_eps", "eps_surprise_pct"]].reset_index(drop=True)


def attach_eps_surprise(events: pd.DataFrame) -> pd.DataFrame:
    """Add eps_surprise_pct / eps_beat / trailing_beat_rate columns to a frame with [ticker, filing_date].

    Events with no Yahoo earnings date within MATCH_WINDOW_DAYS get NaN,
    which encode_features treats as "no surprise information" (0).
    """
    out = events.copy()
    out["eps_surprise_pct"] = np.nan
    out["eps_beat"] = np.nan
    out["trailing_beat_rate"] = np.nan
    filing_dates = pd.to_datetime(out["filing_date"])

    matched = 0
    for ticker in out["ticker"].unique():
        history = fetch_eps_history(ticker)
        if history.empty:
            continue
        for idx in out.index[out["ticker"] == ticker]:
            gaps = (history["earnings_date"] - filing_dates[idx]).abs()
            nearest = gaps.idxmin()
            if gaps[nearest] > pd.Timedelta(days=MATCH_WINDOW_DAYS):
                continue
            row = history.loc[nearest]
            out.at[idx, "eps_surprise_pct"] = float(
                np.clip(row["eps_surprise_pct"], -SURPRISE_CLIP_PCT, SURPRISE_CLIP_PCT)
            )
            out.at[idx, "eps_beat"] = float(row["reported_eps"] > row["eps_estimate"])
            prior = history[history["earnings_date"] <= row["earnings_date"]].nlargest(
                TRAILING_QUARTERS, "earnings_date"
            )
            out.at[idx, "trailing_beat_rate"] = float(
                (prior["reported_eps"] > prior["eps_estimate"]).mean()
            )
            matched += 1

    logger.info("Matched EPS surprise for %d/%d events", matched, len(out))
    return out
