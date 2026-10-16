"""Real post-earnings price-surprise labels, derived from actual market data.

This is the training signal the model is meant to predict: not a sentiment
heuristic, but whether a stock's price reaction around a given filing date
was abnormally positive relative to the broader market.

Methodology (market-adjusted event study — the standard simple baseline;
a full factor model is out of scope here):
    raw_return      = close[t+1] / close[t-1] - 1
    benchmark_return = SPY close[t+1] / SPY close[t-1] - 1
    abnormal_return  = raw_return - benchmark_return
    surprise_label   = abnormal_return > SURPRISE_THRESHOLD

t-1/t+1 is a 1-trading-day window on each side of the filing date, the
standard short event-window used to capture the immediate earnings reaction
while minimizing contamination from unrelated news.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

BENCHMARK_TICKER = "SPY"
SURPRISE_THRESHOLD = 0.0  # abnormal_return > 0 counts as a positive surprise


def _download_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf  # imported lazily — optional/heavy dependency

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return df
    # yfinance can return a MultiIndex column frame for a single ticker
    # depending on version; flatten to simple "Close" access.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def compute_price_surprise(
    ticker: str,
    event_date: str,
    benchmark: str = BENCHMARK_TICKER,
    window_days: int = 1,
) -> dict | None:
    """Compute the market-adjusted price surprise around a single event.

    Args:
        ticker: equity ticker.
        event_date: filing date, "YYYY-MM-DD".
        benchmark: ticker to subtract the market-wide move with.
        window_days: trading-day half-window; reaction measured from
            close[t-window_days] to close[t+window_days].

    Returns a dict with raw_return, benchmark_return, abnormal_return,
    surprise_label, or None if price history isn't available (e.g. the
    event is too recent for a post-event close to exist yet).
    """
    event_dt = datetime.strptime(event_date, "%Y-%m-%d")
    start = (event_dt - timedelta(days=window_days * 4 + 5)).strftime("%Y-%m-%d")
    end = (event_dt + timedelta(days=window_days * 4 + 5)).strftime("%Y-%m-%d")

    stock = _download_history(ticker, start, end)
    bench = _download_history(benchmark, start, end)
    if stock.empty or bench.empty:
        logger.warning("No price history for %s around %s", ticker, event_date)
        return None

    stock_before, stock_after = _closes_around(stock, event_dt, window_days)
    bench_before, bench_after = _closes_around(bench, event_dt, window_days)
    if None in (stock_before, stock_after, bench_before, bench_after):
        logger.warning(
            "Insufficient trading days around %s for %s (event may be too recent)",
            event_date, ticker,
        )
        return None

    raw_return = stock_after / stock_before - 1
    benchmark_return = bench_after / bench_before - 1
    abnormal_return = raw_return - benchmark_return

    return {
        "ticker": ticker,
        "event_date": event_date,
        "raw_return": raw_return,
        "benchmark_return": benchmark_return,
        "abnormal_return": abnormal_return,
        "surprise_label": bool(abnormal_return > SURPRISE_THRESHOLD),
    }


def _closes_around(
    history: pd.DataFrame, event_dt: datetime, window_days: int
) -> tuple[float | None, float | None]:
    """Return (close just before, close just after) the event date."""
    idx = history.index
    before_mask = idx <= (event_dt - timedelta(days=window_days))
    after_mask = idx >= (event_dt + timedelta(days=window_days))

    before = history.loc[before_mask]
    after = history.loc[after_mask]
    if before.empty or after.empty:
        return None, None
    return float(before["Close"].iloc[-1]), float(after["Close"].iloc[0])


def label_events(events: pd.DataFrame) -> pd.DataFrame:
    """Attach price-surprise labels to a frame with [ticker, filing_date].

    Returns a new frame with one row per input event that had usable price
    history, plus raw_return/benchmark_return/abnormal_return/surprise_label.
    Events without enough history (e.g. too recent) are dropped.
    """
    rows = []
    for _, row in events.iterrows():
        label = compute_price_surprise(row["ticker"], row["filing_date"])
        if label is None:
            continue
        merged = row.to_dict()
        merged.update(label)
        rows.append(merged)

    out = pd.DataFrame(rows)
    logger.info(
        "Labeled %d/%d events with real post-earnings price surprises",
        len(out), len(events),
    )
    return out
