"""Live mispricing signals: beat model vs. open Polymarket earnings contracts.

    python -m src.pipeline

Live mode (USE_FIXTURES=false) scores every company with an upcoming
"beat quarterly earnings" contract — no Claude calls, no API keys needed.
Fixture mode (default) runs on bundled sample data in under a second.
Both write data/processed/signals.{parquet,json} for the dashboard.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import Config
from src.ingestion.contract_client import ContractClient
from src.ingestion.edgar_client import EdgarClient, dedupe_earnings_events
from src.modeling.scorer import encode_features, score
from src.modeling.signal import flag_mispricings, write_signals
from src.processing.macro_features import MacroFeatureBuilder

logger = logging.getLogger(__name__)


FIXTURE_LOOKBACK_DAYS = 3650


def run(
    tickers: list[str] | None = None,
    use_fixtures: bool | None = None,
) -> pd.DataFrame:
    """Run the full pipeline and return the signals DataFrame."""
    use_fixtures = Config.USE_FIXTURES if use_fixtures is None else use_fixtures

    # 1. Ingest. Contracts first: in live mode with no explicit tickers, we
    # score exactly the companies that currently have an open earnings
    # contract whose release hasn't happened yet.
    contracts = _upcoming_earnings_contracts(
        _load_contracts(use_fixtures=use_fixtures, tickers=tickers),
        use_fixtures=use_fixtures,
    )
    if not tickers:
        tickers = (
            sorted({c["ticker"] for c in contracts if c["ticker"]})
            if not use_fixtures
            else _default_tickers()
        )
    logger.info("Scoring %d tickers against %d upcoming earnings contracts", len(tickers), len(contracts))
    if not tickers:
        logger.warning("No upcoming earnings contracts to score; aborting pipeline run")
        return pd.DataFrame()

    # 2. Score. Live: next-quarter EPS beat probability from each company's
    # earnings history (src.modeling.beat_model) — the same question the
    # contracts price. Fixtures: the transcript-feature flow on bundled
    # sample filings with the transparent heuristic (no API calls).
    if use_fixtures:
        scored = _score_fixture_transcripts(tickers)
        if scored.empty:
            return pd.DataFrame()
    else:
        from src.modeling.beat_model import score_upcoming

        scored = score_upcoming(tickers)

    # 3. Compare against market and emit signals
    contracts_df = _contracts_to_frame(contracts)
    signals = flag_mispricings(scored, contracts_df)

    write_signals(signals)
    logger.info("Pipeline complete: %d signal rows", len(signals))
    return signals


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _default_tickers() -> list[str]:
    path = Path(Config.SP500_TICKERS_PATH)
    if not path.exists():
        return ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    return pd.read_csv(path)["ticker"].tolist()


def _load_fred_fixture() -> list[dict]:
    path = Path(Config.FIXTURES_DIR) / "fred_indicators.json"
    return json.loads(path.read_text()) if path.exists() else []


def _load_contracts(use_fixtures: bool, tickers: list[str]) -> list[dict]:
    client = ContractClient(use_fixtures=use_fixtures)
    return client.fetch_all(tickers=tickers)


def _score_fixture_transcripts(tickers: list[str]) -> pd.DataFrame:
    """Fixture-mode scoring: bundled transcripts → canned extraction →
    transcript features → heuristic probability."""
    edgar = EdgarClient(use_fixtures=True)
    # Bundled fixtures have fixed historical dates; don't let them age out
    # of the lookback window.
    transcripts = dedupe_earnings_events(
        edgar.fetch_transcripts(tickers, lookback_days=FIXTURE_LOOKBACK_DAYS)
    )
    if not transcripts:
        logger.warning("No transcripts available; aborting pipeline run")
        return pd.DataFrame()
    events = pd.DataFrame(_stub_extract(transcripts))
    macro_df = MacroFeatureBuilder().build_features(pd.DataFrame(_load_fred_fixture()))
    return score(encode_features(events, macro_df), model_path=None)


def _upcoming_earnings_contracts(contracts: list[dict], use_fixtures: bool) -> list[dict]:
    """Keep only EPS beat/miss contracts whose earnings release is still ahead.

    Kalshi's Company-KPI contracts (headcount, deliveries, ...) ask a
    different question than the beat model answers, so they're excluded
    from signals. Contracts whose release date has passed are awaiting
    settlement — the outcome is already public, so a "mispricing" there is
    meaningless. (Fixture contracts carry historical dates, so the date
    filter is skipped in fixture mode.)
    """
    today = pd.Timestamp.today().normalize()
    kept = []
    for c in contracts:
        if c.get("event_type") != "earnings_beat":
            continue
        if not use_fixtures and pd.to_datetime(c.get("event_date") or None) < today:
            continue
        kept.append(c)
    return kept


def _contracts_to_frame(contracts: list[dict]) -> pd.DataFrame:
    if not contracts:
        return pd.DataFrame(
            columns=["ticker", "event_date", "platform", "market_probability", "volume"]
        )
    df = pd.DataFrame(contracts)
    df = df.rename(columns={"price": "market_probability"})
    return df[["ticker", "event_date", "platform", "market_probability", "volume"]]


# Deterministic canned extraction so fixture runs don't hit the Claude API.
# Hand-tuned to match the sentiment cues in data/fixtures/edgar_transcripts.json
# so the dashboard tells a coherent story end-to-end.
_STUB_EXTRACTION = {
    "AAPL": {
        "overall_sentiment": "bullish",
        "sentiment_score": 0.62,
        "forward_guidance": "raised",
        "management_tone": "confident",
        "key_themes": ["services growth", "iPhone Pro demand", "margin expansion"],
        "revenue_surprise_indicator": "beat",
        "analyst_sentiment": "positive",
        "risk_flags": [],
    },
    "MSFT": {
        "overall_sentiment": "bullish",
        "sentiment_score": 0.71,
        "forward_guidance": "raised",
        "management_tone": "confident",
        "key_themes": ["Azure AI growth", "Copilot adoption", "elevated capex"],
        "revenue_surprise_indicator": "beat",
        "analyst_sentiment": "positive",
        "risk_flags": ["FX headwinds"],
    },
    "GOOGL": {
        "overall_sentiment": "bullish",
        "sentiment_score": 0.45,
        "forward_guidance": "maintained",
        "management_tone": "confident",
        "key_themes": ["Search AI Overviews", "Cloud acceleration", "ad softness"],
        "revenue_surprise_indicator": "beat",
        "analyst_sentiment": "mixed",
        "risk_flags": ["financial services ad weakness"],
    },
    "TSLA": {
        "overall_sentiment": "bearish",
        "sentiment_score": -0.42,
        "forward_guidance": "lowered",
        "management_tone": "cautious",
        "key_themes": ["margin compression", "Mexico ramp", "energy storage"],
        "revenue_surprise_indicator": "miss",
        "analyst_sentiment": "negative",
        "risk_flags": ["affordable model timing", "macro demand"],
    },
    "NVDA": {
        "overall_sentiment": "bullish",
        "sentiment_score": 0.85,
        "forward_guidance": "raised",
        "management_tone": "confident",
        "key_themes": ["Blackwell ramp", "sovereign AI", "data center growth"],
        "revenue_surprise_indicator": "beat",
        "analyst_sentiment": "positive",
        "risk_flags": ["export controls"],
    },
}


def _stub_extract(transcripts: list[dict]) -> list[dict]:
    out = []
    for t in transcripts:
        ticker = t["ticker"]
        feat = dict(_STUB_EXTRACTION.get(ticker, {}))
        feat.setdefault("overall_sentiment", "neutral")
        feat.setdefault("sentiment_score", 0.0)
        feat.setdefault("forward_guidance", "not_mentioned")
        feat.setdefault("management_tone", "neutral")
        feat.setdefault("key_themes", [])
        feat.setdefault("risk_flags", [])
        feat["ticker"] = ticker
        feat["filing_date"] = t["filing_date"]
        out.append(feat)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    df = run()
    if not df.empty:
        print(df.to_string(index=False))
