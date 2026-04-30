"""End-to-end local pipeline runner.

Lets you exercise the whole thing without Airflow:

    python -m src.pipeline

In USE_FIXTURES mode (default) this completes in <1 second and produces
data/processed/signals.{parquet,json} that the Streamlit dashboard reads.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import Config
from src.ingestion.contract_client import ContractClient
from src.ingestion.edgar_client import EdgarClient
from src.modeling.scorer import GUIDANCE_MAP, TONE_MAP, encode_features, score
from src.modeling.signal import flag_mispricings, write_signals
from src.processing.macro_features import MacroFeatureBuilder

# Heavy/optional deps (fredapi, anthropic) are imported lazily inside the
# functions that need them so fixture-mode runs work without those installs.

logger = logging.getLogger(__name__)


def run(
    tickers: list[str] | None = None,
    use_fixtures: bool | None = None,
) -> pd.DataFrame:
    """Run the full pipeline and return the signals DataFrame."""
    use_fixtures = Config.USE_FIXTURES if use_fixtures is None else use_fixtures
    tickers = tickers or _default_tickers()

    # 1. Ingest
    edgar = EdgarClient(use_fixtures=use_fixtures)
    transcripts = edgar.fetch_transcripts(tickers, lookback_days=Config.LOOKBACK_DAYS)

    macro_records = _load_fred(use_fixtures=use_fixtures)
    contracts = _load_contracts(use_fixtures=use_fixtures)

    if not transcripts:
        logger.warning("No transcripts available; aborting pipeline run")
        return pd.DataFrame()

    # 2. Extract structured features (Claude API in real mode, deterministic
    # canned values in fixture mode so we don't burn tokens during dev).
    if use_fixtures:
        extracted = _stub_extract(transcripts)
    else:
        from src.processing.transcript_extractor import TranscriptExtractor

        extractor = TranscriptExtractor()
        extracted = extractor.extract_batch(transcripts)
    events = pd.DataFrame(extracted)

    # 3. Build macro features
    macro_df = MacroFeatureBuilder().build_features(pd.DataFrame(macro_records))

    # 4. Encode + score
    feature_df = encode_features(events, macro_df)
    scored = score(feature_df)

    # 5. Compare against market and emit signals
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


def _load_fred(use_fixtures: bool) -> list[dict]:
    if use_fixtures:
        path = Path(Config.FIXTURES_DIR) / "fred_indicators.json"
        return json.loads(path.read_text()) if path.exists() else []
    from src.ingestion.fred_client import FredClient

    return FredClient().fetch_indicators(lookback_days=Config.LOOKBACK_DAYS)


def _load_contracts(use_fixtures: bool) -> list[dict]:
    client = ContractClient(use_fixtures=use_fixtures)
    return client.fetch_all(event_types=["earnings"])


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


# Re-exported so dashboard layers can use the same encodings if they need
# to display a row's heuristic contribution breakdown.
__all__ = ["run", "GUIDANCE_MAP", "TONE_MAP"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    df = run()
    if not df.empty:
        print(df.to_string(index=False))
