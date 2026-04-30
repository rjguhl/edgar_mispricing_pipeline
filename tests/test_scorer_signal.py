"""Tests for scorer + signal modules."""

import os

os.environ["USE_FIXTURES"] = "true"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.modeling.scorer import (  # noqa: E402
    ALL_FEATURES,
    encode_features,
    score,
)
from src.modeling.signal import flag_mispricings  # noqa: E402


def _sample_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "filing_date": "2026-02-01",
                "sentiment_score": 0.6,
                "forward_guidance": "raised",
                "management_tone": "confident",
            },
            {
                "ticker": "TSLA",
                "filing_date": "2026-02-01",
                "sentiment_score": -0.5,
                "forward_guidance": "lowered",
                "management_tone": "cautious",
            },
        ]
    )


def _sample_macro() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=3, freq="MS")
    return pd.DataFrame(
        {
            "cpiaucsl_pct_change": [0.002, 0.003, 0.001],
            "unrate_change": [0.0, 0.1, -0.1],
            "t10y2y_level": [0.3, 0.35, 0.4],
            "umcsent_level": [70.0, 71.0, 72.0],
        },
        index=idx,
    )


class TestEncodeFeatures:
    def test_columns_present(self):
        feats = encode_features(_sample_events(), _sample_macro())
        for col in ALL_FEATURES:
            assert col in feats.columns
        assert len(feats) == 2

    def test_guidance_encoding(self):
        feats = encode_features(_sample_events(), _sample_macro())
        aapl = feats[feats.ticker == "AAPL"].iloc[0]
        tsla = feats[feats.ticker == "TSLA"].iloc[0]
        assert aapl["forward_guidance_score"] == 1.0
        assert tsla["forward_guidance_score"] == -1.0


class TestScore:
    def test_heuristic_returns_probabilities(self):
        feats = encode_features(_sample_events(), _sample_macro())
        scored = score(feats, model_path="/tmp/does_not_exist.pkl")
        assert (scored["model_prob"] >= 0).all()
        assert (scored["model_prob"] <= 1).all()
        assert (scored["source"] == "heuristic").all()

    def test_bullish_scores_above_bearish(self):
        feats = encode_features(_sample_events(), _sample_macro())
        scored = score(feats, model_path="/tmp/does_not_exist.pkl")
        aapl = scored[scored.ticker == "AAPL"]["model_prob"].iloc[0]
        tsla = scored[scored.ticker == "TSLA"]["model_prob"].iloc[0]
        assert aapl > tsla


class TestFlagMispricings:
    def test_flags_when_above_threshold(self):
        scored = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "event_date": ["2026-02-01"],
                "model_prob": [0.8],
            }
        )
        contracts = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "event_date": ["2026-02-01"],
                "platform": ["kalshi"],
                "market_probability": [0.5],
                "volume": [100],
            }
        )
        out = flag_mispricings(scored, contracts, threshold=0.10)
        assert len(out) == 1
        assert out.iloc[0]["signal"] == "underpriced"
        assert np.isclose(out.iloc[0]["divergence"], 0.3)

    def test_no_signal_when_below_threshold(self):
        scored = pd.DataFrame(
            {"ticker": ["AAPL"], "event_date": ["2026-02-01"], "model_prob": [0.55]}
        )
        contracts = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "event_date": ["2026-02-01"],
                "platform": ["kalshi"],
                "market_probability": [0.50],
                "volume": [100],
            }
        )
        out = flag_mispricings(scored, contracts, threshold=0.10)
        assert out.iloc[0]["signal"] == "no_signal"

    def test_empty_inputs_return_empty_frame(self):
        out = flag_mispricings(pd.DataFrame(), pd.DataFrame())
        assert out.empty
        assert "signal" in out.columns
