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


class TestNewTranscriptFeatures:
    def test_extra_fields_and_quarter_over_quarter_change(self):
        events = pd.DataFrame(
            [
                {"ticker": "AAPL", "filing_date": "2026-01-29", "sentiment_score": 0.8,
                 "forward_guidance": "raised", "management_tone": "confident",
                 "revenue_surprise_indicator": "beat", "analyst_sentiment": "positive",
                 "risk_flags": ["fx"]},
                {"ticker": "AAPL", "filing_date": "2026-04-30", "sentiment_score": 0.5,
                 "forward_guidance": "maintained", "management_tone": "cautious",
                 "revenue_surprise_indicator": "miss", "analyst_sentiment": "negative",
                 "risk_flags": ["fx", "tariffs", "china"]},
            ]
        )
        feats = encode_features(events, _sample_macro()).sort_values("event_date")
        first, second = feats.iloc[0], feats.iloc[1]
        assert first["sentiment_change"] == 0.0  # no prior quarter
        assert abs(second["sentiment_change"] - (-0.3)) < 1e-9
        assert second["forward_guidance_change"] == -1.0
        assert second["revenue_surprise_score"] == -1.0
        assert second["analyst_sentiment_score"] == -1.0
        assert second["risk_flag_count"] == 3.0


class TestEpsSurprise:
    def test_matches_nearest_earnings_date_and_clips(self, monkeypatch):
        from src.processing import eps_surprise

        history = pd.DataFrame(
            {
                "earnings_date": pd.to_datetime(["2026-01-29", "2026-04-30"]),
                "eps_estimate": [2.00, 0.01],
                "reported_eps": [2.10, 0.20],
                "eps_surprise_pct": [5.0, 1900.0],
            }
        )
        monkeypatch.setattr(eps_surprise, "fetch_eps_history", lambda t: history)
        events = pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "AAPL"],
                "filing_date": ["2026-01-30", "2026-05-01", "2026-03-15"],
            }
        )
        out = eps_surprise.attach_eps_surprise(events)
        assert out.loc[0, "eps_surprise_pct"] == 5.0
        assert out.loc[0, "eps_beat"] == 1.0
        assert out.loc[1, "eps_surprise_pct"] == eps_surprise.SURPRISE_CLIP_PCT
        assert np.isnan(out.loc[2, "eps_surprise_pct"])  # no release within window


class TestContractMatching:
    def test_matches_latest_filing_before_release_and_skips_stale(self):
        scored = pd.DataFrame(
            {
                "ticker": ["CFG", "CFG", "MTB"],
                "event_date": ["2026-04-15", "2026-07-16", "2026-01-15"],
                "model_prob": [0.40, 0.90, 0.70],
            }
        )
        contracts = pd.DataFrame(
            {
                "ticker": ["CFG", "MTB"],
                "event_date": ["2026-10-16", "2026-10-16"],
                "platform": ["polymarket", "polymarket"],
                "market_probability": [0.70, 0.70],
                "volume": [100, 100],
            }
        )
        out = flag_mispricings(scored, contracts, threshold=0.10)
        # CFG pairs with its July filing (latest before release); MTB's only
        # filing is 9 months old — stale, so no signal row at all.
        assert list(out["ticker"]) == ["CFG"]
        assert out.iloc[0]["filing_date"] == pd.Timestamp("2026-07-16")
        assert np.isclose(out.iloc[0]["divergence"], 0.20)


class TestPolymarketReleaseDate:
    def test_release_date_parsed_from_slug(self):
        from src.ingestion.contract_client import _release_date_from_slug

        assert _release_date_from_slug("cfg-quarterly-earnings-nongaap-eps-10-16-2026-1pt40") == "2026-10-16"
        assert _release_date_from_slug("some-other-market") == ""


class TestBeatModelFeatures:
    def test_release_features_and_next_quarter_label(self):
        from src.modeling.beat_model import release_features

        history = pd.DataFrame(
            {
                "earnings_date": pd.to_datetime(
                    ["2025-01-15", "2025-04-15", "2025-07-15", "2025-10-15", "2026-06-15"]
                ),
                "eps_estimate": [1.0, 1.0, 1.0, 1.0, 1.0],
                "reported_eps": [1.1, 1.2, 0.9, 1.05, 1.3],
                "eps_surprise_pct": [10.0, 20.0, -10.0, 5.0, 30.0],
            }
        )
        f = release_features(history)
        assert list(f["eps_beat"]) == [1.0, 1.0, 0.0, 1.0, 1.0]
        assert list(f["beat_streak"]) == [1.0, 2.0, 0.0, 1.0, 2.0]
        assert f.loc[3, "trailing_beat_rate"] == 0.75
        # next_beat is the following release's outcome; NaN when the next
        # release is > 1 quarter away (2025-10 → 2026-06) or doesn't exist.
        assert f.loc[0, "next_beat"] == 1.0
        assert f.loc[1, "next_beat"] == 0.0
        assert np.isnan(f.loc[3, "next_beat"])
        assert np.isnan(f.loc[4, "next_beat"])


class TestBacktestScoring:
    def test_trade_simulation_pnl(self):
        from src.modeling.backtest import simulate_trades

        bt = pd.DataFrame(
            {
                "model_prob": [0.9, 0.2, 0.55],
                "market_prob": [0.6, 0.7, 0.50],
                "outcome": [1, 1, 0],
            }
        )
        out = simulate_trades(bt, threshold=0.10)
        # Row 0: buy YES at 0.60, wins → +0.40. Row 1: buy NO at 0.30, loses → -0.30.
        # Row 2: divergence 0.05 is under the threshold → no trade.
        assert out["trades"] == 2 and out["buy_yes"] == 1 and out["buy_no"] == 1
        assert np.isclose(out["total_pnl"], 0.10)
        assert np.isclose(out["return_on_cost"], 0.10 / 0.90)


class TestSlugParsing:
    def test_negative_consensus_and_basis(self):
        from src.ingestion.contract_client import _eps_terms_from_slug, _release_date_from_slug

        slug = "aal-quarterly-earnings-nongaap-eps-10-22-2026-neg0pt34"
        assert _release_date_from_slug(slug) == "2026-10-22"
        assert _eps_terms_from_slug(slug) == ("nongaap", -0.34)
        assert _eps_terms_from_slug("mtb-quarterly-earnings-gaap-eps-10-16-2026-5pt00") == ("gaap", 5.0)
        assert _eps_terms_from_slug("will-texas-roadhouse-txrh-beat-quarterly-earnings") == ("", None)


class TestOrderBookSummary:
    def test_top_of_book_and_depth(self):
        from src.ingestion.orderbook_collector import summarize_book

        out = summarize_book(
            {
                "bids": [{"price": "0.60", "size": "50"}, {"price": "0.57", "size": "10"}, {"price": "0.01", "size": "20"}],
                "asks": [{"price": "0.66", "size": "5"}, {"price": "0.99", "size": "100"}],
                "last_trade_price": "0.40",
            }
        )
        assert out["best_bid"] == 0.60 and out["best_ask"] == 0.66
        assert np.isclose(out["spread"], 0.06) and np.isclose(out["mid"], 0.63)
        assert out["bid_depth_5c"] == 60.0  # 0.60 and 0.57 levels; 0.01 excluded
        assert out["ask_depth_5c"] == 5.0

    def test_one_sided_book(self):
        from src.ingestion.orderbook_collector import summarize_book

        out = summarize_book({"bids": [{"price": "0.60", "size": "50"}], "asks": []})
        assert out["best_ask"] is None and out["spread"] is None and out["mid"] is None
