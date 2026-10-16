"""Smoke tests for the fixture/mock path of the ingestion clients."""

import os

# Force fixture mode before any project imports.
os.environ["USE_FIXTURES"] = "true"

from src.ingestion.contract_client import ContractClient  # noqa: E402
from src.ingestion.edgar_client import (  # noqa: E402
    EdgarClient,
    _html_to_text,
    _infer_fiscal_period,
)


class TestEdgarFixtures:
    def test_fetch_loads_fixture_records(self):
        records = EdgarClient(use_fixtures=True).fetch_transcripts(
            ["AAPL", "MSFT"], lookback_days=365
        )
        tickers = {r["ticker"] for r in records}
        assert tickers == {"AAPL", "MSFT"}
        for r in records:
            assert r["transcript_text"]
            assert "filing_date" in r

    def test_fetch_filters_unknown_tickers(self):
        records = EdgarClient(use_fixtures=True).fetch_transcripts(
            ["XYZ_DOES_NOT_EXIST"], lookback_days=365
        )
        assert records == []

    def test_html_to_text_strips_tags(self):
        html = "<html><body><p>Hello   <b>world</b></p></body></html>"
        assert _html_to_text(html) == "Hello world"

    def test_infer_fiscal_period_from_text(self):
        text = "Apple Inc. fiscal first quarter 2026 results were strong."
        q, y = _infer_fiscal_period(text, "2026-01-30")
        assert q == "Q1"
        assert y == 2026

    def test_infer_fiscal_period_fallback(self):
        # No quarter/year cues — should fall back to filing-date calendar quarter.
        q, y = _infer_fiscal_period("blah blah blah", "2026-04-30")
        assert q == "Q2"
        assert y == 2026


class TestContractFixtures:
    def test_kalshi_fixture_loads(self):
        records = ContractClient(use_fixtures=True).fetch_kalshi_contracts()
        assert all(r["platform"] == "kalshi" for r in records)
        assert records, "expected at least one kalshi fixture row"

    def test_polymarket_fixture_loads(self):
        records = ContractClient(use_fixtures=True).fetch_polymarket_contracts()
        assert all(r["platform"] == "polymarket" for r in records)
        assert records

    def test_fetch_all_combines_platforms(self):
        all_records = ContractClient(use_fixtures=True).fetch_all()
        platforms = {r["platform"] for r in all_records}
        assert platforms == {"kalshi", "polymarket"}
        for r in all_records:
            assert 0 <= r["price"] <= 1
            assert r["volume"] >= 0


class TestPolymarketNormalizer:
    def test_earnings_market_maps_ticker_and_mid_price(self):
        from src.ingestion.contract_client import _polymarket_to_common

        rec = _polymarket_to_common(
            {
                "id": "4440714",
                "question": "Will FedEx (FDX) beat quarterly earnings?",
                "outcomePrices": '["0.35", "0.65"]',
                "bestBid": 0.28,
                "bestAsk": 0.42,
                "volumeNum": 5736.4,
                "endDate": "2026-10-28T20:00:00Z",
            }
        )
        assert rec["ticker"] == "FDX"
        assert rec["event_type"] == "earnings_beat"
        assert rec["event_date"] == "2026-10-28"
        assert abs(rec["price"] - 0.35) < 1e-9
        assert rec["volume"] == 5736

    def test_falls_back_to_outcome_prices_without_book(self):
        from src.ingestion.contract_client import _polymarket_to_common

        rec = _polymarket_to_common(
            {"question": "Will Truist (TFC) beat quarterly earnings?", "outcomePrices": '["0.4", "0.6"]'}
        )
        assert rec["ticker"] == "TFC"
        assert abs(rec["price"] - 0.4) < 1e-9


class TestDedupeEarningsEvents:
    def test_drops_10q_filed_right_after_8k(self):
        from src.ingestion.edgar_client import dedupe_earnings_events

        kept = dedupe_earnings_events(
            [
                {"ticker": "AAPL", "form": "8-K", "filing_date": "2026-07-30"},
                {"ticker": "AAPL", "form": "10-Q", "filing_date": "2026-07-31"},
                {"ticker": "MSFT", "form": "10-Q", "filing_date": "2026-07-31"},
            ]
        )
        assert [(t["ticker"], t["form"]) for t in kept] == [("AAPL", "8-K"), ("MSFT", "10-Q")]

    def test_keeps_10q_outside_window(self):
        from src.ingestion.edgar_client import dedupe_earnings_events

        kept = dedupe_earnings_events(
            [
                {"ticker": "AAPL", "form": "8-K", "filing_date": "2026-01-29"},
                {"ticker": "AAPL", "form": "10-Q", "filing_date": "2026-05-01"},
            ]
        )
        assert len(kept) == 2
