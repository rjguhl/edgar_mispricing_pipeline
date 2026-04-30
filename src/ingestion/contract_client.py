"""Prediction-market contract price ingestion (Kalshi + Polymarket).

Design notes
------------
- Kalshi public market metadata (`/trade-api/v2/markets`) does not require
  authentication. A logged-in API key is only needed for trading or for
  reading account-private data. We use the public endpoint here.
- Polymarket exposes two relevant APIs:
    * Gamma (`gamma-api.polymarket.com`) for market metadata + outcome prices
    * CLOB (`clob.polymarket.com`) for live order-book / trade prices
  Gamma is sufficient for our snapshot use case and avoids managing an
  Ethereum wallet for the CLOB API.
- We keep the original normalized schema from the scaffold so dbt's
  `stg_contract_prices` model and `assert_valid_market_probability` test
  continue to work unchanged.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import requests
import re

from src.config import Config

logger = logging.getLogger(__name__)

KALSHI_MARKETS_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com/markets"


class ContractClient:
    """Fetches contract prices from Kalshi and Polymarket APIs."""

    def __init__(self, use_fixtures: bool | None = None):
        self.use_fixtures = (
            Config.USE_FIXTURES if use_fixtures is None else use_fixtures
        )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": Config.EDGAR_USER_AGENT})
        self._s3 = None

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def fetch_kalshi_contracts(
        self, event_types: list[str] | None = None
    ) -> list[dict]:
        """Fetch active contract prices from Kalshi.

        `event_types` is a list of substrings to match against the Kalshi
        market `event_ticker` (e.g., "EARNINGS", "CPI"). None = no filter.
        """
        if self.use_fixtures:
            return self._fixture_subset("kalshi")

        params = {"status": "open", "limit": 200}
        all_markets: list[dict] = []
        cursor: str | None = None

        # Kalshi paginates with a cursor; cap at a sensible number of pages.
        for _ in range(20):
            if cursor:
                params["cursor"] = cursor
            resp = self.session.get(KALSHI_MARKETS_URL, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            all_markets.extend(payload.get("markets", []))
            cursor = payload.get("cursor")
            if not cursor:
                break

        if event_types:
            wanted = [t.upper() for t in event_types]
            all_markets = [
                m
                for m in all_markets
                if any(w in (m.get("event_ticker") or "").upper() for w in wanted)
            ]

        logger.info("Fetched %d Kalshi markets", len(all_markets))
        return [_kalshi_to_common(m) for m in all_markets]

    def fetch_polymarket_contracts(
        self, event_types: list[str] | None = None
    ) -> list[dict]:
        """Fetch active contract prices from Polymarket via the Gamma API."""
        if self.use_fixtures:
            return self._fixture_subset("polymarket")

        params = {"active": "true", "closed": "false", "limit": 200}
        resp = self.session.get(POLYMARKET_GAMMA_URL, params=params, timeout=30)
        resp.raise_for_status()
        markets = resp.json() or []

        if event_types:
            wanted = [t.lower() for t in event_types]
            markets = [
                m
                for m in markets
                if any(
                    w in (m.get("category", "") or "").lower()
                    or w in (m.get("question", "") or "").lower()
                    for w in wanted
                )
            ]

        logger.info("Fetched %d Polymarket markets", len(markets))
        return [_polymarket_to_common(m) for m in markets if m]

    def fetch_all(self, event_types: list[str] | None = None) -> list[dict]:
        """Fetch from all platforms, normalize, and persist."""
        results: list[dict] = []

        try:
            kalshi = self.fetch_kalshi_contracts(event_types)
            results.extend(self._normalize(kalshi, platform="kalshi"))
        except Exception as e:
            logger.error("Kalshi fetch failed: %s", e)

        try:
            poly = self.fetch_polymarket_contracts(event_types)
            results.extend(self._normalize(poly, platform="polymarket"))
        except Exception as e:
            logger.error("Polymarket fetch failed: %s", e)

        self._upload_to_s3(results)
        logger.info("Fetched %d contract snapshots", len(results))
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize(self, contracts: list[dict], platform: str) -> list[dict]:
        """Coerce platform-native dicts into the common feature-store schema."""
        snapshot_ts = datetime.utcnow().isoformat() + "Z"
        normalized: list[dict] = []
        for c in contracts:
            normalized.append(
                {
                    "contract_id": c.get("contract_id") or c.get("id", ""),
                    "platform": c.get("platform", platform),
                    "ticker": c.get("ticker", ""),
                    "event_type": c.get("event_type", ""),
                    "event_date": c.get("event_date", ""),
                    "price": float(c.get("price", 0.0) or 0.0),
                    "volume": int(c.get("volume", 0) or 0),
                    "snapshot_ts": c.get("snapshot_ts") or snapshot_ts,
                }
            )
        return normalized

    def _fixture_subset(self, platform: str) -> list[dict]:
        path = Path(Config.FIXTURES_DIR) / "contract_prices.json"
        if not path.exists():
            logger.warning("Fixture file %s not found", path)
            return []
        records = json.loads(path.read_text())
        subset = [r for r in records if r.get("platform") == platform]
        logger.info(
            "Loaded %d fixture %s contracts (USE_FIXTURES=true)",
            len(subset),
            platform,
        )
        return subset

    def _upload_to_s3(self, records: list[dict]) -> None:
        """Persist raw contract data to S3. No-op if AWS not configured."""
        if not records or not Config.S3_BUCKET or not Config.AWS_ACCESS_KEY_ID:
            return
        if self._s3 is None:
            import boto3

            self._s3 = boto3.client(
                "s3",
                aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
                region_name=Config.AWS_DEFAULT_REGION,
            )

        date_str = datetime.now().strftime("%Y-%m-%d")
        key = f"raw/contracts/{date_str}/prices.json"
        self._s3.put_object(
            Bucket=Config.S3_BUCKET,
            Key=key,
            Body=json.dumps(records).encode("utf-8"),
            ContentType="application/json",
        )


# ----------------------------------------------------------------------
# Platform-specific normalizers
# ----------------------------------------------------------------------


def _kalshi_to_common(m: dict) -> dict:
    """Map a Kalshi market dict → common-schema fields.

    Kalshi prices come as integer cents (0-100); we convert to a 0-1 probability.
    """
    yes_bid_cents = m.get("yes_bid")
    yes_ask_cents = m.get("yes_ask")
    last_price_cents = m.get("last_price")

    # Use mid of bid/ask if both present, else last_price, else 0.
    if yes_bid_cents is not None and yes_ask_cents is not None:
        price = (yes_bid_cents + yes_ask_cents) / 2 / 100
    elif last_price_cents is not None:
        price = last_price_cents / 100
    else:
        price = 0.0

    return {
        "contract_id": m.get("ticker", ""),
        "platform": "kalshi",
        "ticker": _ticker_from_event(m.get("event_ticker", "")),
        "event_type": _event_type_from_kalshi(m),
        "event_date": (m.get("expected_expiration_time") or "")[:10],
        "price": price,
        "volume": int(m.get("volume", 0) or 0),
        "snapshot_ts": datetime.utcnow().isoformat() + "Z",
    }


def _polymarket_to_common(m: dict) -> dict:
    """Map a Polymarket Gamma market dict → common-schema fields."""
    # Gamma's `outcomePrices` is a JSON-encoded list like '["0.62", "0.38"]'
    raw_prices = m.get("outcomePrices")
    price = 0.0
    try:
        if isinstance(raw_prices, str):
            parsed = json.loads(raw_prices)
        else:
            parsed = raw_prices or []
        if parsed:
            price = float(parsed[0])
    except (ValueError, TypeError):
        price = 0.0

    end_date = (m.get("endDate") or "")[:10]
    return {
        "contract_id": str(m.get("id") or m.get("conditionId", "")),
        "platform": "polymarket",
        "ticker": _ticker_from_question(m.get("question", "")),
        "event_type": (m.get("category") or "").lower(),
        "event_date": end_date,
        "price": price,
        "volume": int(float(m.get("volumeNum", 0) or 0)),
        "snapshot_ts": datetime.utcnow().isoformat() + "Z",
    }


# ----------------------------------------------------------------------
# Tiny inference helpers — pure functions, easy to unit test
# ----------------------------------------------------------------------

_TICKER_TOKEN_RE = re.compile(r"\b([A-Z]{1,5})\b")


def _ticker_from_event(event_ticker: str) -> str:
    """Best-effort ticker extraction from Kalshi event_ticker like 'EARNAAPL-26Q1'."""
    if not event_ticker:
        return ""
    match = _TICKER_TOKEN_RE.search(event_ticker.upper())
    return match.group(1) if match else ""


def _ticker_from_question(question: str) -> str:
    """Best-effort ticker extraction from a Polymarket question string."""
    if not question:
        return ""
    match = _TICKER_TOKEN_RE.search(question)
    return match.group(1) if match else ""


def _event_type_from_kalshi(m: dict) -> str:
    """Bucket Kalshi markets into our event_type taxonomy."""
    event_ticker = (m.get("event_ticker") or "").upper()
    if "EARN" in event_ticker:
        return "earnings_beat"
    if "CPI" in event_ticker:
        return "cpi"
    if "FED" in event_ticker:
        return "fed_decision"
    return (m.get("category") or "").lower()
