"""Prediction-market contract prices (Kalshi + Polymarket).

Both platforms' market data endpoints are public — no API keys needed.

- Polymarket lists per-company "Will X (TICK) beat quarterly earnings?"
  markets under the `earnings` tag on its Gamma API. These are what the
  beat model is compared against.
- Kalshi has no EPS beat/miss markets; its closest single-stock contracts
  are "Company KPI" markets (headcount, deliveries, ...). They're collected
  for reference but excluded from signals, since they ask a different
  question than the model answers.

Every contract is returned in one common schema:
    contract_id, platform, ticker, event_type, event_date, price, volume, snapshot_ts
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import requests

from src.config import Config

logger = logging.getLogger(__name__)

KALSHI_MARKETS_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"
POLYMARKET_EVENTS_URL = "https://gamma-api.polymarket.com/events"
# Per-company "Will X (TICK) beat quarterly earnings?" markets are tagged
# `earnings` on Gamma (verified live 2026-10-15).
POLYMARKET_EARNINGS_TAG = "earnings"
POLYMARKET_PAGE_SIZE = 100
POLYMARKET_MAX_PAGES = 50  # 100 events/page; ~1.4k resolved earnings events today

# Kalshi's public markets endpoint rate-limits aggressively; pace requests
# and retry with backoff on 429 rather than hammering it.
KALSHI_REQUEST_INTERVAL_SECONDS = 0.5
KALSHI_MAX_RETRIES = 3

# Kalshi "Company KPI" series per stock ticker (verified live 2026-10-09).
# Series tickers aren't derivable from the stock ticker by a fixed rule
# (KXGOOGA covers GOOGL; KXAAPLA's trailing "A" isn't part of the ticker).
KNOWN_KPI_SERIES: dict[str, list[str]] = {
    "AAPL": ["KXAAPLA"],
    "GOOGL": ["KXGOOGA"],
    "NVDA": ["KXNVDAA"],
    "TSLA": ["KXTSLA"],
}


class ContractClient:
    """Fetches contract prices from Kalshi and Polymarket."""

    def __init__(self, use_fixtures: bool | None = None):
        self.use_fixtures = Config.USE_FIXTURES if use_fixtures is None else use_fixtures
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": Config.EDGAR_USER_AGENT})

    def fetch_all(self, tickers: list[str] | None = None) -> list[dict]:
        """Contract snapshots from both platforms in the common schema."""
        results: list[dict] = []
        for name, fetch in (
            ("Kalshi", self.fetch_kalshi_contracts),
            ("Polymarket", self.fetch_polymarket_contracts),
        ):
            try:
                results.extend(fetch(tickers))
            except Exception as e:
                logger.error("%s fetch failed: %s", name, e)
        logger.info("Fetched %d contract snapshots", len(results))
        return results

    # ------------------------------------------------------------------
    # Kalshi
    # ------------------------------------------------------------------

    def fetch_kalshi_contracts(self, tickers: list[str] | None = None) -> list[dict]:
        """Open Company-KPI contracts for `tickers` (default: every ticker
        in KNOWN_KPI_SERIES)."""
        if self.use_fixtures:
            return self._fixture_subset("kalshi")

        wanted = [t.upper() for t in tickers] if tickers else list(KNOWN_KPI_SERIES)
        contracts: list[dict] = []
        for ticker in wanted:
            for series_ticker in KNOWN_KPI_SERIES.get(ticker, []):
                time.sleep(KALSHI_REQUEST_INTERVAL_SECONDS)
                payload = self._get_kalshi_page(
                    {"series_ticker": series_ticker, "status": "open", "limit": 200}
                )
                contracts.extend(_kalshi_to_common(m, ticker) for m in payload.get("markets", []))
        logger.info("Fetched %d Kalshi markets", len(contracts))
        return contracts

    def _get_kalshi_page(self, params: dict) -> dict:
        """GET one page of Kalshi markets, retrying with backoff on 429."""
        delay = KALSHI_REQUEST_INTERVAL_SECONDS
        for attempt in range(KALSHI_MAX_RETRIES):
            resp = self.session.get(KALSHI_MARKETS_URL, params=params, timeout=30)
            if resp.status_code == 429 and attempt < KALSHI_MAX_RETRIES - 1:
                logger.warning(
                    "Kalshi rate-limited (attempt %d/%d); backing off %.1fs",
                    attempt + 1, KALSHI_MAX_RETRIES, delay,
                )
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp.json()
        return {}

    # ------------------------------------------------------------------
    # Polymarket
    # ------------------------------------------------------------------

    def fetch_polymarket_contracts(self, tickers: list[str] | None = None) -> list[dict]:
        """Open "Will X beat quarterly earnings?" markets, optionally limited
        to `tickers`."""
        if self.use_fixtures:
            return self._fixture_subset("polymarket")

        records = [_polymarket_to_common(m) for m in self.fetch_polymarket_earnings_markets(closed=False)]
        if tickers:
            wanted = {t.upper() for t in tickers}
            records = [r for r in records if r["ticker"] in wanted]
        logger.info("Fetched %d Polymarket earnings markets", len(records))
        return records

    def fetch_polymarket_earnings_markets(self, closed: bool) -> list[dict]:
        """Raw Gamma market dicts for every earnings market, open
        (closed=False) or resolved (closed=True)."""
        markets: list[dict] = []
        for page in range(POLYMARKET_MAX_PAGES):
            params = {
                "tag_slug": POLYMARKET_EARNINGS_TAG,
                "closed": "true" if closed else "false",
                "limit": POLYMARKET_PAGE_SIZE,
                "offset": page * POLYMARKET_PAGE_SIZE,
            }
            resp = self.session.get(POLYMARKET_EVENTS_URL, params=params, timeout=30)
            resp.raise_for_status()
            events = resp.json() or []
            for event in events:
                markets.extend(
                    m for m in event.get("markets") or []
                    if "earnings" in (m.get("question") or "").lower()
                )
            if len(events) < POLYMARKET_PAGE_SIZE:
                break
        return markets

    def fetch_polymarket_earnings_tickers(self) -> list[str]:
        """Every ticker Polymarket has listed an earnings market for (open or
        resolved) — the beat model's training universe, so it learns from
        the same kind of companies it's scored against."""
        tickers = {
            _ticker_from_question(m.get("question", ""))
            for closed in (False, True)
            for m in self.fetch_polymarket_earnings_markets(closed)
        }
        tickers.discard("")
        logger.info("Found %d tickers with Polymarket earnings markets", len(tickers))
        return sorted(tickers)

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    def _fixture_subset(self, platform: str) -> list[dict]:
        path = Path(Config.FIXTURES_DIR) / "contract_prices.json"
        if not path.exists():
            logger.warning("Fixture file %s not found", path)
            return []
        subset = [r for r in json.loads(path.read_text()) if r.get("platform") == platform]
        logger.info("Loaded %d fixture %s contracts (USE_FIXTURES=true)", len(subset), platform)
        return subset


# ----------------------------------------------------------------------
# Platform → common schema
# ----------------------------------------------------------------------


def _kalshi_to_common(m: dict, ticker: str) -> dict:
    """Map a Kalshi market dict → common schema.

    Kalshi returns prices as decimal-dollar strings (`yes_bid_dollars`,
    `yes_ask_dollars`, `last_price_dollars`) and volume as `volume_fp`; the
    integer-cent `yes_bid`/`yes_ask`/`volume` fields of an older API version
    now read as 0 for every market.
    """
    yes_bid = _to_float(m.get("yes_bid_dollars"))
    yes_ask = _to_float(m.get("yes_ask_dollars"))
    last_price = _to_float(m.get("last_price_dollars"))
    if yes_bid is not None and yes_ask is not None:
        price = (yes_bid + yes_ask) / 2
    elif last_price is not None:
        price = last_price
    else:
        price = 0.0

    return {
        "contract_id": m.get("ticker", ""),
        "platform": "kalshi",
        "ticker": ticker,
        "event_type": "company_kpi",
        "event_date": (m.get("expected_expiration_time") or "")[:10],
        "price": price,
        "volume": int(_to_float(m.get("volume_fp")) or 0),
        "snapshot_ts": datetime.utcnow().isoformat() + "Z",
    }


def _polymarket_to_common(m: dict) -> dict:
    """Map a Polymarket Gamma market dict → common schema."""
    # Prefer the live order-book mid; fall back to Gamma's `outcomePrices`,
    # a JSON-encoded list like '["0.62", "0.38"]' (YES first).
    best_bid = _to_float(m.get("bestBid"))
    best_ask = _to_float(m.get("bestAsk"))
    price = 0.0
    if best_bid is not None and best_ask is not None and best_ask > 0:
        price = (best_bid + best_ask) / 2
    else:
        try:
            raw = m.get("outcomePrices")
            parsed = json.loads(raw) if isinstance(raw, str) else (raw or [])
            if parsed:
                price = float(parsed[0])
        except (ValueError, TypeError):
            price = 0.0

    question = m.get("question", "") or ""
    return {
        "contract_id": str(m.get("id") or m.get("conditionId", "")),
        "platform": "polymarket",
        "ticker": _ticker_from_question(question),
        "event_type": "earnings_beat" if "earnings" in question.lower() else "",
        "event_date": _release_date_from_slug(m.get("slug", "")) or (m.get("endDate") or "")[:10],
        "price": price,
        "volume": int(float(m.get("volumeNum", 0) or 0)),
        "snapshot_ts": datetime.utcnow().isoformat() + "Z",
    }


def _to_float(value) -> float | None:
    """Parse a numeric(-string) field, treating "" and None as missing."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Earnings market slugs encode the EPS basis, expected release date and
# consensus threshold, e.g. "cfg-quarterly-earnings-nongaap-eps-10-16-2026-1pt40"
# or, for a loss, "aal-quarterly-earnings-nongaap-eps-10-22-2026-neg0pt34".
# The market's endDate is only a resolution deadline (often weeks later), so
# the release date is what we time against.
_SLUG_RE = re.compile(
    r"-(?P<basis>gaap|nongaap)-eps-(?P<month>\d{2})-(?P<day>\d{2})-(?P<year>\d{4})"
    r"-(?P<neg>neg)?(?P<whole>\d+)pt(?P<frac>\d+)$"
)
_PAREN_TICKER_RE = re.compile(r"\(([A-Z]{1,5}(?:\.[A-Z])?)\)")
_TICKER_TOKEN_RE = re.compile(r"\b([A-Z]{1,5})\b")


def _release_date_from_slug(slug: str) -> str:
    match = _SLUG_RE.search(slug or "")
    return f"{match['year']}-{match['month']}-{match['day']}" if match else ""


def _eps_terms_from_slug(slug: str) -> tuple[str, float | None]:
    """(basis, consensus EPS threshold) — e.g. ("nongaap", -0.34)."""
    match = _SLUG_RE.search(slug or "")
    if not match:
        return "", None
    threshold = float(f"{match['whole']}.{match['frac']}")
    return match["basis"], -threshold if match["neg"] else threshold


def _ticker_from_question(question: str) -> str:
    """Ticker from a Polymarket question. Earnings questions carry it in
    parentheses ("Will FedEx (FDX) beat ...?"); the bare all-caps token is a
    fallback that can misfire on other capitalized words."""
    if not question:
        return ""
    paren = _PAREN_TICKER_RE.search(question)
    if paren:
        return paren.group(1)
    match = _TICKER_TOKEN_RE.search(question)
    return match.group(1) if match else ""
