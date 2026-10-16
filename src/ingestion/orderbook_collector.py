"""Hourly order-book snapshots of open Polymarket earnings contracts.

The backtest prices contracts at their last trade, which can sit far from
anything executable on these thin markets (e.g. last trade 0.99 with the
best bid at 0.60 and no asks). These snapshots record what a trader could
actually have bought or sold at — best bid/ask, size, and nearby depth —
for re-testing the backtest and paper-trading at real prices.

Run (appends to one JSONL file per UTC day):
    python -m src.ingestion.orderbook_collector --out data/orderbook

Scheduled hourly by .github/workflows/orderbook.yml, which commits the files
to the `orderbook-data` branch.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.ingestion.contract_client import (
    ContractClient,
    _eps_terms_from_slug,
    _release_date_from_slug,
    _ticker_from_question,
)

logger = logging.getLogger(__name__)

CLOB_BOOK_URL = "https://clob.polymarket.com/book"
REQUEST_INTERVAL_SECONDS = 0.2
DEPTH_WINDOW = 0.05  # size resting within 5 cents of the best price


def summarize_book(book: dict) -> dict:
    """Top-of-book and near-touch depth from a CLOB /book response."""
    bids = sorted(((float(b["price"]), float(b["size"])) for b in book.get("bids", [])), reverse=True)
    asks = sorted((float(a["price"]), float(a["size"])) for a in book.get("asks", []))
    best_bid, bid_size = bids[0] if bids else (None, 0.0)
    best_ask, ask_size = asks[0] if asks else (None, 0.0)
    last = book.get("last_trade_price")
    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "spread": round(best_ask - best_bid, 4) if bids and asks else None,
        "mid": round((best_ask + best_bid) / 2, 4) if bids and asks else None,
        "bid_depth_5c": sum(s for p, s in bids if p >= best_bid - DEPTH_WINDOW) if bids else 0.0,
        "ask_depth_5c": sum(s for p, s in asks if p <= best_ask + DEPTH_WINDOW) if asks else 0.0,
        "bid_levels": len(bids),
        "ask_levels": len(asks),
        # Recorded but unreliable: on many books it reports the NO side's
        # trade (e.g. 0.04 with the YES book at 0.92/0.93). Use bid/ask.
        "last_trade_price": float(last) if last not in (None, "") else None,
    }


def collect() -> list[dict]:
    """One snapshot row per open earnings market."""
    snapshot_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    session = requests.Session()
    rows = []
    for m in ContractClient(use_fixtures=False).fetch_polymarket_earnings_markets(closed=False):
        try:
            yes_token = json.loads(m.get("clobTokenIds") or "[]")[0]
        except (ValueError, IndexError):
            continue
        time.sleep(REQUEST_INTERVAL_SECONDS)
        try:
            resp = session.get(CLOB_BOOK_URL, params={"token_id": yes_token}, timeout=30)
            resp.raise_for_status()
            book = resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning("Order book unavailable for market %s: %s", m.get("id"), e)
            continue
        basis, threshold = _eps_terms_from_slug(m.get("slug", ""))
        rows.append(
            {
                "snapshot_ts": snapshot_ts,
                "contract_id": str(m.get("id")),
                "ticker": _ticker_from_question(m.get("question", "")),
                "release_date": _release_date_from_slug(m.get("slug", "")),
                "eps_basis": basis,
                "eps_threshold": threshold,
                "slug": m.get("slug", ""),
                "volume": float(m.get("volumeNum") or 0.0),
                **summarize_book(book),
            }
        )
    logger.info("Captured %d order-book snapshots", len(rows))
    return rows


def append_jsonl(rows: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/orderbook"))
    args = parser.parse_args()
    path = append_jsonl(collect(), args.out)
    logger.info("Appended to %s", path)
