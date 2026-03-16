"""Prediction market contract price ingestion client (Kalshi + Polymarket)."""

import logging
from datetime import datetime

import boto3
import requests

from src.config import Config

logger = logging.getLogger(__name__)


class ContractClient:
    """Fetches contract prices from Kalshi and Polymarket APIs."""

    def __init__(self):
        self.session = requests.Session()
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
            region_name=Config.AWS_DEFAULT_REGION,
        )

    def fetch_kalshi_contracts(self, event_types: list[str] = None) -> list[dict]:
        """
        Fetch active contract prices from Kalshi.

        Args:
            event_types: Optional filter for contract categories
                         (e.g., ['earnings', 'economic'])

        Returns:
            List of contract price snapshots
        """
        # TODO: Implement Kalshi API v2 integration
        # Docs: https://trading-api.readme.io/reference
        raise NotImplementedError

    def fetch_polymarket_contracts(
        self, event_types: list[str] = None
    ) -> list[dict]:
        """
        Fetch active contract prices from Polymarket.

        Args:
            event_types: Optional filter for contract categories

        Returns:
            List of contract price snapshots
        """
        # TODO: Implement Polymarket CLOB API integration
        # Docs: https://docs.polymarket.com
        raise NotImplementedError

    def fetch_all(self, event_types: list[str] = None) -> list[dict]:
        """Fetch from all platforms, normalize, and persist."""
        results = []

        try:
            kalshi = self.fetch_kalshi_contracts(event_types)
            results.extend(self._normalize(kalshi, platform="kalshi"))
        except Exception as e:
            logger.error(f"Kalshi fetch failed: {e}")

        try:
            poly = self.fetch_polymarket_contracts(event_types)
            results.extend(self._normalize(poly, platform="polymarket"))
        except Exception as e:
            logger.error(f"Polymarket fetch failed: {e}")

        self._upload_to_s3(results)
        logger.info(f"Fetched {len(results)} contract snapshots")
        return results

    def _normalize(self, contracts: list[dict], platform: str) -> list[dict]:
        """Normalize contract data to common schema."""
        normalized = []
        snapshot_ts = datetime.utcnow().isoformat()

        for c in contracts:
            normalized.append(
                {
                    "contract_id": c.get("id", ""),
                    "platform": platform,
                    "ticker": c.get("ticker", ""),
                    "event_type": c.get("event_type", ""),
                    "event_date": c.get("event_date", ""),
                    "price": c.get("price", 0.0),
                    "volume": c.get("volume", 0),
                    "snapshot_ts": snapshot_ts,
                }
            )
        return normalized

    def _upload_to_s3(self, records: list[dict]):
        """Persist raw contract data to S3."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        key = f"raw/contracts/{date_str}/prices.json"
        self.s3.put_object(
            Bucket=Config.S3_BUCKET,
            Key=key,
            Body=str(records),
            ContentType="application/json",
        )
