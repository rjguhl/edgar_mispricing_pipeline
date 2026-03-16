"""FRED macroeconomic indicator ingestion client."""

import logging
from datetime import datetime, timedelta

import boto3
from fredapi import Fred

from src.config import Config

logger = logging.getLogger(__name__)

# Key macro indicators for earnings surprise prediction
INDICATOR_IDS = {
    "CPIAUCSL": "Consumer Price Index (All Urban Consumers)",
    "UNRATE": "Unemployment Rate",
    "T10Y2Y": "10-Year / 2-Year Treasury Yield Spread",
    "UMCSENT": "University of Michigan Consumer Sentiment",
    "FEDFUNDS": "Federal Funds Effective Rate",
    "INDPRO": "Industrial Production Index",
}


class FredClient:
    """Fetches macroeconomic indicators from the FRED API."""

    def __init__(self):
        self.fred = Fred(api_key=Config.FRED_API_KEY)
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
            region_name=Config.AWS_DEFAULT_REGION,
        )

    def fetch_indicators(self, lookback_days: int = 90) -> list[dict]:
        """
        Pull latest values for all tracked macro indicators.

        Args:
            lookback_days: Number of days of history to fetch

        Returns:
            List of dicts with keys: indicator_id, indicator_name,
            observation_date, value, units
        """
        cutoff = datetime.now() - timedelta(days=lookback_days)
        results = []

        for indicator_id, name in INDICATOR_IDS.items():
            try:
                series = self.fred.get_series(
                    indicator_id,
                    observation_start=cutoff.strftime("%Y-%m-%d"),
                )
                for date, value in series.items():
                    if value is not None:
                        record = {
                            "indicator_id": indicator_id,
                            "indicator_name": name,
                            "observation_date": date.strftime("%Y-%m-%d"),
                            "value": float(value),
                            "units": self._get_units(indicator_id),
                        }
                        results.append(record)

                logger.info(
                    f"Fetched {len(series)} observations for {indicator_id}"
                )
            except Exception as e:
                logger.error(f"Failed to fetch {indicator_id}: {e}")

        self._upload_to_s3(results)
        return results

    def _get_units(self, indicator_id: str) -> str:
        """Return the unit label for a given indicator."""
        units_map = {
            "CPIAUCSL": "index",
            "UNRATE": "percent",
            "T10Y2Y": "percent",
            "UMCSENT": "index",
            "FEDFUNDS": "percent",
            "INDPRO": "index",
        }
        return units_map.get(indicator_id, "unknown")

    def _upload_to_s3(self, records: list[dict]):
        """Persist raw FRED data to S3."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        key = f"raw/fred/{date_str}/indicators.json"
        self.s3.put_object(
            Bucket=Config.S3_BUCKET,
            Key=key,
            Body=str(records),
            ContentType="application/json",
        )
