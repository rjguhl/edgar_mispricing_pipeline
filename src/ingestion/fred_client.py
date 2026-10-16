"""FRED macroeconomic indicator ingestion client."""

import logging
from datetime import datetime, timedelta

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
        # and only if AWS is actually configured (mirrors EdgarClient/ContractClient).

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
