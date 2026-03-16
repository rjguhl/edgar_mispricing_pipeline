"""Macro feature engineering from FRED indicators."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class MacroFeatureBuilder:
    """Transforms raw FRED indicators into model-ready features."""

    # Indicators and their expected frequency
    INDICATORS = {
        "CPIAUCSL": "monthly",
        "UNRATE": "monthly",
        "T10Y2Y": "daily",
        "UMCSENT": "monthly",
        "FEDFUNDS": "daily",
        "INDPRO": "monthly",
    }

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build engineered features from raw FRED time series.

        Input df columns: indicator_id, observation_date, value
        Output: pivoted dataframe with one row per date, feature columns

        Features per indicator:
          - level (raw value)
          - change (period-over-period delta)
          - pct_change (period-over-period % change)
          - rolling_avg_3m (3-month rolling mean)
          - surprise (deviation from rolling average, proxy for consensus miss)
        """
        df = df.copy()
        df["observation_date"] = pd.to_datetime(df["observation_date"])
        df = df.sort_values(["indicator_id", "observation_date"])

        feature_frames = []

        for indicator_id in self.INDICATORS:
            subset = df[df["indicator_id"] == indicator_id].copy()
            if subset.empty:
                logger.warning(f"No data for {indicator_id}")
                continue

            prefix = indicator_id.lower()

            subset[f"{prefix}_level"] = subset["value"]
            subset[f"{prefix}_change"] = subset["value"].diff()
            subset[f"{prefix}_pct_change"] = subset["value"].pct_change()
            subset[f"{prefix}_rolling_avg_3m"] = (
                subset["value"].rolling(window=3, min_periods=1).mean()
            )
            subset[f"{prefix}_surprise"] = (
                subset["value"] - subset[f"{prefix}_rolling_avg_3m"]
            )

            feature_cols = [
                "observation_date",
                f"{prefix}_level",
                f"{prefix}_change",
                f"{prefix}_pct_change",
                f"{prefix}_rolling_avg_3m",
                f"{prefix}_surprise",
            ]
            feature_frames.append(subset[feature_cols].set_index("observation_date"))

        if not feature_frames:
            return pd.DataFrame()

        result = pd.concat(feature_frames, axis=1)
        result = result.sort_index().ffill()

        logger.info(
            f"Built {len(result.columns)} macro features across "
            f"{len(result)} dates"
        )
        return result
