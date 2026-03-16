"""Tests for the macro feature engineering module."""

import pandas as pd
import pytest

from src.processing.macro_features import MacroFeatureBuilder


@pytest.fixture
def sample_fred_data():
    """Minimal FRED data for testing."""
    return pd.DataFrame(
        {
            "indicator_id": ["CPIAUCSL"] * 4 + ["UNRATE"] * 4,
            "observation_date": [
                "2025-01-01",
                "2025-02-01",
                "2025-03-01",
                "2025-04-01",
            ]
            * 2,
            "value": [308.0, 309.5, 310.2, 311.8, 4.1, 4.0, 3.9, 4.0],
        }
    )


class TestMacroFeatureBuilder:
    def test_build_features_returns_dataframe(self, sample_fred_data):
        builder = MacroFeatureBuilder()
        result = builder.build_features(sample_fred_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_build_features_has_expected_columns(self, sample_fred_data):
        builder = MacroFeatureBuilder()
        result = builder.build_features(sample_fred_data)
        assert "cpiaucsl_level" in result.columns
        assert "cpiaucsl_change" in result.columns
        assert "cpiaucsl_surprise" in result.columns
        assert "unrate_level" in result.columns

    def test_build_features_empty_input(self):
        builder = MacroFeatureBuilder()
        empty = pd.DataFrame(columns=["indicator_id", "observation_date", "value"])
        result = builder.build_features(empty)
        assert result.empty

    def test_change_calculation(self, sample_fred_data):
        builder = MacroFeatureBuilder()
        result = builder.build_features(sample_fred_data)
        # Second CPI change should be 309.5 - 308.0 = 1.5
        cpi_changes = result["cpiaucsl_change"].dropna().values
        assert abs(cpi_changes[0] - 1.5) < 0.01
