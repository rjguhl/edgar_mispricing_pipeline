"""Central configuration loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    # API Keys
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")

    # Pipeline
    MISPRICING_THRESHOLD: float = float(os.getenv("MISPRICING_THRESHOLD", "0.10"))

    # EDGAR
    EDGAR_USER_AGENT: str = os.getenv(
        "EDGAR_USER_AGENT", "EDGAR Mispricing Pipeline rjguhl@example.com"
    )
    SP500_TICKERS_PATH: str = os.getenv("SP500_TICKERS_PATH", "data/sp500_tickers.csv")

    # Dev mode: when true, ingestion clients read from local fixtures
    # instead of hitting live APIs. Lets the full pipeline run without keys.
    USE_FIXTURES: bool = os.getenv("USE_FIXTURES", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    FIXTURES_DIR: str = os.getenv("FIXTURES_DIR", "data/fixtures")
