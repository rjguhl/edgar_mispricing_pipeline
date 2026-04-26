"""Central configuration loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # API Keys
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
    KALSHI_API_KEY: str = os.getenv("KALSHI_API_KEY", "")
    POLYMARKET_API_KEY: str = os.getenv("POLYMARKET_API_KEY", "")

    # AWS
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_DEFAULT_REGION: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "edgar-mispricing-pipeline")

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "postgresql://user:password@localhost:5432/mispricing"
    )

    # Pipeline
    MISPRICING_THRESHOLD: float = float(os.getenv("MISPRICING_THRESHOLD", "0.10"))
    LOOKBACK_DAYS: int = int(os.getenv("LOOKBACK_DAYS", "90"))

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
