"""SEC EDGAR earnings call transcript ingestion client."""

import logging
from datetime import datetime, timedelta

import boto3
import requests

from src.config import Config

logger = logging.getLogger(__name__)

EDGAR_BASE_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_FILINGS_URL = "https://www.sec.gov/cgi-bin/browse-edgar"


class EdgarClient:
    """Fetches earnings call transcripts from SEC EDGAR."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": Config.EDGAR_USER_AGENT})
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
            region_name=Config.AWS_DEFAULT_REGION,
        )

    def fetch_transcripts(self, tickers: list[str], lookback_days: int = 90):
        """
        Fetch earnings call transcripts for given tickers
        within the lookback window.

        Args:
            tickers: List of stock ticker symbols (e.g., ['AAPL', 'MSFT'])
            lookback_days: Number of days to look back for filings

        Returns:
            List of dicts with keys: ticker, filing_date, transcript_text, filing_url
        """
        cutoff = datetime.now() - timedelta(days=lookback_days)
        results = []

        for ticker in tickers:
            try:
                filings = self._search_filings(ticker, cutoff)
                for filing in filings:
                    transcript = self._download_transcript(filing)
                    if transcript:
                        results.append(transcript)
                        self._upload_to_s3(transcript)
            except Exception as e:
                logger.error(f"Failed to fetch transcripts for {ticker}: {e}")

        logger.info(f"Fetched {len(results)} transcripts for {len(tickers)} tickers")
        return results

    def _search_filings(self, ticker: str, cutoff: datetime) -> list[dict]:
        """Search EDGAR for 8-K and 10-Q filings after cutoff date."""
        # TODO: Implement EDGAR full-text search API call
        # Filter for earnings-related filings (8-K with Item 2.02)
        raise NotImplementedError

    def _download_transcript(self, filing: dict) -> dict | None:
        """Download and parse the transcript text from a filing."""
        # TODO: Fetch filing document, extract transcript text
        raise NotImplementedError

    def _upload_to_s3(self, transcript: dict):
        """Persist raw transcript to S3 with partitioned key."""
        key = (
            f"raw/edgar/{transcript['ticker']}/"
            f"{transcript['filing_date']}.json"
        )
        self.s3.put_object(
            Bucket=Config.S3_BUCKET,
            Key=key,
            Body=str(transcript),
            ContentType="application/json",
        )
