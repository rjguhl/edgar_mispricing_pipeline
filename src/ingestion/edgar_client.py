"""SEC EDGAR earnings-related filing ingestion client.

Design note
-----------
SEC EDGAR does not host verbatim earnings call transcripts (those live on
paid services like Seeking Alpha). The closest authoritative free source is
the 8-K filing with Item 2.02 ("Results of Operations and Financial
Condition"), which typically contains the earnings press release as an
exhibit. This client targets those 8-Ks and treats the exhibit text as the
"transcript" payload for downstream Claude extraction.

We use SEC's per-company submissions JSON endpoint rather than the
full-text search API because:
  - It returns all recent filings for a CIK in a single request.
  - The endpoint is explicitly documented and rate-friendly.
  - Full-text search has stricter QPS limits and is fuzzier on form filters.

SEC requires every request to carry a descriptive User-Agent including a
contact email, and to stay under 10 requests per second.
"""

from __future__ import annotations

import html as htmllib
import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

from src.config import Config

logger = logging.getLogger(__name__)

EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
# 10-Q/10-K filed within this many days after an earnings 8-K are treated
# as the same event (see dedupe_earnings_events).
EARNINGS_DEDUPE_WINDOW_DAYS = 14

MAX_RETRIES = 5
RETRY_BASE_DELAY_SECONDS = 2.0
RETRYABLE_STATUS_CODES = {403, 429, 500, 502, 503, 504}

EDGAR_FILING_INDEX_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dash}/index.json"
)
EDGAR_ARCHIVE_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dash}/{document}"
)

# SEC asks for <10 req/sec; we give a generous margin.
REQUEST_INTERVAL_SECONDS = 0.15


class EdgarClient:
    """Fetches earnings-related filings (8-K Item 2.02) from SEC EDGAR."""

    def __init__(self, use_fixtures: bool | None = None):
        # Honor explicit override; otherwise fall back to env-driven config.
        self.use_fixtures = (
            Config.USE_FIXTURES if use_fixtures is None else use_fixtures
        )
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": Config.EDGAR_USER_AGENT,
                "Accept-Encoding": "gzip, deflate",
            }
        )
        self._cik_cache: dict[str, int] | None = None


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_transcripts(
        self, tickers: list[str], lookback_days: int = 90
    ) -> list[dict]:
        """Fetch earnings-related filings for the given tickers.

        Returns a list of dicts:
            {ticker, filing_date, fiscal_quarter, fiscal_year,
             transcript_text, filing_url, loaded_at}
        """
        if self.use_fixtures:
            return self._fetch_from_fixtures(tickers, lookback_days)

        cutoff = datetime.now() - timedelta(days=lookback_days)
        results: list[dict] = []

        for ticker in tickers:
            try:
                filings = self._search_filings(ticker, cutoff)
                for filing in filings:
                    transcript = self._download_transcript(filing)
                    if transcript:
                        results.append(transcript)
            except Exception as e:
                logger.error("Failed to fetch transcripts for %s: %s", ticker, e)

        logger.info(
            "Fetched %d transcripts for %d tickers", len(results), len(tickers)
        )
        return results

    # ------------------------------------------------------------------
    # Live-API path
    # ------------------------------------------------------------------

    def _search_filings(self, ticker: str, cutoff: datetime) -> list[dict]:
        """Return 8-K (Item 2.02) and 10-Q filings for `ticker` after `cutoff`.

        Each filing dict carries the metadata needed to fetch the document.
        """
        cik = self._lookup_cik(ticker)
        if cik is None:
            logger.warning("No CIK found for ticker %s", ticker)
            return []

        url = EDGAR_SUBMISSIONS_URL.format(cik=cik)
        data = self._get_json(url)

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accession_nos = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])
        items = recent.get("items", [])

        filings: list[dict] = []
        for i, form in enumerate(forms):
            try:
                filing_date = datetime.strptime(filing_dates[i], "%Y-%m-%d")
            except (ValueError, IndexError):
                continue
            if filing_date < cutoff:
                continue

            # Only earnings-relevant forms.
            if form == "8-K":
                # Item 2.02 = "Results of Operations and Financial Condition"
                item_str = items[i] if i < len(items) else ""
                if "2.02" not in (item_str or ""):
                    continue
            elif form not in ("10-Q", "10-K"):
                continue

            accession_no = accession_nos[i]
            filings.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "form": form,
                    "filing_date": filing_dates[i],
                    "accession_no": accession_no,
                    "primary_document": primary_docs[i],
                    "filing_url": EDGAR_ARCHIVE_URL.format(
                        cik=cik,
                        accession_no_dash=accession_no.replace("-", ""),
                        document=primary_docs[i],
                    ),
                }
            )

        logger.info("Found %d earnings-relevant filings for %s", len(filings), ticker)
        return filings

    def _download_transcript(self, filing: dict) -> dict | None:
        """Download the filing document and strip HTML to plain text."""
        url = filing["filing_url"]
        try:
            if filing.get("form") == "8-K":
                text = self._download_8k_exhibits(filing)
            else:
                text = _html_to_text(self._get_text(url))
        except Exception as e:
            logger.error("Failed to download %s: %s", url, e)
            return None

        if len(text) < 500:
            # Skip stub filings — likely cover pages with no real content.
            logger.info("Skipping short filing for %s (%d chars)", filing["ticker"], len(text))
            return None

        fiscal_quarter, fiscal_year = _infer_fiscal_period(text, filing["filing_date"])

        return {
            "ticker": filing["ticker"],
            "form": filing.get("form", ""),
            "filing_date": filing["filing_date"],
            "fiscal_quarter": fiscal_quarter,
            "fiscal_year": fiscal_year,
            "transcript_text": text,
            "filing_url": url,
            "loaded_at": datetime.utcnow().isoformat() + "Z",
        }

    def _download_8k_exhibits(self, filing: dict) -> str:
        """Return the text of an earnings 8-K's exhibits.

        The 8-K primary document is just a cover page; the earnings press
        release (and sometimes CFO commentary) lives in separate exhibit
        files whose names vary by filer (msft-ex99_1.htm, q2fy27pr.htm,
        exhibit991.htm, ...). Take every .htm in the filing except the
        cover, index pages and XBRL viewer pages, 99.x exhibits first.
        Falls back to the primary document if no exhibits are found.
        """
        accession_no_dash = filing["accession_no"].replace("-", "")
        index = self._get_json(
            EDGAR_FILING_INDEX_URL.format(
                cik=filing["cik"], accession_no_dash=accession_no_dash
            )
        )
        names = [item["name"] for item in index.get("directory", {}).get("item", [])]
        exhibits = [
            n
            for n in names
            if n.lower().endswith((".htm", ".html"))
            and n != filing["primary_document"]
            and "index" not in n.lower()
            and not re.fullmatch(r"R\d+\.htm", n)
        ]
        exhibits.sort(key=lambda n: "99" not in n)

        if not exhibits:
            return _html_to_text(self._get_text(filing["filing_url"]))

        parts = []
        for name in exhibits:
            exhibit_url = EDGAR_ARCHIVE_URL.format(
                cik=filing["cik"], accession_no_dash=accession_no_dash, document=name
            )
            parts.append(_html_to_text(self._get_text(exhibit_url)))
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Fixtures path (mock-first dev mode)
    # ------------------------------------------------------------------

    def _fetch_from_fixtures(
        self, tickers: list[str], lookback_days: int
    ) -> list[dict]:
        path = Path(Config.FIXTURES_DIR) / "edgar_transcripts.json"
        if not path.exists():
            logger.warning("Fixture file %s not found", path)
            return []

        all_records = json.loads(path.read_text())
        ticker_set = {t.upper() for t in tickers}
        cutoff = (datetime.now() - timedelta(days=lookback_days)).date()

        results = [
            r
            for r in all_records
            if r["ticker"].upper() in ticker_set
            and datetime.strptime(r["filing_date"], "%Y-%m-%d").date() >= cutoff
        ]
        logger.info(
            "Loaded %d fixture transcripts for %d tickers (USE_FIXTURES=true)",
            len(results),
            len(tickers),
        )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _lookup_cik(self, ticker: str) -> int | None:
        """Map ticker → CIK via SEC's company_tickers.json (cached in-memory)."""
        if self._cik_cache is None:
            data = self._get_json(EDGAR_TICKERS_URL)
            # File is a dict keyed by string indices; values have ticker + cik_str
            self._cik_cache = {
                row["ticker"].upper(): int(row["cik_str"]) for row in data.values()
            }
            logger.info("Loaded %d ticker→CIK mappings", len(self._cik_cache))
        return self._cik_cache.get(ticker.upper())

    def _get_json(self, url: str) -> dict:
        return self._get(url).json()

    def _get_text(self, url: str) -> str:
        return self._get(url).text

    def _get(self, url: str) -> requests.Response:
        """GET with pacing and exponential-backoff retries.

        SEC throttles bursts (403/429) and occasionally drops connections;
        without retries one transient failure silently drops a whole
        ticker's filings from the run.
        """
        delay = RETRY_BASE_DELAY_SECONDS
        for attempt in range(MAX_RETRIES):
            time.sleep(REQUEST_INTERVAL_SECONDS)
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code not in RETRYABLE_STATUS_CODES:
                    resp.raise_for_status()
                    return resp
                reason = f"HTTP {resp.status_code}"
            except (requests.ConnectionError, requests.Timeout) as e:
                reason = type(e).__name__
            if attempt == MAX_RETRIES - 1:
                break
            logger.warning(
                "EDGAR request failed (%s), retry %d/%d in %.0fs: %s",
                reason, attempt + 1, MAX_RETRIES - 1, delay, url,
            )
            time.sleep(delay)
            delay *= 2
        raise requests.HTTPError(f"EDGAR request failed after {MAX_RETRIES} attempts ({reason}): {url}")

# ----------------------------------------------------------------------
# Module-level helpers (pure functions — easy to unit test)
# ----------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def dedupe_earnings_events(
    transcripts: list[dict], window_days: int = EARNINGS_DEDUPE_WINDOW_DAYS
) -> list[dict]:
    """Keep one filing per earnings event.

    A company typically files its 8-K earnings press release and its
    10-Q/10-K within days of each other; both would get labeled with
    the same price reaction, double-counting the event. Keep the 8-K
    (it's what the market reacts to) and drop any 10-Q/10-K filed within
    `window_days` after an 8-K for the same ticker. A periodic report
    with no nearby 8-K is kept as the event's only source.
    """
    def _date(t: dict) -> datetime:
        return datetime.strptime(t["filing_date"], "%Y-%m-%d")

    eightk_dates: dict[str, list[datetime]] = {}
    for t in transcripts:
        if t.get("form") == "8-K":
            eightk_dates.setdefault(t["ticker"], []).append(_date(t))

    kept = []
    for t in transcripts:
        if t.get("form") != "8-K":
            d = _date(t)
            if any(
                timedelta(0) <= d - k <= timedelta(days=window_days)
                for k in eightk_dates.get(t["ticker"], [])
            ):
                continue
        kept.append(t)
    return kept


def _html_to_text(html: str) -> str:
    """Strip HTML tags and collapse whitespace.

    Intentionally lightweight — full BeautifulSoup parsing would be more
    accurate but adds a dependency we don't otherwise need. EDGAR HTML is
    typically clean enough for regex stripping to produce usable text for
    LLM extraction downstream.
    """
    # Inline-XBRL filings embed a hidden <ix:header> block of tagging
    # metadata before the visible document; drop it so it doesn't eat the
    # extractor's character budget.
    html = _IX_HEADER_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", html)
    text = htmllib.unescape(text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


_IX_HEADER_RE = re.compile(r"<ix:header>.*?</ix:header>", re.IGNORECASE | re.DOTALL)


_QUARTER_RE = re.compile(r"(?:fiscal\s+)?(?:Q([1-4])|first|second|third|fourth)\s+quarter", re.IGNORECASE)
_YEAR_RE = re.compile(r"(20\d{2})")


def _infer_fiscal_period(text: str, filing_date_str: str) -> tuple[str, int]:
    """Best-effort extraction of fiscal Q# / year from filing text.

    Falls back to calendar quarter / year of the filing date.
    """
    sample = text[:2000]  # only scan the head — quarter is mentioned early.
    fiscal_year: int
    quarter: str

    q_match = _QUARTER_RE.search(sample)
    if q_match:
        word_to_q = {
            "first": "Q1",
            "second": "Q2",
            "third": "Q3",
            "fourth": "Q4",
        }
        if q_match.group(1):
            quarter = f"Q{q_match.group(1)}"
        else:
            # The first capture group was empty, so the match used a word
            # alternative ("first"/"second"/...). Scan tokens for it because
            # the match may also include a leading "fiscal".
            tokens = [t.lower() for t in q_match.group(0).split()]
            quarter = next(
                (word_to_q[t] for t in tokens if t in word_to_q),
                "Q?",
            )
    else:
        # Calendar-quarter fallback from filing date.
        month = datetime.strptime(filing_date_str, "%Y-%m-%d").month
        quarter = f"Q{((month - 1) // 3) + 1}"

    y_match = _YEAR_RE.search(sample)
    fiscal_year = int(y_match.group(1)) if y_match else int(filing_date_str[:4])

    return quarter, fiscal_year
