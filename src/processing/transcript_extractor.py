"""Structured extraction from earnings call transcripts using Claude API."""

import hashlib
import json
import logging
from pathlib import Path

import anthropic

from src.config import Config

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are a financial analyst extracting structured data from an earnings call transcript.

Analyze the following transcript and return a JSON object with exactly these fields:

{
  "overall_sentiment": "bullish" | "bearish" | "neutral",
  "sentiment_score": <float between -1.0 and 1.0>,
  "forward_guidance": "raised" | "maintained" | "lowered" | "withdrawn" | "not_mentioned",
  "management_tone": "confident" | "cautious" | "defensive" | "evasive" | "neutral",
  "key_themes": [<list of 3-5 short theme strings, e.g. "margin expansion", "supply chain risk">],
  "revenue_surprise_indicator": "beat" | "miss" | "inline" | "unclear",
  "analyst_sentiment": "positive" | "negative" | "mixed" | "neutral",
  "risk_flags": [<list of 0-3 short risk strings, e.g. "regulatory headwind", "demand softening">]
}

Return ONLY valid JSON. No explanation, no markdown."""

# Enforced server-side via structured outputs, so the response is always
# parseable JSON (no markdown fences, no missing fields).
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_sentiment": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
        "sentiment_score": {"type": "number"},
        "forward_guidance": {
            "type": "string",
            "enum": ["raised", "maintained", "lowered", "withdrawn", "not_mentioned"],
        },
        "management_tone": {
            "type": "string",
            "enum": ["confident", "cautious", "defensive", "evasive", "neutral"],
        },
        "key_themes": {"type": "array", "items": {"type": "string"}},
        "revenue_surprise_indicator": {
            "type": "string",
            "enum": ["beat", "miss", "inline", "unclear"],
        },
        "analyst_sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "mixed", "neutral"],
        },
        "risk_flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "overall_sentiment",
        "sentiment_score",
        "forward_guidance",
        "management_tone",
        "key_themes",
        "revenue_surprise_indicator",
        "analyst_sentiment",
        "risk_flags",
    ],
    "additionalProperties": False,
}


# Extractions are cached per (model, filing URL) so re-running training
# never pays twice for the same filing.
EXTRACTION_CACHE_DIR = Path("data/processed/extraction_cache")


class TranscriptExtractor:
    """Extracts structured features from earnings transcripts via Claude API."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.model = "claude-haiku-5-5"

    def extract(self, transcript_text: str, ticker: str = "") -> dict | None:
        """
        Extract structured sentiment and theme features from a transcript.

        Args:
            transcript_text: Full text of the earnings call transcript
            ticker: Stock ticker for logging context

        Returns:
            Dict of extracted features, or None on failure
        """
        try:
            # Truncate very long transcripts to fit context window
            max_chars = 80_000
            if len(transcript_text) > max_chars:
                transcript_text = transcript_text[:max_chars]
                logger.warning(
                    f"Truncated {ticker} transcript to {max_chars} chars"
                )

            # The model thinks by default and thinking tokens count against
            # max_tokens, so leave headroom; low effort keeps this simple
            # classification task cheap.
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                output_config={
                    "effort": "low",
                    "format": {"type": "json_schema", "schema": EXTRACTION_SCHEMA},
                },
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"{EXTRACTION_PROMPT}\n\n"
                            f"--- TRANSCRIPT ({ticker}) ---\n"
                            f"{transcript_text}"
                        ),
                    }
                ],
            )

            if response.stop_reason in ("max_tokens", "refusal"):
                logger.error(f"Extraction for {ticker} stopped early: {response.stop_reason}")
                return None

            # The response may lead with a thinking block; the JSON is in
            # the text block.
            raw_text = next(b.text for b in response.content if b.type == "text")
            features = json.loads(raw_text)

            # Validate expected fields
            required_fields = {
                "overall_sentiment",
                "sentiment_score",
                "forward_guidance",
                "management_tone",
                "key_themes",
            }
            if not required_fields.issubset(features.keys()):
                missing = required_fields - features.keys()
                logger.warning(f"Missing fields for {ticker}: {missing}")

            features["ticker"] = ticker
            logger.info(
                f"Extracted features for {ticker}: "
                f"sentiment={features.get('overall_sentiment')}, "
                f"score={features.get('sentiment_score')}"
            )
            return features

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error for {ticker}: {e}")
            return None
        except Exception as e:
            logger.error(f"Extraction failed for {ticker}: {e}")
            return None

    def _cache_path(self, filing_url: str) -> Path | None:
        if not filing_url:
            return None
        key = hashlib.sha256(f"{self.model}|{filing_url}".encode()).hexdigest()[:24]
        return EXTRACTION_CACHE_DIR / f"{key}.json"

    def extract_batch(self, transcripts: list[dict], cached_only: bool = False) -> list[dict]:
        """
        Extract features from multiple transcripts.

        Args:
            transcripts: List of dicts with 'ticker' and 'transcript_text'

        Returns:
            List of extracted feature dicts
        """
        results = []
        cache_hits = 0
        skipped_uncached = 0
        for t in transcripts:
            cache_path = self._cache_path(t.get("filing_url", ""))
            if cache_path and cache_path.exists():
                features = json.loads(cache_path.read_text())
                cache_hits += 1
            elif cached_only:
                # Never call the API in this mode — skip uncached filings.
                skipped_uncached += 1
                continue
            else:
                features = self.extract(
                    transcript_text=t["transcript_text"],
                    ticker=t.get("ticker", "UNKNOWN"),
                )
                if features and cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(features))
            if features:
                features["filing_date"] = t.get("filing_date", "")
                results.append(features)

        logger.info(
            f"Batch extraction complete: {len(results)}/{len(transcripts)} succeeded "
            f"({cache_hits} from cache, {skipped_uncached} uncached skipped)"
        )
        return results
