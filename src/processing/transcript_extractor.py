"""Structured extraction from earnings call transcripts using Claude API."""

import json
import logging

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


class TranscriptExtractor:
    """Extracts structured features from earnings transcripts via Claude API."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.model = "claude-sonnet-4-20250514"

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

            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
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

            raw_text = response.content[0].text.strip()
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

    def extract_batch(self, transcripts: list[dict]) -> list[dict]:
        """
        Extract features from multiple transcripts.

        Args:
            transcripts: List of dicts with 'ticker' and 'transcript_text'

        Returns:
            List of extracted feature dicts
        """
        results = []
        for t in transcripts:
            features = self.extract(
                transcript_text=t["transcript_text"],
                ticker=t.get("ticker", "UNKNOWN"),
            )
            if features:
                features["filing_date"] = t.get("filing_date", "")
                results.append(features)

        logger.info(
            f"Batch extraction complete: {len(results)}/{len(transcripts)} succeeded"
        )
        return results
