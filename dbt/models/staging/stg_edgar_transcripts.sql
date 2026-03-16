-- Staging model: clean raw EDGAR transcript data
-- Source: S3 → raw.edgar_transcripts (loaded by ingestion layer)

with source as (
    select * from {{ source('raw', 'edgar_transcripts') }}
),

cleaned as (
    select
        ticker,
        filing_date,
        fiscal_quarter,
        fiscal_year,
        transcript_text,
        filing_url,
        char_length(transcript_text) as transcript_length,
        loaded_at
    from source
    where transcript_text is not null
      and char_length(transcript_text) > 500  -- filter empty/stub filings
)

select * from cleaned
