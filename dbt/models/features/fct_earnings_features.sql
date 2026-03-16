-- Feature store: join transcript extraction features with macro indicators
-- and latest contract prices, keyed by ticker + event_date

with transcript_features as (
    select * from {{ ref('stg_edgar_transcripts') }}
),

macro as (
    select
        observation_date,
        indicator_id,
        value,
        value - lag(value) over (
            partition by indicator_id order by observation_date
        ) as value_change,
        avg(value) over (
            partition by indicator_id
            order by observation_date
            rows between 30 preceding and current row
        ) as rolling_avg_30d
    from {{ ref('stg_fred_indicators') }}
),

-- Pivot macro indicators into columns per event date
macro_wide as (
    select
        observation_date,
        max(case when indicator_id = 'CPIAUCSL' then value end) as cpi,
        max(case when indicator_id = 'CPIAUCSL' then value_change end) as cpi_change,
        max(case when indicator_id = 'UNRATE' then value end) as unemployment,
        max(case when indicator_id = 'UNRATE' then value_change end) as unemployment_change,
        max(case when indicator_id = 'T10Y2Y' then value end) as yield_curve,
        max(case when indicator_id = 'UMCSENT' then value end) as consumer_sentiment
    from macro
    group by observation_date
),

contracts as (
    select * from {{ ref('stg_contract_prices') }}
),

joined as (
    select
        t.ticker,
        t.filing_date as event_date,
        t.fiscal_quarter,
        t.fiscal_year,
        t.transcript_length,

        -- Macro features (latest available before filing)
        m.cpi,
        m.cpi_change,
        m.unemployment,
        m.unemployment_change,
        m.yield_curve,
        m.consumer_sentiment,

        -- Contract price (latest snapshot before filing)
        c.market_probability,
        c.platform,
        c.volume as contract_volume

    from transcript_features t

    left join macro_wide m
        on m.observation_date = (
            select max(observation_date)
            from macro_wide
            where observation_date <= t.filing_date
        )

    left join contracts c
        on c.ticker = t.ticker
        and c.event_date = t.filing_date
        and c.snapshot_ts = (
            select max(snapshot_ts)
            from contracts
            where ticker = t.ticker
              and event_date = t.filing_date
              and snapshot_ts <= t.filing_date
        )
)

select * from joined
