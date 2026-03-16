-- Staging model: clean raw prediction market contract prices

with source as (
    select * from {{ source('raw', 'contract_prices') }}
),

cleaned as (
    select
        contract_id,
        platform,          -- 'kalshi' or 'polymarket'
        ticker,
        event_type,
        event_date,
        price as market_probability,
        volume,
        snapshot_ts,
        loaded_at
    from source
    where price between 0.01 and 0.99  -- filter degenerate contracts
      and volume > 0
)

select * from cleaned
