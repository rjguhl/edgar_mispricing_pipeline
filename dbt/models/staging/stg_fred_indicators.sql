-- Staging model: clean raw FRED macroeconomic indicators

with source as (
    select * from {{ source('raw', 'fred_indicators') }}
),

cleaned as (
    select
        indicator_id,
        indicator_name,
        observation_date,
        value,
        units,
        loaded_at
    from source
    where value is not null
)

select * from cleaned
