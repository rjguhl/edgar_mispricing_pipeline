-- Test: market_probability must be between 0 and 1
select
    contract_id,
    market_probability
from {{ ref('stg_contract_prices') }}
where market_probability < 0 or market_probability > 1
