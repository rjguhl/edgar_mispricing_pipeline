"""
EDGAR Mispricing Pipeline — Airflow DAG

Nightly orchestration: ingest → transform (dbt) → score → refresh dashboard.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from src.config import Config
from src.ingestion.contract_client import ContractClient
from src.ingestion.edgar_client import EdgarClient
from src.ingestion.fred_client import FredClient
from src.modeling.scorer import encode_features, score
from src.modeling.signal import flag_mispricings, write_signals
from src.processing.macro_features import MacroFeatureBuilder
from src.processing.transcript_extractor import TranscriptExtractor

logger = logging.getLogger(__name__)


default_args = {
    "owner": "rjguhl",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# ----------------------------------------------------------------------
# Task callables
# ----------------------------------------------------------------------


def _default_tickers() -> list[str]:
    path = Path(Config.SP500_TICKERS_PATH)
    if not path.exists():
        return ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    import pandas as pd

    return pd.read_csv(path)["ticker"].tolist()


def ingest_edgar(**kwargs):
    """Fetch earnings-related filings from SEC EDGAR and persist to S3."""
    transcripts = EdgarClient().fetch_transcripts(
        _default_tickers(), lookback_days=Config.LOOKBACK_DAYS
    )
    return len(transcripts)


def ingest_fred(**kwargs):
    """Pull macroeconomic indicators from FRED and persist to S3."""
    return len(FredClient().fetch_indicators(lookback_days=Config.LOOKBACK_DAYS))


def ingest_contracts(**kwargs):
    """Retrieve prediction market contract prices from Kalshi + Polymarket."""
    return len(ContractClient().fetch_all(event_types=["earnings"]))


def extract_features(**kwargs):
    """Run Claude API extraction on new transcripts and stash for scoring."""
    edgar = EdgarClient()
    transcripts = edgar.fetch_transcripts(
        _default_tickers(), lookback_days=Config.LOOKBACK_DAYS
    )
    extracted = TranscriptExtractor().extract_batch(transcripts)
    out = Path("data/processed/extracted_features.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(extracted, indent=2, default=str))
    return len(extracted)


def score_model(**kwargs):
    """Generate calibrated probabilities from XGBoost model."""
    import pandas as pd

    extracted = json.loads(Path("data/processed/extracted_features.json").read_text())
    events = pd.DataFrame(extracted)

    macro_records = FredClient().fetch_indicators(lookback_days=Config.LOOKBACK_DAYS)
    macro_df = MacroFeatureBuilder().build_features(pd.DataFrame(macro_records))

    feature_df = encode_features(events, macro_df)
    scored = score(feature_df)

    out = Path("data/processed/scored.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        scored.to_parquet(out, index=False)
    except Exception:
        scored.to_json(out.with_suffix(".json"), orient="records")
    return len(scored)


def flag_mispricings_task(**kwargs):
    """Compare model probs to market prices, flag divergences."""
    import pandas as pd

    scored = pd.read_parquet("data/processed/scored.parquet")
    contract_records = ContractClient().fetch_all(event_types=["earnings"])
    contracts = pd.DataFrame(contract_records).rename(
        columns={"price": "market_probability"}
    )
    signals = flag_mispricings(scored, contracts)
    write_signals(signals)
    return int((signals["signal"] != "no_signal").sum())


def refresh_dashboard(**kwargs):
    """The dashboard reads signals.json on demand, so just verify it exists."""
    p = Path("data/processed/signals.json")
    if not p.exists():
        raise FileNotFoundError(p)
    return p.stat().st_size


# ----------------------------------------------------------------------
# DAG definition
# ----------------------------------------------------------------------


with DAG(
    dag_id="edgar_mispricing_pipeline",
    default_args=default_args,
    description="Nightly pipeline: EDGAR + FRED ingestion → dbt → scoring → dashboard",
    schedule_interval="0 2 * * *",  # 2 AM daily
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mispricing", "edgar", "prediction-markets"],
) as dag:

    t_ingest_edgar = PythonOperator(
        task_id="ingest_edgar", python_callable=ingest_edgar
    )
    t_ingest_fred = PythonOperator(
        task_id="ingest_fred", python_callable=ingest_fred
    )
    t_ingest_contracts = PythonOperator(
        task_id="ingest_contracts", python_callable=ingest_contracts
    )

    t_dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command="cd /opt/airflow/dbt && dbt run --profiles-dir .",
    )
    t_dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command="cd /opt/airflow/dbt && dbt test --profiles-dir .",
    )

    t_extract = PythonOperator(
        task_id="extract_features", python_callable=extract_features
    )
    t_score = PythonOperator(task_id="score_model", python_callable=score_model)
    t_flag = PythonOperator(
        task_id="flag_mispricings", python_callable=flag_mispricings_task
    )
    t_dashboard = PythonOperator(
        task_id="refresh_dashboard", python_callable=refresh_dashboard
    )

    [t_ingest_edgar, t_ingest_fred, t_ingest_contracts] >> t_dbt_run
    t_dbt_run >> t_dbt_test >> t_extract >> t_score >> t_flag >> t_dashboard
