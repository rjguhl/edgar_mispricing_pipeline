"""
EDGAR Mispricing Pipeline — Airflow DAG

Nightly orchestration: ingest → transform (dbt) → score → refresh dashboard.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "rjguhl",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="edgar_mispricing_pipeline",
    default_args=default_args,
    description="Nightly pipeline: EDGAR + FRED ingestion → dbt → scoring → dashboard",
    schedule_interval="0 2 * * *",  # 2 AM daily
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mispricing", "edgar", "prediction-markets"],
) as dag:

    # --- Stage 1: Ingestion ---

    def ingest_edgar(**kwargs):
        """Fetch earnings call transcripts from SEC EDGAR."""
        # TODO: Implement in src/ingestion/edgar_client.py
        pass

    def ingest_fred(**kwargs):
        """Pull macroeconomic indicators from FRED."""
        # TODO: Implement in src/ingestion/fred_client.py
        pass

    def ingest_contracts(**kwargs):
        """Retrieve prediction market contract prices."""
        # TODO: Implement in src/ingestion/contract_client.py
        pass

    t_ingest_edgar = PythonOperator(
        task_id="ingest_edgar",
        python_callable=ingest_edgar,
    )

    t_ingest_fred = PythonOperator(
        task_id="ingest_fred",
        python_callable=ingest_fred,
    )

    t_ingest_contracts = PythonOperator(
        task_id="ingest_contracts",
        python_callable=ingest_contracts,
    )

    # --- Stage 2: Transform (dbt) ---

    t_dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command="cd /opt/airflow/dbt && dbt run --profiles-dir .",
    )

    t_dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command="cd /opt/airflow/dbt && dbt test --profiles-dir .",
    )

    # --- Stage 3: Scoring ---

    def extract_features(**kwargs):
        """Run Claude API extraction on new transcripts."""
        # TODO: Implement in src/processing/transcript_extractor.py
        pass

    def score_model(**kwargs):
        """Generate calibrated probabilities from XGBoost model."""
        # TODO: Implement in src/modeling/scorer.py
        pass

    def flag_mispricings(**kwargs):
        """Compare model probs to market prices, flag divergences."""
        # TODO: Implement in src/modeling/signal.py
        pass

    t_extract = PythonOperator(
        task_id="extract_features",
        python_callable=extract_features,
    )

    t_score = PythonOperator(
        task_id="score_model",
        python_callable=score_model,
    )

    t_flag = PythonOperator(
        task_id="flag_mispricings",
        python_callable=flag_mispricings,
    )

    # --- Stage 4: Dashboard refresh ---

    def refresh_dashboard(**kwargs):
        """Update Streamlit data cache."""
        # TODO: Implement in src/dashboard/refresh.py
        pass

    t_dashboard = PythonOperator(
        task_id="refresh_dashboard",
        python_callable=refresh_dashboard,
    )

    # --- Dependencies ---
    # Ingestion runs in parallel
    [t_ingest_edgar, t_ingest_fred, t_ingest_contracts] >> t_dbt_run
    t_dbt_run >> t_dbt_test >> t_extract >> t_score >> t_flag >> t_dashboard
