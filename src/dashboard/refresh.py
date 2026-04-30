"""Lightweight wrapper used by the Airflow DAG to refresh dashboard data.

The Streamlit app is read-only against `data/processed/signals.{parquet,json}`,
so "refreshing" the dashboard just means rerunning the pipeline so the
signals file is up to date. We keep this as a thin shim because the DAG's
Python callable signature differs from `pipeline.run`.
"""

from __future__ import annotations

import logging

from src.pipeline import run as run_pipeline

logger = logging.getLogger(__name__)


def refresh(**kwargs) -> int:
    """Run the pipeline and return the row count for Airflow XCom logging."""
    df = run_pipeline()
    n = len(df)
    logger.info("Dashboard refresh complete: %d signal rows", n)
    return n


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    refresh()
