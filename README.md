# EDGAR Mispricing Pipeline

An end-to-end data pipeline that extracts signals from SEC earnings call transcripts and FRED macroeconomic indicators, builds a calibrated probability model, and detects mispricings in prediction market contracts on Kalshi and Polymarket.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![Airflow](https://img.shields.io/badge/orchestration-Apache%20Airflow-017CEE)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

Prediction markets price discrete real-world events, but contracts are often thinly traded and driven by fragmented information. This pipeline explores whether structured analysis of authoritative financial text sources can reveal early indicators of market-relevant information before they are reflected in contract premiums.

**Key idea:** Rather than treating model output as a standalone prediction, the system anchors it against a live market price — framing the task as *mispricing detection*.

## Architecture

```
┌─────────────┐   ┌─────────────┐   ┌─────────────────┐
│  SEC EDGAR  │   │  FRED API   │   │ Kalshi/Polymarket│
│ Transcripts │   │ Macro Data  │   │ Contract Prices  │
└──────┬──────┘   └──────┬──────┘   └────────┬─────────┘
       │                 │                    │
       ▼                 ▼                    ▼
┌──────────────────────────────────────────────────────┐
│              Apache Airflow (Nightly DAG)             │
├──────────────────────────────────────────────────────┤
│  Ingest  →  Transform (dbt)  →  Score  →  Dashboard  │
└──────────────────────────────────────────────────────┘
       │                 │
       ▼                 ▼
┌─────────────┐   ┌─────────────┐
│  Amazon S3  │   │ PostgreSQL  │
│  (Raw Data) │   │(Feature Store)│
└─────────────┘   └──────┬──────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Claude API         │
              │  Structured Extract │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  XGBoost + Platt    │
              │  Calibrated Probs   │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Streamlit Dashboard │
              │ Signals & Backtest  │
              └─────────────────────┘
```

## Pipeline Stages

### 1. Ingestion
- SEC EDGAR earnings call transcripts for S&P 500 constituents (rolling 90-day window)
- FRED macroeconomic indicators (CPI, unemployment, yield curve, consumer sentiment)
- Kalshi and Polymarket contract prices with timestamps

### 2. Transformation (dbt)
- Raw data staged and transformed into analysis-ready tables
- Schema contracts and data quality tests enforced
- Feature store populated in PostgreSQL, keyed by ticker and event date

### 3. Processing & Feature Engineering
- **Claude API extraction**: Structured sentiment scores, financial theme tags, forward guidance indicators, and management tone from earnings transcripts
- **Macro features**: Rate-of-change, surprise metrics (actual vs. consensus), rolling volatility

### 4. Modeling
- XGBoost classifier on historical earnings events
- Platt scaling for probability calibration
- Lead-lag analysis aligning feature spikes with contract price movements

### 5. Signal & Output
- Mispricing signal flagged when calibrated probability diverges from market price by configurable threshold
- Streamlit dashboard: live signals, model vs. market probability, sentiment trends, pipeline health
- Backtesting module: directional accuracy, precision/recall

## Tech Stack

| Layer | Tools |
|-------|-------|
| Ingestion | SEC EDGAR REST API, FRED API, Kalshi API, Polymarket API |
| Orchestration | Apache Airflow |
| Storage | Amazon S3 (raw), PostgreSQL (feature store) |
| Transformation | dbt |
| NLP & Features | Claude API (structured extraction), spaCy |
| Modeling | XGBoost, LightGBM, scikit-learn (Platt scaling) |
| Dashboard | Streamlit, matplotlib |
| CI/CD | GitHub Actions |

## Project Structure

```
edgar-mispricing-pipeline/
├── dags/                    # Airflow DAG definitions
│   └── mispricing_dag.py
├── dbt/                     # dbt project
│   ├── models/
│   │   ├── staging/         # Raw → cleaned
│   │   └── features/        # Cleaned → feature store
│   └── tests/               # Data quality tests
├── src/
│   ├── ingestion/           # API clients (EDGAR, FRED, Kalshi, Polymarket)
│   ├── processing/          # Claude API extraction, macro feature engineering
│   ├── modeling/            # XGBoost training, calibration, lead-lag
│   └── dashboard/           # Streamlit app
├── tests/                   # Unit and integration tests
├── notebooks/               # Exploratory analysis
├── docs/                    # Additional documentation
├── .github/workflows/       # CI/CD
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/edgar-mispricing-pipeline.git
cd edgar-mispricing-pipeline

# Environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add your API keys to .env
```

## Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```
ANTHROPIC_API_KEY=your_key_here
FRED_API_KEY=your_key_here
KALSHI_API_KEY=your_key_here
POLYMARKET_API_KEY=your_key_here
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_key_here
S3_BUCKET=your_bucket_name
DATABASE_URL=postgresql://user:pass@localhost:5432/mispricing
```

## License

MIT
