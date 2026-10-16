"""Streamlit dashboard for the EDGAR Mispricing Pipeline.

Run locally:
    streamlit run src/dashboard/app.py

Pages:
    - Signals  : beat model vs. open Polymarket earnings contracts
    - Backtest : model vs. market on resolved contracts (src.modeling.backtest)
    - Pipeline : trigger a fresh signals run from the UI
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure the project root is importable when launched via
# `streamlit run src/dashboard/app.py` (Streamlit doesn't add CWD to sys.path).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import Config  # noqa: E402
from src.modeling.backtest import (  # noqa: E402
    RESULTS_PATH,
    calibration_table,
    score_forecasts,
    simulate_trades,
)
from src.pipeline import run as run_pipeline  # noqa: E402

SIGNALS_PATH = Path("data/processed/signals.json")


# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="EDGAR Mispricing Pipeline",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Data loaders (cached)
# ----------------------------------------------------------------------


@st.cache_data(ttl=60)
def load_signals() -> pd.DataFrame:
    if not SIGNALS_PATH.exists():
        return pd.DataFrame()
    df = pd.DataFrame(json.loads(SIGNALS_PATH.read_text()))
    if df.empty:
        return df
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df


@st.cache_data(ttl=60)
def load_backtest() -> pd.DataFrame:
    return pd.read_parquet(RESULTS_PATH) if RESULTS_PATH.exists() else pd.DataFrame()


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

st.sidebar.title("EDGAR Mispricing")
st.sidebar.caption("Calibrated probabilities vs. market prices")

page = st.sidebar.radio(
    "Page",
    ["Signals", "Backtest", "Pipeline"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.write(f"**Mode:** {'Fixtures' if Config.USE_FIXTURES else 'Live APIs'}")
st.sidebar.write(f"**Threshold:** {Config.MISPRICING_THRESHOLD:.2f}")

if SIGNALS_PATH.exists():
    mtime = pd.Timestamp(SIGNALS_PATH.stat().st_mtime, unit="s")
    st.sidebar.write(f"**Last run:** {mtime:%Y-%m-%d %H:%M}")
else:
    st.sidebar.warning("No signals file yet — visit Pipeline to run.")


# ----------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------


def page_signals() -> None:
    st.title("Signals")
    st.caption("Live mispricings: model probability vs. market contract price.")

    df = load_signals()
    if df.empty:
        st.info("No signals yet. Open **Pipeline** in the sidebar to run a fresh pass.")
        return

    # Top-line metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total contracts", len(df))
    c2.metric("Underpriced", int((df["signal"] == "underpriced").sum()))
    c3.metric("Overpriced", int((df["signal"] == "overpriced").sum()))
    c4.metric("Avg |divergence|", f"{df['abs_divergence'].mean():.3f}")

    st.divider()

    # Filters
    f1, f2, f3 = st.columns([2, 2, 1])
    selected_signals = f1.multiselect(
        "Signal type",
        options=["underpriced", "overpriced", "no_signal"],
        default=["underpriced", "overpriced"],
    )
    selected_platforms = f2.multiselect(
        "Platform",
        options=sorted(df["platform"].unique()),
        default=sorted(df["platform"].unique()),
    )
    min_volume = f3.number_input("Min volume", min_value=0, value=0, step=1000)

    filtered = df[
        df["signal"].isin(selected_signals)
        & df["platform"].isin(selected_platforms)
        & (df["volume"] >= min_volume)
    ]

    # Scatter: market vs model probability
    if not filtered.empty:
        fig = px.scatter(
            filtered,
            x="market_prob",
            y="model_prob",
            color="signal",
            symbol="platform",
            size=filtered["volume"].clip(lower=1),
            hover_data=["ticker", "event_date", "divergence"],
            color_discrete_map={
                "underpriced": "#38b676",
                "overpriced": "#ef5350",
                "no_signal": "#888",
            },
            title="Model vs market probability",
        )
        fig.add_shape(
            type="line", x0=0, y0=0, x1=1, y1=1,
            line={"dash": "dash", "color": "#666"},
        )
        fig.update_layout(
            xaxis_range=[0, 1],
            yaxis_range=[0, 1],
            xaxis_title="Market probability",
            yaxis_title="Model probability (calibrated)",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Signal table
    st.subheader("Ranked signals")
    table = filtered.copy()
    table["event_date"] = table["event_date"].dt.strftime("%Y-%m-%d")
    table["model_prob"] = table["model_prob"].round(3)
    table["market_prob"] = table["market_prob"].round(3)
    table["divergence"] = table["divergence"].round(3)
    st.dataframe(
        table[
            [
                "ticker",
                "event_date",
                "filing_date",
                "platform",
                "model_prob",
                "market_prob",
                "divergence",
                "signal",
                "volume",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def page_backtest() -> None:
    st.title("Backtest")
    st.caption(
        "Resolved Polymarket \"beat quarterly earnings\" contracts: the market's "
        "last price before the release vs. the beat model's walk-forward "
        "probability, scored against how each contract actually resolved."
    )

    bt = load_backtest()
    if bt.empty:
        st.info("No backtest results yet. Run `python -m src.modeling.backtest`.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Contracts", len(bt))
    c2.metric("Period", f"{bt['release_date'].min():%b %Y} – {bt['release_date'].max():%b %Y}")
    c3.metric("Actual beat rate", f"{bt['outcome'].mean():.0%}")

    st.subheader("Forecast accuracy (lower Brier / log loss is better)")
    st.dataframe(score_forecasts(bt).round(4), use_container_width=True, hide_index=True)

    st.subheader("Calibration")
    left, right = st.columns(2)
    left.write("**Market**")
    left.dataframe(calibration_table(bt, "market_prob"), use_container_width=True)
    right.write("**Model**")
    right.dataframe(calibration_table(bt, "model_prob"), use_container_width=True)

    st.subheader("Trading simulation")
    st.caption("Buy the side the model favors when it disagrees with the market by more than the threshold. No fees or spread.")
    st.dataframe(
        pd.DataFrame([simulate_trades(bt, t) for t in (0.10, 0.20, 0.30)]).round(3),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.scatter(
        bt, x="market_prob", y="model_prob", color=bt["outcome"].map({1: "beat", 0: "miss"}),
        hover_data=["ticker", "release_date"], opacity=0.6,
        color_discrete_map={"beat": "#38b676", "miss": "#ef5350"},
        title="Model vs market probability, colored by actual outcome",
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "#666"})
    fig.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1], height=500)
    st.plotly_chart(fig, use_container_width=True)


def page_pipeline() -> None:
    st.title("Pipeline")
    st.caption("Trigger a fresh signals run from the browser.")

    st.write(
        "This calls `src.pipeline.run()`, which fetches open Polymarket earnings "
        "contracts, scores each company's next-quarter beat probability, and "
        "writes `data/processed/signals.json`."
    )

    if Config.USE_FIXTURES:
        st.success("USE_FIXTURES=true — runs locally with bundled sample data, no API calls.")
    else:
        st.warning("USE_FIXTURES=false — this will hit live Polymarket / Kalshi / Yahoo Finance endpoints (no Claude calls).")

    if st.button("Run pipeline now", type="primary"):
        with st.spinner("Running pipeline…"):
            df = run_pipeline()
        st.cache_data.clear()
        if df.empty:
            st.error("Pipeline produced no signals. Check logs.")
        else:
            st.success(f"Wrote {len(df)} signal rows. Open the Signals page to view.")
            st.dataframe(df.head(20), use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------

PAGES = {
    "Signals": page_signals,
    "Backtest": page_backtest,
    "Pipeline": page_pipeline,
}
PAGES[page]()
