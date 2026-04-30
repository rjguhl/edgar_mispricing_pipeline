"""Streamlit dashboard for the EDGAR Mispricing Pipeline.

Run locally:
    streamlit run src/dashboard/app.py

Pages:
    - Signals       : current mispricings flagged by the model
    - Macro         : engineered FRED feature time series
    - Transcripts   : per-ticker structured extraction view
    - Backtest      : naive directional accuracy over the signals frame
    - Pipeline      : trigger a fresh end-to-end run from the UI
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import Config
from src.pipeline import run as run_pipeline
from src.processing.macro_features import MacroFeatureBuilder

# Ensure the project root is importable when launched via
# `streamlit run src/dashboard/app.py` (Streamlit doesn't add CWD to sys.path).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

SIGNALS_PATH = Path("data/processed/signals.json")
FIXTURES_DIR = Path(Config.FIXTURES_DIR)


# ----------------------------------------------------------------------
# Page config + global styling
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="EDGAR Mispricing Pipeline",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .signal-card {padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;}
    .signal-underpriced {background: #1e3a2e; border-left: 4px solid #38b676;}
    .signal-overpriced  {background: #3a1e1e; border-left: 4px solid #ef5350;}
    .signal-none        {background: #2c2c2c; border-left: 4px solid #6b6b6b;}
    .metric-label {color: #aaa; font-size: 0.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
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
def load_fred_fixture() -> pd.DataFrame:
    path = FIXTURES_DIR / "fred_indicators.json"
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame(json.loads(path.read_text()))


@st.cache_data(ttl=60)
def load_transcripts_fixture() -> pd.DataFrame:
    path = FIXTURES_DIR / "edgar_transcripts.json"
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame(json.loads(path.read_text()))


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

st.sidebar.title("EDGAR Mispricing")
st.sidebar.caption("Calibrated probabilities vs. market prices")

page = st.sidebar.radio(
    "Page",
    ["Signals", "Macro", "Transcripts", "Backtest", "Pipeline"],
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


def page_macro() -> None:
    st.title("Macro features")
    st.caption("Engineered FRED indicator series feeding the model.")

    raw = load_fred_fixture()
    if raw.empty:
        st.warning("No FRED data available.")
        return

    feats = MacroFeatureBuilder().build_features(raw)
    if feats.empty:
        st.warning("Macro feature build returned empty.")
        return

    indicator = st.selectbox(
        "Indicator",
        options=sorted(raw["indicator_id"].unique()),
        format_func=lambda x: f"{x} — {raw[raw.indicator_id==x]['indicator_name'].iloc[0]}",
    )
    prefix = indicator.lower()

    cols = [c for c in feats.columns if c.startswith(prefix)]
    if not cols:
        st.warning(f"No engineered features found for {indicator}.")
        return

    plot_df = feats[cols].reset_index().melt(id_vars="observation_date", var_name="feature")
    fig = px.line(
        plot_df,
        x="observation_date",
        y="value",
        color="feature",
        title=f"{indicator} engineered features",
    )
    fig.update_layout(height=480, xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw feature table"):
        st.dataframe(feats[cols], use_container_width=True)


def page_transcripts() -> None:
    st.title("Transcripts")
    st.caption("EDGAR 8-K Item 2.02 / 10-Q content + (planned) Claude extraction output.")

    df = load_transcripts_fixture()
    if df.empty:
        st.warning("No transcripts available.")
        return

    ticker = st.selectbox("Ticker", options=sorted(df["ticker"].unique()))
    rows = df[df["ticker"] == ticker].sort_values("filing_date", ascending=False)

    for _, row in rows.iterrows():
        with st.expander(
            f"{row['ticker']} — {row['filing_date']} ({row.get('fiscal_quarter', '')} {row.get('fiscal_year', '')})"
        ):
            st.write(f"**Filing URL:** {row.get('filing_url', 'n/a')}")
            st.write(f"**Length:** {len(row['transcript_text'])} characters")
            st.text_area(
                "Transcript",
                value=row["transcript_text"],
                height=300,
                key=f"tx-{row['ticker']}-{row['filing_date']}",
            )


def page_backtest() -> None:
    st.title("Backtest")
    st.caption(
        "Naive accuracy: did the side our signal favored (high model_prob = beat) "
        "actually realize? Real labels will replace the toy ones below in Phase 3."
    )

    signals = load_signals()
    if signals.empty:
        st.info("Run the pipeline first.")
        return

    # Toy label: use the heuristic-strong signals as proxies for realized outcomes.
    # When real earnings outcomes are joined in, this section will report
    # precision/recall against actual beat/miss labels.
    s = signals.copy()
    s["predicted_beat"] = s["model_prob"] > 0.5
    s["market_beat"] = s["market_prob"] > 0.5
    s["agree"] = s["predicted_beat"] == s["market_beat"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(s))
    c2.metric("Model–market agreement", f"{s['agree'].mean():.0%}")
    c3.metric("Avg signal size", f"{s['abs_divergence'].mean():.3f}")

    st.subheader("Divergence distribution")
    fig = px.histogram(
        s, x="divergence", color="signal", nbins=20,
        color_discrete_map={
            "underpriced": "#38b676", "overpriced": "#ef5350", "no_signal": "#888",
        },
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "**Phase-3 note:** real earnings actuals are not yet joined in. "
        "These metrics are illustrative scaffolding only."
    )


def page_pipeline() -> None:
    st.title("Pipeline")
    st.caption("Trigger an end-to-end pipeline run from the browser.")

    st.write(
        "This calls `src.pipeline.run()` which ingests transcripts + FRED + "
        "contracts, builds features, scores them, and writes "
        "`data/processed/signals.json`."
    )

    if Config.USE_FIXTURES:
        st.success("USE_FIXTURES=true — runs locally with bundled sample data, no API calls.")
    else:
        st.warning("USE_FIXTURES=false — this will hit live SEC / FRED / Kalshi / Polymarket endpoints.")

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
    "Macro": page_macro,
    "Transcripts": page_transcripts,
    "Backtest": page_backtest,
    "Pipeline": page_pipeline,
}
PAGES[page]()
