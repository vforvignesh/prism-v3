"""
PRISM v3 — Quantitative Portfolio Intelligence Dashboard
Bloomberg Terminal-Inspired Design
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

from core.data_pipeline import fetch_all_data
from core.scoring import run_scoring

# ---------------------------------------------------------------------------
#  Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PRISM v3",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
#  Load Config
# ---------------------------------------------------------------------------
CONFIG_DIR = Path(__file__).parent / "config"

@st.cache_data(ttl=3600)
def load_config():
    with open(CONFIG_DIR / "settings.json") as f:
        settings = json.load(f)
    with open(CONFIG_DIR / "watchlist.json") as f:
        watchlist = json.load(f)
    return settings, watchlist

settings, watchlist = load_config()

# API keys from environment (Streamlit secrets / Render) or fallback to config
import os
FMP_KEY = os.environ.get("FMP_API_KEY", settings["api_keys"]["fmp"])
FH_KEY = os.environ.get("FINNHUB_API_KEY", settings["api_keys"]["finnhub"])

# Also check st.secrets for Streamlit Cloud
try:
    FMP_KEY = st.secrets.get("FMP_API_KEY", FMP_KEY)
    FH_KEY = st.secrets.get("FINNHUB_API_KEY", FH_KEY)
except Exception:
    pass

# ---------------------------------------------------------------------------
#  Bloomberg Terminal CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ===== GLOBAL OVERRIDES ===== */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Root background — deep terminal black */
    .stApp {
        background-color: #0a0e17 !important;
    }

    /* Main content area */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1600px;
    }

    /* ===== HEADER ===== */
    .prism-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 0 0.75rem 0;
        border-bottom: 2px solid #FF6600;
        margin-bottom: 1rem;
    }
    .prism-logo {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        color: #FF6600;
    }
    .prism-sub {
        color: #8899AA;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 2px;
    }
    .prism-live {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #00FF88;
        letter-spacing: 0.05em;
    }
    .prism-live::before {
        content: '●';
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: #080c14 !important;
        border-right: 1px solid #1a2233;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4 {
        color: #C0CCD8 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    section[data-testid="stSidebar"] label {
        color: #8899AA !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
    }
    section[data-testid="stSidebar"] .stCaption p {
        color: #556677 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #0d1320;
        border-bottom: 1px solid #1a2233;
        padding: 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #6B7B8D !important;
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #C0CCD8 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #FF6600 !important;
        border-bottom: 2px solid #FF6600 !important;
        background: transparent !important;
    }

    /* ===== KPI METRIC CARDS ===== */
    div[data-testid="stMetric"] {
        background: #0d1320 !important;
        border: 1px solid #1a2233 !important;
        border-radius: 4px !important;
        padding: 0.75rem 1rem !important;
    }
    div[data-testid="stMetric"] label {
        color: #6B7B8D !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.65rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ===== DATAFRAME / TABLE ===== */
    .stDataFrame {
        border: 1px solid #1a2233;
        border-radius: 4px;
    }
    .stDataFrame [data-testid="stDataFrameResizable"] {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Target the table headers and cells */
    .stDataFrame th {
        background: #f7fafc !important;
        color: #2d3748 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        border-bottom: 1px solid #e2e8f0 !important;
    }
    .stDataFrame td {
        color: #1a202c !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        border-bottom: 1px solid #edf2f7 !important;
    }

    /* ===== SECTION HEADERS ===== */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #C0CCD8 !important;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: 0.02em;
    }
    .stMarkdown h3 {
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #8899AA !important;
        border-bottom: 1px solid #1a2233;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        background: #FF6600 !important;
        color: #FFFFFF !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.04em;
        border: none !important;
        border-radius: 3px !important;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: #FF8833 !important;
    }

    /* ===== SCORE BADGES ===== */
    .score-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 3px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 0.8rem;
        letter-spacing: 0.02em;
    }
    .score-high { background: #00402A; color: #00FF88; border: 1px solid #00FF88; }
    .score-mid { background: #3D2E00; color: #FFB800; border: 1px solid #FFB800; }
    .score-low { background: #3D0000; color: #FF4444; border: 1px solid #FF4444; }

    /* ===== DIVIDERS ===== */
    .stMarkdown hr {
        border-color: #1a2233 !important;
    }

    /* ===== CAPTIONS ===== */
    .stCaption p, .stCaption {
        color: #556677 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
    }

    /* ===== SELECT BOXES & INPUTS ===== */
    .stSelectbox label, .stMultiSelect label, .stNumberInput label, .stTextInput label {
        color: #8899AA !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.04em;
    }

    /* ===== MULTISELECT TAGS ===== */
    [data-baseweb="tag"] {
        background: #152030 !important;
        border: 1px solid #FF6600 !important;
        color: #FF6600 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
    }

    /* ===== SLIDER ===== */
    .stSlider label {
        color: #8899AA !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
    }

    /* ===== WARNINGS / INFO ===== */
    .stAlert {
        background: #1a1500 !important;
        border: 1px solid #FFB800 !important;
        color: #FFB800 !important;
        font-family: 'JetBrains Mono', monospace !important;
        border-radius: 3px;
    }

    /* ===== SPINNER ===== */
    .stSpinner > div {
        color: #FF6600 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ===== FOOTER ===== */
    .prism-footer {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: #3D4F5F;
        letter-spacing: 0.04em;
        text-align: center;
        padding: 1rem 0;
        border-top: 1px solid #1a2233;
        margin-top: 2rem;
    }

    /* ===== PLOTLY CHART CONTAINERS ===== */
    .stPlotlyChart {
        border: 1px solid #1a2233;
        border-radius: 4px;
        padding: 0.25rem;
    }

    /* ===== CUSTOM KPI ROW ===== */
    .kpi-card {
        background: #0d1320;
        border: 1px solid #1a2233;
        border-radius: 4px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .kpi-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        color: #6B7B8D;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .kpi-value.green { color: #00FF88; }
    .kpi-value.amber { color: #FFB800; }
    .kpi-value.red { color: #FF4444; }
    .kpi-value.orange { color: #FF6600; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
#  Header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="prism-header">
    <div>
        <div class="prism-logo">◈ PRISM v3</div>
        <div class="prism-sub">Portfolio Risk & Intelligence Scoring Model</div>
    </div>
    <div class="prism-live">LIVE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
#  Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙ Configuration")

    # Watchlist management
    st.markdown("#### Watchlist")
    all_tickers = list(watchlist["tickers"].keys())

    # Add ticker
    new_ticker = st.text_input("Add ticker", placeholder="e.g. AAPL", key="add_ticker")
    if new_ticker and st.button("Add", key="btn_add"):
        sym = new_ticker.upper().strip()
        if sym not in watchlist["tickers"]:
            watchlist["tickers"][sym] = {
                "sector": "", "thesis": "", "added": datetime.now().strftime("%Y-%m-%d")
            }
            with open(CONFIG_DIR / "watchlist.json", "w") as f:
                json.dump(watchlist, f, indent=2)
            st.cache_data.clear()
            st.rerun()
        else:
            st.warning(f"{sym} already in watchlist")

    # Active tickers
    active_tickers = st.multiselect(
        "Active tickers",
        options=all_tickers,
        default=all_tickers,
        key="active_tickers"
    )

    st.markdown("---")

    # Scoring weights
    st.markdown("#### Scoring Weights")
    w_fund = st.slider("Fundamental", 0, 100, int(settings["scoring_weights"]["fundamental"] * 100), 5, key="w_fund")
    w_risk = st.slider("Risk", 0, 100, int(settings["scoring_weights"]["risk"] * 100), 5, key="w_risk")
    w_tech = st.slider("Technical", 0, 100, int(settings["scoring_weights"]["technical"] * 100), 5, key="w_tech")

    total_w = w_fund + w_risk + w_tech
    if total_w > 0:
        settings["scoring_weights"]["fundamental"] = w_fund / total_w
        settings["scoring_weights"]["risk"] = w_risk / total_w
        settings["scoring_weights"]["technical"] = w_tech / total_w
    st.caption(f"Normalized: {w_fund/total_w:.0%} | {w_risk/total_w:.0%} | {w_tech/total_w:.0%}" if total_w > 0 else "")

    st.markdown("---")

    # Index benchmarks
    st.markdown("#### Benchmark")
    idx_pe = st.number_input("Index PE", value=22.0, step=0.5, key="idx_pe")
    idx_growth = st.number_input("Index Growth (%)", value=13.0, step=0.5, key="idx_growth")

    st.markdown("---")

    # Refresh
    if st.button("⟳ REFRESH DATA", use_container_width=True, key="btn_refresh"):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Last: {datetime.now().strftime('%H:%M:%S')}")


# ---------------------------------------------------------------------------
#  Data Fetch & Scoring
# ---------------------------------------------------------------------------
if not active_tickers:
    st.warning("Select at least one ticker in the sidebar.")
    st.stop()

with st.spinner("LOADING MARKET DATA..."):
    raw_df = fetch_all_data(
        tickers=active_tickers,
        fmp_key=FMP_KEY,
        fh_key=FH_KEY,
        growth_overrides=watchlist.get("growth_overrides", {}),
        default_growth=settings["portfolio"]["default_growth_rate"]
    )

    scored_df = run_scoring(raw_df, settings)


# ---------------------------------------------------------------------------
#  Plotly Theme — Bloomberg-style
# ---------------------------------------------------------------------------
PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="#0a0e17",
    plot_bgcolor="#0d1320",
    font=dict(family="JetBrains Mono, monospace", size=11, color="#8899AA"),
    title_font=dict(family="JetBrains Mono, monospace", size=12, color="#C0CCD8"),
    margin=dict(l=10, r=20, t=40, b=10),
    coloraxis_showscale=False,
    showlegend=False,
)

def plotly_layout(**overrides):
    """Return a merged plotly layout dict with grid defaults."""
    base = dict(**PLOTLY_BASE)
    # Apply grid defaults for axes unless overridden
    if "xaxis" not in overrides:
        base["xaxis"] = dict(gridcolor="#1a2233", zerolinecolor="#1a2233")
    if "yaxis" not in overrides:
        base["yaxis"] = dict(gridcolor="#1a2233", zerolinecolor="#1a2233")
    base.update(overrides)
    return base

# Color scale: red → amber → green
PRISM_COLORSCALE = ["#FF4444", "#FFB800", "#00FF88"]


# ---------------------------------------------------------------------------
#  Tab Layout
# ---------------------------------------------------------------------------
tab_watchlist, tab_detail, tab_stress, tab_quality = st.tabs([
    "WATCHLIST", "STOCK DETAIL", "STRESS TEST", "DATA QUALITY"
])


# ========================= TAB 1: WATCHLIST ================================
with tab_watchlist:

    # Custom KPI row using HTML for full styling control
    avg_prism = scored_df["PRISM Score"].mean()
    avg_shock = scored_df["SHOCK Score"].mean()
    avg_edge = scored_df["Edge Ratio"].mean()
    buy_zone_count = len(scored_df[scored_df["52w Position"] < 0.3])

    prism_color = "green" if avg_prism >= 65 else "amber" if avg_prism >= 50 else "red"
    shock_color = "green" if avg_shock > -8 else "amber" if avg_shock > -15 else "red"
    edge_color = "green" if avg_edge > 5 else "amber" if avg_edge > 2 else "red"
    buy_color = "green" if buy_zone_count > 0 else "amber"

    st.markdown(f"""
    <div style="display:grid; grid-template-columns: repeat(5, 1fr); gap: 0.5rem; margin-bottom: 1rem;">
        <div class="kpi-card">
            <div class="kpi-label">STOCKS</div>
            <div class="kpi-value orange">{len(scored_df)}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">AVG PRISM SCORE</div>
            <div class="kpi-value {prism_color}">{avg_prism:.1f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">AVG SHOCK</div>
            <div class="kpi-value {shock_color}">{avg_shock:.1f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">AVG EDGE RATIO</div>
            <div class="kpi-value {edge_color}">{avg_edge:.2f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">IN BUY ZONE</div>
            <div class="kpi-value {buy_color}">{buy_zone_count}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main ranked table
    display_cols = [
        "Rank", "Ticker", "Price", "PE (Fwd)", "Growth",
        "Relative PEG", "Alpha Score", "PRISM Score",
        "Fundamental Score", "Risk Score", "technical_score",
        "SHOCK Score", "Edge Ratio", "1Y Recovery",
        "52w Position", "rsi_14", "Fragile Flags"
    ]
    display_cols = [c for c in display_cols if c in scored_df.columns]
    display_df = scored_df[display_cols].copy()

    # Rename for display
    rename = {"technical_score": "Tech Score", "rsi_14": "RSI"}
    display_df = display_df.rename(columns=rename)

    # Color coding — high contrast for dark background
    def color_prism(val):
        if isinstance(val, (int, float)):
            if val >= 65:
                return "background-color: #003D22; color: #00FF88; font-weight: 600"
            elif val >= 50:
                return "background-color: #3D2E00; color: #FFB800; font-weight: 600"
            else:
                return "background-color: #3D0000; color: #FF4444; font-weight: 600"
        return "color: #1a202c"

    def color_shock(val):
        try:
            v = float(str(val).replace("%", ""))
            if v > -8:
                return "background-color: #003D22; color: #00FF88"
            elif v > -15:
                return "background-color: #3D2E00; color: #FFB800"
            else:
                return "background-color: #3D0000; color: #FF4444"
        except (ValueError, TypeError):
            return "color: #1a202c"

    def color_52w(val):
        try:
            v = float(str(val).replace("%", "")) / 100 if "%" in str(val) else float(val)
            if v < 0.3:
                return "background-color: #003D22; color: #00FF88"
            elif v > 0.9:
                return "background-color: #3D0000; color: #FF4444"
            return "color: #1a202c"
        except (ValueError, TypeError):
            return "color: #1a202c"

    def color_rsi(val):
        try:
            v = float(val)
            if v < 30:
                return "background-color: #003D22; color: #00FF88"  # oversold = opportunity
            elif v > 70:
                return "background-color: #3D0000; color: #FF4444"  # overbought
            return "color: #1a202c"
        except (ValueError, TypeError):
            return "color: #1a202c"

    def color_edge(val):
        try:
            v = float(val)
            if v > 5:
                return "color: #00FF88; font-weight: 600"
            elif v > 2:
                return "color: #FFB800"
            elif v < 0:
                return "color: #FF4444"
            return "color: #1a202c"
        except (ValueError, TypeError):
            return "color: #1a202c"

    def color_recovery(val):
        if val == "YES":
            return "background-color: #003D22; color: #00FF88; font-weight: 600"
        elif val == "MAYBE":
            return "background-color: #3D2E00; color: #FFB800"
        elif val == "NO":
            return "background-color: #3D0000; color: #FF4444"
        return "color: #1a202c"

    def base_text(val):
        return "color: #1a202c"

    def color_flags(val):
        if val and val != "—":
            return "color: #FF4444; font-weight: 500"
        return "color: #556677"

    # Build styler
    styled = display_df.style.format({
        k: v for k, v in {
            "Price": "${:.2f}",
            "PE (Fwd)": "{:.1f}x",
            "Growth": "{:.1%}",
            "Relative PEG": "{:.2f}",
            "Alpha Score": "{:.3f}",
            "PRISM Score": "{:.1f}",
            "Fundamental Score": "{:.0f}",
            "Risk Score": "{:.0f}",
            "Tech Score": "{:.0f}",
            "SHOCK Score": "{:.1f}%",
            "Edge Ratio": "{:.2f}",
            "52w Position": "{:.0%}",
            "RSI": "{:.0f}",
        }.items() if k in display_df.columns
    }, na_rep="—").set_properties(**{
        "color": "#1a202c",
        "font-family": "JetBrains Mono, monospace",
        "font-size": "0.8rem",
        "border-bottom": "1px solid #111927",
    }).set_table_styles([
        {"selector": "th", "props": [
            ("background-color", "#0d1320"),
            ("color", "#8899AA"),
            ("font-family", "JetBrains Mono, monospace"),
            ("font-size", "0.7rem"),
            ("letter-spacing", "0.04em"),
            ("text-transform", "uppercase"),
            ("border-bottom", "1px solid #1a2233"),
            ("padding", "8px 12px"),
        ]},
        {"selector": "td", "props": [
            ("padding", "6px 12px"),
        ]},
    ])

    # Apply conditional formatting
    if "PRISM Score" in display_df.columns:
        styled = styled.map(color_prism, subset=["PRISM Score"])
    if "SHOCK Score" in display_df.columns:
        styled = styled.map(color_shock, subset=["SHOCK Score"])
    if "52w Position" in display_df.columns:
        styled = styled.map(color_52w, subset=["52w Position"])
    if "RSI" in display_df.columns:
        styled = styled.map(color_rsi, subset=["RSI"])
    if "Edge Ratio" in display_df.columns:
        styled = styled.map(color_edge, subset=["Edge Ratio"])
    if "1Y Recovery" in display_df.columns:
        styled = styled.map(color_recovery, subset=["1Y Recovery"])
    if "Fragile Flags" in display_df.columns:
        styled = styled.map(color_flags, subset=["Fragile Flags"])

    # Apply base text color to non-conditional columns
    plain_cols = [c for c in display_df.columns if c not in ["PRISM Score", "SHOCK Score", "52w Position", "RSI", "Edge Ratio", "1Y Recovery", "Fragile Flags"]]
    if plain_cols:
        styled = styled.map(base_text, subset=plain_cols)

    st.dataframe(styled, use_container_width=True, height=min(600, 45 * len(display_df) + 40))

    # --- Charts ---
    st.markdown("### Score Distribution")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        fig_prism = px.bar(
            scored_df.sort_values("PRISM Score", ascending=True),
            x="PRISM Score", y="Ticker", orientation="h",
            color="PRISM Score",
            color_continuous_scale=PRISM_COLORSCALE,
            title="PRISM SCORE BY STOCK"
        )
        fig_prism.update_layout(
            **plotly_layout(
                height=max(300, 35 * len(scored_df)),
                yaxis=dict(autorange="reversed", gridcolor="#1a2233", zerolinecolor="#1a2233"),
            )
        )
        fig_prism.update_traces(
            texttemplate="%{x:.1f}", textposition="outside",
            textfont=dict(color="#C0CCD8", size=10, family="JetBrains Mono")
        )
        st.plotly_chart(fig_prism, use_container_width=True)

    with col_chart2:
        fig_scatter = px.scatter(
            scored_df,
            x="SHOCK Score", y="Growth",
            size="Market Cap ($B)" if "Market Cap ($B)" in scored_df.columns else None,
            color="PRISM Score",
            color_continuous_scale=PRISM_COLORSCALE,
            hover_name="Ticker",
            title="RISK vs GROWTH (BUBBLE = MARKET CAP)",
            labels={"SHOCK Score": "SHOCK Score (%)", "Growth": "Expected Growth"}
        )
        fig_scatter.update_layout(
            **plotly_layout(
                height=max(300, 35 * len(scored_df)),
            )
        )
        fig_scatter.update_yaxes(tickformat=".0%")
        # Add ticker labels to each point
        fig_scatter.update_traces(
            textfont=dict(color="#C0CCD8", size=9, family="JetBrains Mono"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# ========================= TAB 2: STOCK DETAIL =============================
with tab_detail:
    selected_ticker = st.selectbox("Select stock", active_tickers, key="detail_ticker")
    row = scored_df[scored_df["Ticker"] == selected_ticker]

    if row.empty:
        st.warning("No data for selected ticker.")
    else:
        row = row.iloc[0]

        # Header with score badge
        col_h1, col_h2 = st.columns([2, 1])
        with col_h1:
            st.markdown(f"## {selected_ticker}")
            thesis = watchlist["tickers"].get(selected_ticker, {}).get("thesis", "")
            if thesis:
                st.caption(f"Thesis: {thesis}")
        with col_h2:
            prism = row.get("PRISM Score", 0)
            badge_class = "score-high" if prism >= 65 else "score-mid" if prism >= 50 else "score-low"
            st.markdown(f'<div style="text-align:right; margin-top:0.5rem"><span class="score-badge {badge_class}">PRISM {prism:.1f}</span> <span style="color:#6B7B8D; font-family:JetBrains Mono; font-size:0.75rem; margin-left:8px">RANK #{int(row.get("Rank", 0))}</span></div>', unsafe_allow_html=True)

        st.markdown("---")

        # Metrics grid
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Price", f"${row.get('Price', 0):.2f}")
            st.metric("PE (Forward)", f"{row.get('PE (Fwd)', 0):.1f}x")
            st.metric("PE (Trailing)", f"{row.get('PE (TTM)', 0):.1f}x")
        with c2:
            st.metric("Growth", f"{row.get('Growth', 0):.1%}")
            st.metric("Relative PEG", f"{row.get('Relative PEG', 0):.2f}")
            st.metric("Alpha Score", f"{row.get('Alpha Score', 0):.3f}")
        with c3:
            st.metric("SHOCK Score", f"{row.get('SHOCK Score', 0):.1f}%")
            st.metric("Edge Ratio", f"{row.get('Edge Ratio', 0):.2f}")
            st.metric("1Y Recovery", row.get("1Y Recovery", "—"))
        with c4:
            st.metric("52w Position", f"{row.get('52w Position', 0):.0%}")
            st.metric("RSI (14)", f"{row.get('rsi_14', 0):.0f}")
            st.metric("Tech Score", f"{row.get('technical_score', 0):.0f}")

        # Score breakdown chart
        st.markdown("### Score Breakdown")
        scores_data = {
            "Dimension": ["Fundamental", "Risk", "Technical"],
            "Score": [
                row.get("Fundamental Score", 50),
                row.get("Risk Score", 50),
                row.get("technical_score", 50)
            ],
            "Weight": [
                settings["scoring_weights"]["fundamental"],
                settings["scoring_weights"]["risk"],
                settings["scoring_weights"]["technical"]
            ]
        }
        scores_df = pd.DataFrame(scores_data)
        scores_df["Weighted"] = scores_df["Score"] * scores_df["Weight"]

        fig_breakdown = go.Figure()
        fig_breakdown.add_trace(go.Bar(
            name="Raw Score", x=scores_df["Dimension"], y=scores_df["Score"],
            marker_color="#1a2233",
            text=scores_df["Score"].apply(lambda x: f"{x:.0f}"),
            textposition="outside",
            textfont=dict(color="#8899AA", size=11, family="JetBrains Mono")
        ))
        fig_breakdown.add_trace(go.Bar(
            name="Weighted", x=scores_df["Dimension"], y=scores_df["Weighted"],
            marker_color="#FF6600",
            text=scores_df["Weighted"].apply(lambda x: f"{x:.1f}"),
            textposition="outside",
            textfont=dict(color="#FF6600", size=11, family="JetBrains Mono")
        ))
        fig_breakdown.update_layout(
            **plotly_layout(
                barmode="group",
                height=300,
                yaxis=dict(range=[0, 110], gridcolor="#1a2233", zerolinecolor="#1a2233"),
                showlegend=True,
                legend=dict(font=dict(color="#8899AA", size=10)),
            )
        )
        st.plotly_chart(fig_breakdown, use_container_width=True)

        # SHOCK scenario breakdown
        st.markdown("### SHOCK Scenarios")
        shock_cols = [c for c in scored_df.columns if c.startswith("SHOCK: ")]
        if shock_cols:
            shock_data = []
            for col in shock_cols:
                sname = col.replace("SHOCK: ", "")
                prob = settings["shock_scenarios"].get(sname, {}).get("probability", 0)
                dd = row.get(col, 0)
                shock_data.append({
                    "Scenario": sname,
                    "Drawdown": dd,
                    "Probability": prob,
                    "Weighted Impact": dd * prob
                })
            shock_df = pd.DataFrame(shock_data)

            fig_shock = px.bar(
                shock_df, x="Scenario", y="Drawdown",
                color="Drawdown",
                color_continuous_scale=["#FF4444", "#FFB800", "#00FF88"],
                title=f"EXPECTED DRAWDOWN — {selected_ticker}",
            )
            fig_shock.update_layout(
                **plotly_layout(
                    height=300,
                )
            )
            fig_shock.update_traces(
                text=shock_df["Drawdown"].apply(lambda x: f"{x:.1f}%"),
                textposition="outside",
                textfont=dict(color="#C0CCD8", size=10, family="JetBrains Mono")
            )
            st.plotly_chart(fig_shock, use_container_width=True)

        # Fragile flags
        flags = row.get("Fragile Flags", "—")
        if flags and flags != "—":
            st.markdown(f"""
            <div style="background:#3D0000; border:1px solid #FF4444; border-radius:3px; padding:0.6rem 1rem; margin:0.5rem 0; font-family:JetBrains Mono, monospace; font-size:0.8rem; color:#FF4444;">
                ⚠ FRAGILE FLAGS: {flags}
            </div>
            """, unsafe_allow_html=True)

        # Technical detail
        st.markdown("### Technical Indicators")
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            st.metric("SMA(50)", f"${row.get('sma_50', 0):.2f}")
            sma200_val = row.get('sma_200', np.nan)
            st.metric("SMA(200)", f"${sma200_val:.2f}" if not np.isnan(sma200_val) else "N/A")
        with tc2:
            st.metric("MACD", f"{row.get('macd', 0):.2f}")
            st.metric("MACD Signal", f"{row.get('macd_signal', 0):.2f}")
        with tc3:
            bb_pctb = row.get('bb_pctb', np.nan)
            st.metric("BB %B", f"{bb_pctb:.2f}" if not np.isnan(bb_pctb) else "N/A")
            rv = row.get('rel_volume', np.nan)
            st.metric("Rel. Volume", f"{rv:.2f}x" if not np.isnan(rv) else "N/A")


# ========================= TAB 3: STRESS TEST ==============================
with tab_stress:
    st.markdown("### Portfolio-Level Stress Test")
    st.caption("Aggregate SHOCK impact across active watchlist")

    # Portfolio-level SHOCK
    shock_cols = [c for c in scored_df.columns if c.startswith("SHOCK: ")]
    if shock_cols:
        port_shock = {}
        for col in shock_cols:
            sname = col.replace("SHOCK: ", "")
            avg_dd = scored_df[col].mean()
            prob = settings["shock_scenarios"].get(sname, {}).get("probability", 0)
            port_shock[sname] = {"Avg Drawdown": avg_dd, "Probability": prob, "Weighted": avg_dd * prob}

        port_df = pd.DataFrame(port_shock).T.reset_index()
        port_df.columns = ["Scenario", "Avg Drawdown (%)", "Probability", "Weighted Impact (%)"]

        st.dataframe(
            port_df.style.format({
                "Avg Drawdown (%)": "{:.1f}%",
                "Probability": "{:.0%}",
                "Weighted Impact (%)": "{:.2f}%"
            }).set_properties(**{
                "color": "#1a202c",
                "font-family": "JetBrains Mono, monospace",
                "font-size": "0.8rem",
            }).set_table_styles([
                {"selector": "th", "props": [
                    ("background-color", "#0d1320"),
                    ("color", "#8899AA"),
                    ("font-family", "JetBrains Mono, monospace"),
                    ("font-size", "0.7rem"),
                    ("text-transform", "uppercase"),
                    ("letter-spacing", "0.04em"),
                ]},
            ]),
            use_container_width=True
        )

        # Chart
        fig_port_shock = px.bar(
            port_df, x="Scenario", y="Avg Drawdown (%)",
            color="Avg Drawdown (%)",
            color_continuous_scale=["#FF4444", "#FFB800", "#00FF88"],
            title="AVERAGE PORTFOLIO DRAWDOWN BY SCENARIO",
        )
        fig_port_shock.update_layout(
            **plotly_layout(
                height=350,
            )
        )
        fig_port_shock.update_traces(
            text=port_df["Avg Drawdown (%)"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            textfont=dict(color="#C0CCD8", size=10, family="JetBrains Mono")
        )
        st.plotly_chart(fig_port_shock, use_container_width=True)

    # Recovery analysis
    st.markdown("### Recovery Analysis")
    recovery_cols = ["Ticker", "PRISM Score", "Growth", "SHOCK Score", "Edge Ratio", "Recovery Needed (%)", "1Y Recovery"]
    recovery_cols = [c for c in recovery_cols if c in scored_df.columns]
    recovery_df = scored_df[recovery_cols].copy()

    def color_recovery_table(val):
        if val == "YES":
            return "background-color: #003D22; color: #00FF88; font-weight: 600"
        elif val == "MAYBE":
            return "background-color: #3D2E00; color: #FFB800"
        elif val == "NO":
            return "background-color: #3D0000; color: #FF4444"
        return "color: #1a202c"

    st.dataframe(
        recovery_df.style.format({
            "PRISM Score": "{:.1f}",
            "Growth": "{:.1%}",
            "SHOCK Score": "{:.1f}%",
            "Edge Ratio": "{:.2f}",
            "Recovery Needed (%)": "{:.1f}%"
        }, na_rep="—").map(
            color_recovery_table, subset=["1Y Recovery"] if "1Y Recovery" in recovery_df.columns else []
        ).set_properties(**{
            "color": "#1a202c",
            "font-family": "JetBrains Mono, monospace",
            "font-size": "0.8rem",
        }).set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#0d1320"),
                ("color", "#8899AA"),
                ("font-family", "JetBrains Mono, monospace"),
                ("font-size", "0.7rem"),
                ("text-transform", "uppercase"),
                ("letter-spacing", "0.04em"),
            ]},
        ]),
        use_container_width=True
    )


# ========================= TAB 4: DATA QUALITY =============================
with tab_quality:
    st.markdown("### Data Quality Scorecard")
    st.caption("Data source, growth rate, and confidence level per stock")

    quality_df = raw_df.attrs.get("quality_log")
    if quality_df is not None and not quality_df.empty:

        def color_confidence(val):
            val_str = str(val)
            if "HIGH" in val_str:
                return "background-color: #003D22; color: #00FF88; font-weight: 600"
            elif "MED" in val_str:
                return "background-color: #3D2E00; color: #FFB800"
            elif "LOW" in val_str or "DEFAULT" in val_str:
                return "background-color: #3D0000; color: #FF4444"
            elif "OVERRIDE" in val_str:
                return "background-color: #1a1040; color: #A78BFA"
            return "color: #1a202c"

        st.dataframe(
            quality_df.style.map(
                color_confidence, subset=["Used"]
            ).set_properties(**{
                "color": "#1a202c",
                "font-family": "JetBrains Mono, monospace",
                "font-size": "0.8rem",
            }).set_table_styles([
                {"selector": "th", "props": [
                    ("background-color", "#0d1320"),
                    ("color", "#8899AA"),
                    ("font-family", "JetBrains Mono, monospace"),
                    ("font-size", "0.7rem"),
                    ("text-transform", "uppercase"),
                    ("letter-spacing", "0.04em"),
                ]},
            ]),
            use_container_width=True
        )

        # Summary
        conf_counts = quality_df["Used"].apply(
            lambda x: "HIGH" if "HIGH" in str(x) else "MED" if "MED" in str(x) else "LOW" if "LOW" in str(x) else "DEFAULT" if "DEFAULT" in str(x) else "OVERRIDE"
        ).value_counts()

        st.markdown(f"""
        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; margin-top: 1rem;">
            <div class="kpi-card">
                <div class="kpi-label">HIGH CONFIDENCE</div>
                <div class="kpi-value green">{conf_counts.get("HIGH", 0)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">MEDIUM</div>
                <div class="kpi-value amber">{conf_counts.get("MED", 0)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">LOW / DEFAULT</div>
                <div class="kpi-value red">{conf_counts.get("LOW", 0) + conf_counts.get("DEFAULT", 0)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">OVERRIDES</div>
                <div class="kpi-value" style="color: #A78BFA">{conf_counts.get("OVERRIDE", 0)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("Quality data not available. Try refreshing.")

    # Growth overrides section
    st.markdown("### Growth Rate Overrides")
    st.caption("Manually set growth rates for stocks with unreliable data")

    override_ticker = st.selectbox("Ticker to override", active_tickers, key="override_ticker")
    override_val = st.number_input("Growth rate (%)", value=10.0, step=1.0, key="override_val")

    if st.button("SET OVERRIDE", key="btn_override"):
        watchlist["growth_overrides"][override_ticker] = override_val / 100
        with open(CONFIG_DIR / "watchlist.json", "w") as f:
            json.dump(watchlist, f, indent=2)
        st.success(f"Set {override_ticker} growth to {override_val}%")
        st.cache_data.clear()
        st.rerun()

    if watchlist.get("growth_overrides"):
        st.markdown("**Active Overrides:**")
        for tk, g in watchlist["growth_overrides"].items():
            st.markdown(f"- `{tk}`: {g:.1%}")


# ---------------------------------------------------------------------------
#  Footer
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="prism-footer">
    PRISM v3.0 &nbsp;|&nbsp; DATA: FMP + FINNHUB + YFINANCE &nbsp;|&nbsp; {len(active_tickers)} STOCKS &nbsp;|&nbsp; {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>
""", unsafe_allow_html=True)
