"""
PRISM v3 — Quantitative Portfolio Intelligence Dashboard
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

# API keys from environment (Render secrets) or fallback to config
import os
FMP_KEY = os.environ.get("FMP_API_KEY", settings["api_keys"]["fmp"])
FH_KEY = os.environ.get("FINNHUB_API_KEY", settings["api_keys"]["finnhub"])

# ---------------------------------------------------------------------------
#  Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main container */
    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }

    /* Header styling */
    .prism-header {
        display: flex; align-items: center; gap: 0.75rem;
        padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #2A2F3E;
        margin-bottom: 1.5rem;
    }
    .prism-logo {
        font-size: 2rem; font-weight: 800; letter-spacing: -0.03em;
        color: #00C896;
    }
    .prism-sub { color: #8892A4; font-size: 0.85rem; margin-top: 0.25rem; }

    /* Score badges */
    .score-badge {
        display: inline-block; padding: 0.15rem 0.5rem;
        border-radius: 4px; font-weight: 600; font-size: 0.8rem;
    }
    .score-high { background: #00C89620; color: #00C896; }
    .score-mid { background: #F59E0B20; color: #F59E0B; }
    .score-low { background: #EF444420; color: #EF4444; }

    /* Metric cards */
    .metric-card {
        background: #1A1F2E; border-radius: 8px; padding: 1rem;
        border: 1px solid #2A2F3E;
    }
    .metric-label { color: #8892A4; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #E0E0E0; }

    /* Table tweaks */
    .stDataFrame { font-size: 0.85rem; }

    /* KPI row */
    div[data-testid="stMetric"] {
        background: #1A1F2E; border: 1px solid #2A2F3E;
        border-radius: 8px; padding: 0.75rem 1rem;
    }
    div[data-testid="stMetric"] label { color: #8892A4 !important; font-size: 0.75rem !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #0B0E14; }
    section[data-testid="stSidebar"] .stMarkdown h1 { font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
#  Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="prism-header">
    <div>
        <div class="prism-logo">◈ PRISM v3</div>
        <div class="prism-sub">Portfolio Risk & Intelligence Scoring Model</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
#  Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

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
            # Persist
            with open(CONFIG_DIR / "watchlist.json", "w") as f:
                json.dump(watchlist, f, indent=2)
            st.cache_data.clear()
            st.rerun()
        else:
            st.warning(f"{sym} already in watchlist")

    # Active tickers (toggle)
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
    st.caption(f"Normalized: {w_fund/total_w:.0%} / {w_risk/total_w:.0%} / {w_tech/total_w:.0%}" if total_w > 0 else "")

    st.markdown("---")

    # Index benchmarks
    st.markdown("#### Benchmark")
    idx_pe = st.number_input("Index PE", value=22.0, step=0.5, key="idx_pe")
    idx_growth = st.number_input("Index Growth (%)", value=13.0, step=0.5, key="idx_growth")

    st.markdown("---")

    # Refresh
    if st.button("🔄 Refresh Data", use_container_width=True, key="btn_refresh"):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S HKT')}")


# ---------------------------------------------------------------------------
#  Data Fetch & Scoring
# ---------------------------------------------------------------------------
if not active_tickers:
    st.warning("Select at least one ticker in the sidebar.")
    st.stop()

with st.spinner("Fetching live data & computing scores..."):
    raw_df = fetch_all_data(
        tickers=active_tickers,
        fmp_key=FMP_KEY,
        fh_key=FH_KEY,
        growth_overrides=watchlist.get("growth_overrides", {}),
        default_growth=settings["portfolio"]["default_growth_rate"]
    )

    scored_df = run_scoring(raw_df, settings)


# ---------------------------------------------------------------------------
#  Tab Layout
# ---------------------------------------------------------------------------
tab_watchlist, tab_detail, tab_stress, tab_quality = st.tabs([
    "📊 Watchlist", "🔍 Stock Detail", "⚡ Stress Test", "📋 Data Quality"
])


# ========================= TAB 1: WATCHLIST ================================
with tab_watchlist:

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Stocks", len(scored_df))
    with col2:
        avg_prism = scored_df["PRISM Score"].mean()
        st.metric("Avg PRISM Score", f"{avg_prism:.1f}")
    with col3:
        avg_shock = scored_df["SHOCK Score"].mean()
        st.metric("Avg SHOCK", f"{avg_shock:.1f}%")
    with col4:
        avg_edge = scored_df["Edge Ratio"].mean()
        st.metric("Avg Edge Ratio", f"{avg_edge:.2f}")
    with col5:
        buy_zone_count = len(scored_df[scored_df["52w Position"] < 0.3])
        st.metric("In Buy Zone", buy_zone_count)

    st.markdown("")

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

    # Format columns
    format_map = {
        "Price": "${:.2f}",
        "PE (Fwd)": "{:.1f}x",
        "Growth": "{:.1%}",
        "Relative PEG": "{:.2f}",
        "Alpha Score": "{:.3f}",
        "PRISM Score": "{:.1f}",
        "Fundamental Score": "{:.0f}",
        "Risk Score": "{:.0f}",
        "technical_score": "{:.0f}",
        "SHOCK Score": "{:.1f}%",
        "Edge Ratio": "{:.2f}",
        "52w Position": "{:.0%}",
        "rsi_14": "{:.0f}",
    }

    # Color coding function
    def color_prism(val):
        if isinstance(val, (int, float)):
            if val >= 65:
                return "background-color: #00C89620; color: #00C896"
            elif val >= 50:
                return "background-color: #F59E0B20; color: #F59E0B"
            else:
                return "background-color: #EF444420; color: #EF4444"
        return ""

    def color_shock(val):
        try:
            v = float(str(val).replace("%", ""))
            if v > -8:
                return "background-color: #00C89620"
            elif v > -15:
                return "background-color: #F59E0B20"
            else:
                return "background-color: #EF444420"
        except (ValueError, TypeError):
            return ""

    def color_52w(val):
        try:
            v = float(str(val).replace("%", "")) / 100 if "%" in str(val) else float(val)
            if v < 0.3:
                return "background-color: #00C89620"  # buy zone
            elif v > 0.9:
                return "background-color: #EF444420"  # near highs
            return ""
        except (ValueError, TypeError):
            return ""

    # Rename for cleaner display
    rename = {"technical_score": "Tech Score", "rsi_14": "RSI"}
    display_df = display_df.rename(columns=rename)

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
    }, na_rep="—").map(
        color_prism, subset=["PRISM Score"] if "PRISM Score" in display_df.columns else []
    ).map(
        color_shock, subset=["SHOCK Score"] if "SHOCK Score" in display_df.columns else []
    ).map(
        color_52w, subset=["52w Position"] if "52w Position" in display_df.columns else []
    )

    st.dataframe(styled, use_container_width=True, height=min(600, 45 * len(display_df) + 40))

    # --- Charts ---
    st.markdown("### Score Distribution")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        # PRISM Score bar chart
        fig_prism = px.bar(
            scored_df.sort_values("PRISM Score", ascending=True),
            x="PRISM Score", y="Ticker", orientation="h",
            color="PRISM Score",
            color_continuous_scale=["#EF4444", "#F59E0B", "#00C896"],
            title="PRISM Score by Stock"
        )
        fig_prism.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            height=max(300, 35 * len(scored_df)),
            showlegend=False, coloraxis_showscale=False,
            margin=dict(l=0, r=20, t=40, b=0),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_prism, use_container_width=True)

    with col_chart2:
        # Risk vs Return scatter
        fig_scatter = px.scatter(
            scored_df,
            x="SHOCK Score", y="Growth",
            size="Market Cap ($B)" if "Market Cap ($B)" in scored_df.columns else None,
            color="PRISM Score",
            color_continuous_scale=["#EF4444", "#F59E0B", "#00C896"],
            hover_name="Ticker",
            title="Risk vs Growth (Bubble = Market Cap)",
            labels={"SHOCK Score": "SHOCK Score (%)", "Growth": "Expected Growth"}
        )
        fig_scatter.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            height=max(300, 35 * len(scored_df)),
            coloraxis_showscale=False,
            margin=dict(l=0, r=20, t=40, b=0)
        )
        fig_scatter.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_scatter, use_container_width=True)


# ========================= TAB 2: STOCK DETAIL =============================
with tab_detail:
    selected_ticker = st.selectbox("Select stock", active_tickers, key="detail_ticker")
    row = scored_df[scored_df["Ticker"] == selected_ticker]

    if row.empty:
        st.warning("No data for selected ticker.")
    else:
        row = row.iloc[0]

        # Header
        col_h1, col_h2 = st.columns([2, 1])
        with col_h1:
            st.markdown(f"## {selected_ticker}")
            thesis = watchlist["tickers"].get(selected_ticker, {}).get("thesis", "")
            if thesis:
                st.caption(f"Thesis: {thesis}")
        with col_h2:
            prism = row.get("PRISM Score", 0)
            badge_class = "score-high" if prism >= 65 else "score-mid" if prism >= 50 else "score-low"
            st.markdown(f'<div style="text-align:right"><span class="score-badge {badge_class}">PRISM {prism:.1f}</span> &nbsp; Rank #{int(row.get("Rank", 0))}</div>', unsafe_allow_html=True)

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
            marker_color="#2A2F3E", text=scores_df["Score"].apply(lambda x: f"{x:.0f}"),
            textposition="outside"
        ))
        fig_breakdown.add_trace(go.Bar(
            name="Weighted", x=scores_df["Dimension"], y=scores_df["Weighted"],
            marker_color="#00C896", text=scores_df["Weighted"].apply(lambda x: f"{x:.1f}"),
            textposition="outside"
        ))
        fig_breakdown.update_layout(
            template="plotly_dark", barmode="group",
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            height=300, margin=dict(l=0, r=0, t=20, b=0),
            yaxis=dict(range=[0, 110])
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
                color_continuous_scale=["#EF4444", "#F59E0B", "#00C896"],
                title=f"Expected Drawdown by Scenario — {selected_ticker}",
                text=shock_df["Drawdown"].apply(lambda x: f"{x:.1f}%")
            )
            fig_shock.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                height=300, showlegend=False, coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig_shock, use_container_width=True)

        # Fragile flags
        flags = row.get("Fragile Flags", "—")
        if flags and flags != "—":
            st.warning(f"Fragile Flags: {flags}")

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
    st.caption("Showing aggregate SHOCK impact across your active watchlist")

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
            }),
            use_container_width=True
        )

        # Chart
        fig_port_shock = px.bar(
            port_df, x="Scenario", y="Avg Drawdown (%)",
            color="Avg Drawdown (%)",
            color_continuous_scale=["#EF4444", "#F59E0B", "#00C896"],
            title="Average Portfolio Drawdown by Scenario",
            text=port_df["Avg Drawdown (%)"].apply(lambda x: f"{x:.1f}%")
        )
        fig_port_shock.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            height=350, showlegend=False, coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_port_shock, use_container_width=True)

    # Recovery analysis
    st.markdown("### Recovery Analysis")
    recovery_cols = ["Ticker", "PRISM Score", "Growth", "SHOCK Score", "Edge Ratio", "Recovery Needed (%)", "1Y Recovery"]
    recovery_cols = [c for c in recovery_cols if c in scored_df.columns]
    recovery_df = scored_df[recovery_cols].copy()

    def color_recovery(val):
        if val == "YES":
            return "background-color: #00C89620; color: #00C896"
        elif val == "MAYBE":
            return "background-color: #F59E0B20; color: #F59E0B"
        elif val == "NO":
            return "background-color: #EF444420; color: #EF4444"
        return ""

    st.dataframe(
        recovery_df.style.format({
            "PRISM Score": "{:.1f}",
            "Growth": "{:.1%}",
            "SHOCK Score": "{:.1f}%",
            "Edge Ratio": "{:.2f}",
            "Recovery Needed (%)": "{:.1f}%"
        }, na_rep="—").map(color_recovery, subset=["1Y Recovery"] if "1Y Recovery" in recovery_df.columns else []),
        use_container_width=True
    )


# ========================= TAB 4: DATA QUALITY =============================
with tab_quality:
    st.markdown("### Data Quality Scorecard")
    st.caption("Shows which data source was used for each stock's growth rate and confidence level")

    quality_df = raw_df.attrs.get("quality_log")
    if quality_df is not None and not quality_df.empty:

        def color_confidence(val):
            val_str = str(val)
            if "HIGH" in val_str:
                return "background-color: #00C89620; color: #00C896"
            elif "MED" in val_str:
                return "background-color: #F59E0B20; color: #F59E0B"
            elif "LOW" in val_str or "DEFAULT" in val_str:
                return "background-color: #EF444420; color: #EF4444"
            elif "OVERRIDE" in val_str:
                return "background-color: #8B5CF620; color: #8B5CF6"
            return ""

        st.dataframe(
            quality_df.style.map(color_confidence, subset=["Used"]),
            use_container_width=True
        )

        # Summary
        conf_counts = quality_df["Used"].apply(
            lambda x: "HIGH" if "HIGH" in str(x) else "MED" if "MED" in str(x) else "LOW" if "LOW" in str(x) else "DEFAULT" if "DEFAULT" in str(x) else "OVERRIDE"
        ).value_counts()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("✅ High Confidence", conf_counts.get("HIGH", 0))
        with c2:
            st.metric("⚠️ Medium", conf_counts.get("MED", 0))
        with c3:
            st.metric("🚩 Low / Default", conf_counts.get("LOW", 0) + conf_counts.get("DEFAULT", 0))
        with c4:
            st.metric("🔧 Overrides", conf_counts.get("OVERRIDE", 0))

    else:
        st.info("Quality data not available. Try refreshing.")

    # Growth overrides section
    st.markdown("### Growth Rate Overrides")
    st.caption("Manually set growth rates for stocks with unreliable data")

    override_ticker = st.selectbox("Ticker to override", active_tickers, key="override_ticker")
    override_val = st.number_input("Growth rate (%)", value=10.0, step=1.0, key="override_val")

    if st.button("Set Override", key="btn_override"):
        watchlist["growth_overrides"][override_ticker] = override_val / 100
        with open(CONFIG_DIR / "watchlist.json", "w") as f:
            json.dump(watchlist, f, indent=2)
        st.success(f"Set {override_ticker} growth to {override_val}%")
        st.cache_data.clear()
        st.rerun()

    if watchlist.get("growth_overrides"):
        st.markdown("**Active Overrides:**")
        for tk, g in watchlist["growth_overrides"].items():
            st.markdown(f"- {tk}: {g:.1%}")


# ---------------------------------------------------------------------------
#  Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(f"PRISM v3.0 — Data: FMP + Finnhub + yfinance | {len(active_tickers)} stocks | {datetime.now().strftime('%Y-%m-%d %H:%M HKT')}")
