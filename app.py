"""PRISM — Streamlit front end on the prism package engine.

Run:  streamlit run app.py
"""
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from prism.config import load_config
from prism.data.fetch import fetch_all_data
from prism.export import export_csvs
from prism.htmlreport import render_html_report
from prism.portfolio import build_portfolios
from prism.scanner import get_index_constituents, scan_index
from prism.scoring import run_prism
from prism.shock import SHOCK_SCENARIOS, run_shock

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.yaml"

st.set_page_config(page_title="PRISM", page_icon="◈", layout="wide",
                   initial_sidebar_state="expanded")

# --- Terminal-style theme (carried over from the v3 dashboard) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');
.stApp { background-color: #0a0e17; }
h1, h2, h3 { font-family: 'JetBrains Mono', monospace; color: #e8ecf3; }
.prism-header { border-bottom: 2px solid #FF6600; padding-bottom: .6rem; margin-bottom: 1rem; }
.prism-logo { font-family: 'JetBrains Mono', monospace; font-size: 1.7rem;
              font-weight: 700; color: #FF6600; letter-spacing: .05em; }
.prism-sub { color: #8899AA; font-family: 'JetBrains Mono', monospace;
             font-size: .7rem; letter-spacing: .08em; text-transform: uppercase; }
[data-testid="stSidebar"] { background-color: #0e1420; }
.stButton > button { background: #1a2233; color: #e8ecf3; border: 1px solid #FF6600; }
.stButton > button:hover { background: #FF6600; color: #0a0e17; }
</style>
<div class="prism-header">
  <div class="prism-logo">◈ PRISM</div>
  <div class="prism-sub">Portfolio Risk &amp; Intelligence Scoring Model</div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
#  Config round-trip
# ---------------------------------------------------------------------------

def read_raw_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def save_tickers(tickers):
    raw = read_raw_config()
    raw["tickers"] = tickers
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)


def run_pipeline(refresh):
    tickers, cfg = load_config(CONFIG_PATH)
    bar = st.progress(0.0, text="Fetching market data…")
    df, qdf = fetch_all_data(
        tickers, cfg, cache_dir=str(ROOT / ".cache"), refresh=refresh,
        progress_cb=lambda i, n, s: bar.progress(i / n, text=f"Fetching {s} ({i}/{n})"))
    bar.empty()
    df = run_prism(df, cfg)
    portfolios, fragile, reps = build_portfolios(df, cfg)
    shock = run_shock(df, portfolios)
    out_dir = export_csvs(df, portfolios, shock, qdf, cfg)
    html = render_html_report(df, portfolios, shock, qdf, cfg, fragile_stocks=fragile)
    (out_dir / "prism_report.html").write_text(html, encoding="utf-8")
    return {"df": df, "qdf": qdf, "portfolios": portfolios, "fragile": fragile,
            "shock": shock, "cfg": cfg, "html": html, "out_dir": out_dir}


# ---------------------------------------------------------------------------
#  Sidebar — watchlist editor + run
# ---------------------------------------------------------------------------

raw_cfg = read_raw_config()
with st.sidebar:
    st.markdown("### WATCHLIST")
    tickers_text = st.text_area(
        "One ticker per line", value="\n".join(raw_cfg["tickers"]),
        height=320, label_visibility="collapsed")
    edited = [t.strip().upper() for t in tickers_text.splitlines() if t.strip()]
    if edited != raw_cfg["tickers"]:
        if st.button("💾 SAVE WATCHLIST", use_container_width=True):
            save_tickers(edited)
            st.success(f"Saved {len(edited)} tickers to config.yaml")
            st.rerun()

    st.markdown("---")
    force_refresh = st.checkbox("Force refetch (ignore same-day cache)")
    if st.button("▶ RUN PRISM", type="primary", use_container_width=True):
        with st.spinner("Scoring…"):
            st.session_state["run"] = run_pipeline(force_refresh)

# ---------------------------------------------------------------------------
#  Tabs
# ---------------------------------------------------------------------------

tab_rank, tab_stress, tab_scan, tab_hist = st.tabs(
    ["RANKINGS", "STRESS TEST", "SCANNER", "HISTORY"])

run = st.session_state.get("run")

with tab_rank:
    if not run:
        st.info("Edit the watchlist in the sidebar and hit **RUN PRISM**.")
    else:
        df = run["df"]
        show = df[["Stock", "prism_rank", "prism_score", "dim_growth", "dim_value",
                   "dim_momentum", "dim_resilience", "2026 Growth Rate",
                   "2027 Growth Rate", "P/E.1", "peg_proxy", "Beta", "Ret 6M",
                   "52w_position", "Average Outcome", "shock_sector",
                   "fragile_flag"]].rename(columns={
                       "prism_rank": "Rank", "prism_score": "Score",
                       "dim_growth": "Growth", "dim_value": "Value",
                       "dim_momentum": "Momentum", "dim_resilience": "Resilience",
                       "P/E.1": "Fwd P/E", "peg_proxy": "PEG",
                       "52w_position": "52w Pos", "Average Outcome": "Upside",
                       "shock_sector": "Sector", "fragile_flag": "Fragile"})
        st.dataframe(
            show, hide_index=True, use_container_width=True, height=600,
            column_config={
                "Score": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
                "Growth": st.column_config.NumberColumn(format="%.0f"),
                "Value": st.column_config.NumberColumn(format="%.0f"),
                "Momentum": st.column_config.NumberColumn(format="%.0f"),
                "Resilience": st.column_config.NumberColumn(format="%.0f"),
                "2026 Growth Rate": st.column_config.NumberColumn("G26", format="percent"),
                "2027 Growth Rate": st.column_config.NumberColumn("G27", format="percent"),
                "Ret 6M": st.column_config.NumberColumn(format="percent"),
                "52w Pos": st.column_config.NumberColumn(format="percent"),
                "Upside": st.column_config.NumberColumn(format="percent"),
                "Fwd P/E": st.column_config.NumberColumn(format="%.1f"),
                "PEG": st.column_config.NumberColumn(format="%.2f"),
                "Beta": st.column_config.NumberColumn(format="%.2f"),
            })

        st.markdown("#### Portfolios")
        for pname, sl in run["portfolios"].items():
            st.markdown(f"**{pname}** ({len(sl)} × {100/len(sl):.1f}%): "
                        + " · ".join(sl))
        if run["fragile"]:
            st.warning(f"Fragile flags: {', '.join(run['fragile'])}")

        best = max(run["shock"], key=lambda p: run["shock"][p]["Edge Ratio"])
        b = run["shock"][best]
        st.success(f"**Recommendation: {best}** — Edge Ratio {b['Edge Ratio']:.2f}x, "
                   f"growth {b['Growth']:.0%}, expected shock {b['SHOCK Score']:+.1f}%")

        st.download_button("⬇ DOWNLOAD HTML REPORT", run["html"],
                           file_name="prism_report.html", mime="text/html")

        defaulted = run["qdf"][(run["qdf"]["G26_conf"] == "DEFAULT")
                               | (run["qdf"]["G27_conf"] == "DEFAULT")]["Stock"].tolist()
        if defaulted:
            st.caption(f"⚠️ No growth estimate for {', '.join(defaulted)} — 4% assumed. "
                       "Pin better numbers via growth_overrides in config.yaml.")

with tab_stress:
    if not run:
        st.info("Run PRISM first.")
    else:
        rows = []
        for pn, r in run["shock"].items():
            row = {"Portfolio": pn, **{sn: r[sn] for sn in SHOCK_SCENARIOS}}
            row["SHOCK Score"] = r["SHOCK Score"]
            row["Edge Ratio"] = r["Edge Ratio"]
            row["1Y Recovery"] = r["1Y Recovery"]
            rows.append(row)
        sdf = pd.DataFrame(rows).set_index("Portfolio")
        num_cols = list(SHOCK_SCENARIOS) + ["SHOCK Score"]
        st.dataframe(
            sdf.style.background_gradient(cmap="RdYlGn", subset=num_cols,
                                          vmin=-60, vmax=0)
               .format({c: "{:+.0f}%" for c in num_cols} | {"Edge Ratio": "{:.2f}x"}),
            use_container_width=True)
        st.caption("Cell values are probability-scenario expected drawdowns; "
                   "greener is more resilient. Edge Ratio = growth per unit of "
                   "expected drawdown.")

with tab_scan:
    st.markdown("#### Earnings-vs-price growth scanner")
    st.caption("Finds stocks whose price growth trails net-income growth over "
               "1/3/5 years (positive gap → potential swing entry). Yahoo's "
               "statements cover ~4 fiscal years, so 5Y gaps may be blank.")
    c1, c2, c3 = st.columns([2, 3, 1])
    index_name = c1.selectbox("Index", ["S&P 500", "Nasdaq 100"])
    try:
        constituents = get_index_constituents(index_name, cache_dir=str(ROOT / ".cache"))
        sectors = sorted(constituents["Sector"].dropna().unique())
        chosen = c2.multiselect("Sectors (empty = all)", sectors)
        limit = c3.number_input("Max stocks", 10, 600, 100, step=10)
        subset = constituents if not chosen else constituents[
            constituents["Sector"].isin(chosen)]
        st.caption(f"{len(subset)} constituents match — scanning the first {min(limit, len(subset))}.")
        if st.button(f"SCAN {index_name.upper()}", type="primary"):
            bar = st.progress(0.0)
            result = scan_index(
                subset.head(int(limit)),
                progress_cb=lambda i, n, s: bar.progress(i / n, text=f"{s} ({i}/{n})"))
            bar.empty()
            st.session_state["scan"] = result
    except RuntimeError as e:
        st.error(str(e))

    scan = st.session_state.get("scan")
    if scan is not None and not scan.empty:
        pct_cols = [c for c in scan.columns if "CAGR" in c or "Gap" in c]
        st.dataframe(
            scan, hide_index=True, use_container_width=True, height=520,
            column_config={c: st.column_config.NumberColumn(format="percent")
                           for c in pct_cols})
        st.download_button("⬇ DOWNLOAD SCAN CSV",
                           scan.to_csv(index=False), file_name="prism_scan.csv")

with tab_hist:
    hist_path = ROOT / "outputs" / "history.csv"
    if not hist_path.exists():
        st.info("No run history yet — history.csv appears after the first run.")
    else:
        hist = pd.read_csv(hist_path)
        st.markdown(f"#### Score history — {hist['Date'].nunique()} run day(s)")
        picks = st.multiselect("Stocks", sorted(hist["Stock"].unique()),
                               default=sorted(hist["Stock"].unique())[:6])
        if picks:
            pivot = (hist[hist["Stock"].isin(picks)]
                     .pivot_table(index="Date", columns="Stock", values="prism_score"))
            st.line_chart(pivot)
        st.markdown("#### Past runs")
        run_dirs = sorted([d for d in (ROOT / "outputs").iterdir() if d.is_dir()],
                          reverse=True)
        for d in run_dirs[:20]:
            report = d / "prism_report.html"
            cols = st.columns([1, 3])
            cols[0].markdown(f"**{d.name}**")
            if report.exists():
                cols[1].download_button(
                    "report.html", report.read_text(encoding="utf-8"),
                    file_name=f"prism_report_{d.name}.html", mime="text/html",
                    key=f"dl_{d.name}")
        chosen_day = st.selectbox("Inspect a run's scored watchlist",
                                  [d.name for d in run_dirs])
        wl = ROOT / "outputs" / chosen_day / "prism_scored_watchlist.csv"
        if wl.exists():
            st.dataframe(pd.read_csv(wl), hide_index=True,
                         use_container_width=True, height=400)
