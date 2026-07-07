"""Data fetch orchestration — Yahoo Finance is the single market-data source.

Growth-estimate confidence levels:
  YF        - from Yahoo's analyst growth estimates
  OVERRIDE  - manually pinned in config.yaml growth_overrides
  DEFAULT   - no estimate available; a conservative 4% is assumed
"""
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .yahoo import (fetch_yahoo_forward_eps, fetch_yahoo_growth,
                    fetch_yahoo_returns, fetch_yahoo_snapshot)

log = logging.getLogger("prism.fetch")

CONF_ICONS = {"YF": "📊Y", "DEFAULT": "❓D", "OVERRIDE": "🔧O"}


def _cache_paths(cache_dir):
    d = Path(cache_dir)
    stamp = pd.Timestamp.today().strftime("%Y-%m-%d")
    return d / f"fetch_{stamp}.parquet", d / f"quality_{stamp}.parquet"


def load_cached_fetch(cache_dir=".cache"):
    """Return (df, qdf) from today's cache, or None if absent."""
    df_path, q_path = _cache_paths(cache_dir)
    if df_path.exists() and q_path.exists():
        return pd.read_parquet(df_path), pd.read_parquet(q_path)
    return None


def save_fetch_cache(df, qdf, cache_dir=".cache"):
    df_path, q_path = _cache_paths(cache_dir)
    df_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(df_path, index=False)
    qdf.to_parquet(q_path, index=False)


def fetch_all_data(tickers, cfg, cache_dir=".cache", refresh=False,
                   progress_cb=None):
    """Fetch all inputs for the watchlist. Returns (df, quality_df).

    Uses a same-day disk cache unless refresh=True. progress_cb, if given,
    is called with (done, total, symbol) as each ticker completes.
    """
    if not refresh:
        cached = load_cached_fetch(cache_dir)
        if cached is not None:
            print(f"  Using cached fetch from {cache_dir}/ (pass --refresh to refetch)")
            return cached

    print(f"\n{'='*90}")
    print("  PRISM — Data Fetch (Yahoo Finance)")
    print(f"{'='*90}")
    print(f"  Tickers: {len(tickers)}\n")

    rows = []
    quality_log = []

    for i, sym in enumerate(tickers):
        if progress_cb:
            progress_cb(i + 1, len(tickers), sym)
        tk, yh = fetch_yahoo_snapshot(sym)
        price, mcap, beta = yh["price"], yh["mcap"], yh["beta"]
        high52, low52 = yh["high52"], yh["low52"]
        industry, target_mean = yh["industry"], yh["target_mean"]

        avg_outcome = ((target_mean / price) - 1) if (
            pd.notna(price) and pd.notna(target_mean) and price > 0) else 0.0

        r3m, r6m, r12m = fetch_yahoo_returns(tk, sym)

        # --- Growth estimates ---
        g2026, g2027 = fetch_yahoo_growth(tk, sym)
        c26 = "YF" if pd.notna(g2026) else "DEFAULT"
        c27 = "YF" if pd.notna(g2027) else "DEFAULT"

        if sym in cfg["growth_overrides"]:
            g2026, g2027 = cfg["growth_overrides"][sym]
            c26 = c27 = "OVERRIDE"

        if pd.isna(g2026):
            g2026 = 0.04
        if pd.isna(g2027):
            g2027 = 0.04

        # --- Forward P/E ---
        fwd_pe = np.nan
        yf_eps = fetch_yahoo_forward_eps(tk, sym)
        if pd.notna(yf_eps) and yf_eps > 0 and pd.notna(price):
            fwd_pe = price / yf_eps
        if pd.isna(fwd_pe):
            fwd_pe = yh.get("forward_pe_info", np.nan)

        quality_log.append({
            "Stock": sym, "G26_conf": c26, "G27_conf": c27,
            "Final_g26": g2026, "Final_g27": g2027})
        rows.append({
            "Stock": sym, "Price": price, "Market Cap": mcap, "Beta": beta,
            "52 Week High": high52, "52 Week Low": low52, "P/E.1": fwd_pe,
            "Industry": industry, "Target Mean": target_mean,
            "Average Outcome": avg_outcome,
            "2026 Growth Rate": g2026, "2027 Growth Rate": g2027,
            "Ret 3M": r3m, "Ret 6M": r6m, "Ret 12M": r12m})

        pe_s = f"{fwd_pe:>6.1f}" if pd.notna(fwd_pe) else "   N/A"
        b_s = f"{beta:>5.2f}" if pd.notna(beta) else "  N/A"
        print(f"  [{i+1:2d}/{len(tickers)}] {sym:>6} | G26:{g2026:>+6.0%} "
              f"[{CONF_ICONS.get(c26, '?'):>3}] | G27:{g2027:>+6.0%} "
              f"[{CONF_ICONS.get(c27, '?'):>3}] | PE:{pe_s} | B:{b_s}")
        time.sleep(0.3)

    df = pd.DataFrame(rows)
    qdf = pd.DataFrame(quality_log)
    print_quality_scorecard(qdf)
    save_fetch_cache(df, qdf, cache_dir)
    return df, qdf


def print_quality_scorecard(qdf):
    print(f"\n{'='*90}")
    print("  DATA QUALITY SCORECARD")
    print(f"{'='*90}")
    for label, col in [("2026 Growth", "G26_conf"), ("2027 Growth", "G27_conf")]:
        ct = qdf[col].value_counts()
        t = len(qdf)
        y, d, o = ct.get("YF", 0), ct.get("DEFAULT", 0), ct.get("OVERRIDE", 0)
        print(f"  {label}: 📊YF:{y} 🔧OVERRIDE:{o} ❓DEFAULT:{d} "
              f"| {(y + o) / t * 100:.0f}% covered")
    bad = qdf[(qdf["G26_conf"] == "DEFAULT") | (qdf["G27_conf"] == "DEFAULT")]
    if len(bad) > 0:
        print(f"\n  ⚠️  No growth estimate (4% assumed) — consider growth_overrides "
              f"in config.yaml: {bad['Stock'].tolist()}")
