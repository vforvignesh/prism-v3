"""Multi-source fetch orchestration: FMP (primary) + Finnhub (validation) + yfinance."""
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .common import validate_growth
from .finnhub import calc_growth_from_finnhub, fetch_finnhub_estimates
from .fmp import calc_growth_from_fmp, fetch_fmp_estimates
from .yahoo import (fetch_yahoo_forward_eps, fetch_yahoo_growth,
                    fetch_yahoo_returns, fetch_yahoo_snapshot)

log = logging.getLogger("prism.fetch")

CONF_ICONS = {"HIGH": "✅H", "MED": "⚠M", "LOW": "🚩L", "YF": "📊Y",
              "DEFAULT": "❓D", "OVERRIDE": "🔧O", "NONE": "❓?"}


def make_session():
    """HTTP session with retry/backoff on transient errors and rate limits."""
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1.0,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


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


def fetch_all_data(tickers, cfg, fmp_key, fh_key, cache_dir=".cache", refresh=False):
    """Fetch all inputs for the watchlist. Returns (df, quality_df).

    Uses a same-day disk cache unless refresh=True.
    """
    if not refresh:
        cached = load_cached_fetch(cache_dir)
        if cached is not None:
            print(f"  Using cached fetch from {cache_dir}/ (pass --refresh to refetch)")
            return cached

    print(f"\n{'='*90}")
    print("  PRISM — Multi-Source Data Fetch")
    print(f"{'='*90}")
    print(f"  Tickers: {len(tickers)} | Sources: FMP (stable) + Finnhub + yfinance\n")

    session = make_session()
    rows = []
    quality_log = []

    # FMP gates some symbols per-tier, so it stays enabled unless the key
    # itself is rejected. Finnhub's eps-estimate endpoint is all-or-nothing.
    fmp_enabled = bool(fmp_key)
    fh_enabled = bool(fh_key)
    if not fmp_enabled:
        print("  ⚠️  No FMP_API_KEY set — FMP disabled")
    if not fh_enabled:
        print("  ⚠️  No FINNHUB_API_KEY set — Finnhub disabled")
    fmp_gated = []
    fmp_ok_count = 0
    fh_ok_count = 0

    for i, sym in enumerate(tickers):
        # --- yfinance: price, beta, market cap, targets ---
        tk, yh = fetch_yahoo_snapshot(sym)
        price, mcap, beta = yh["price"], yh["mcap"], yh["beta"]
        high52, low52 = yh["high52"], yh["low52"]
        industry, target_mean = yh["industry"], yh["target_mean"]

        avg_outcome = ((target_mean / price) - 1) if (
            pd.notna(price) and pd.notna(target_mean) and price > 0) else 0.0

        r3m, r6m, r12m = fetch_yahoo_returns(tk, sym)

        # --- FMP: analyst estimates (primary growth source) ---
        fmp_g26 = fmp_g27 = fmp_fwd_eps = np.nan
        if fmp_enabled:
            fmp_data, fmp_status = fetch_fmp_estimates(sym, fmp_key, session=session)
            if fmp_status == "ok":
                fmp_g26, fmp_g27, fmp_fwd_eps, _ = calc_growth_from_fmp(fmp_data)
                fmp_ok_count += 1
                if fmp_ok_count == 1:
                    print(f"  ✅ FMP connected — got {len(fmp_data)} entries for {sym}")
            elif fmp_status == "gated":
                fmp_gated.append(sym)
            elif fmp_status == "auth":
                print("  ❌ FMP rejected the API key — disabling FMP")
                fmp_enabled = False

        # --- Finnhub: EPS estimates (validation) ---
        fh_g26 = fh_g27 = np.nan
        if fh_enabled:
            fh_data, fh_status = fetch_finnhub_estimates(sym, fh_key, session=session)
            if fh_status == "ok":
                fh_g26, fh_g27 = calc_growth_from_finnhub(fh_data)
                fh_ok_count += 1
                if fh_ok_count == 1:
                    print(f"  ✅ Finnhub connected — got {len(fh_data)} entries for {sym}")
            elif fh_status == "auth":
                print("  ❌ Finnhub eps-estimate not available on this key "
                      "(premium endpoint) — disabling Finnhub")
                fh_enabled = False

        # --- Cross-validate FMP vs Finnhub ---
        g2026, c26 = validate_growth(fmp_g26, fh_g26)
        g2027, c27 = validate_growth(fmp_g27, fh_g27)

        # --- yfinance fallback for growth ---
        if pd.isna(g2026) or pd.isna(g2027):
            yf_cy, yf_ny = fetch_yahoo_growth(tk, sym)
            if pd.isna(g2026) and pd.notna(yf_cy):
                g2026, c26 = yf_cy, "YF"
            if pd.isna(g2027) and pd.notna(yf_ny):
                g2027, c27 = yf_ny, "YF"

        # --- Manual overrides ---
        if sym in cfg["growth_overrides"]:
            g2026, g2027 = cfg["growth_overrides"][sym]
            c26 = c27 = "OVERRIDE"

        # --- Default missing ---
        if pd.isna(g2026):
            g2026, c26 = 0.04, "DEFAULT"
        if pd.isna(g2027):
            g2027, c27 = 0.04, "DEFAULT"

        # --- Forward P/E: FMP first, then yfinance ---
        fwd_pe = np.nan
        if pd.notna(fmp_fwd_eps) and fmp_fwd_eps > 0 and pd.notna(price) and price > 0:
            fwd_pe = price / fmp_fwd_eps
            # FMP reports some foreign listings' EPS in local currency while the
            # price is the USD ADR (e.g. TSM in TWD) — an absurd P/E means a
            # currency mismatch, so discard it and fall back to yfinance.
            if fwd_pe < 3:
                log.warning("%s: FMP-implied P/E %.2f looks like a currency mismatch"
                            " — falling back to yfinance", sym, fwd_pe)
                fwd_pe = np.nan
        if pd.isna(fwd_pe):
            yf_eps = fetch_yahoo_forward_eps(tk, sym)
            if pd.notna(yf_eps) and yf_eps > 0 and pd.notna(price):
                fwd_pe = price / yf_eps
            if pd.isna(fwd_pe):
                fwd_pe = yh.get("forward_pe_info", np.nan)

        quality_log.append({
            "Stock": sym, "G26_conf": c26, "G27_conf": c27,
            "FMP_g26": fmp_g26, "FH_g26": fh_g26, "Final_g26": g2026,
            "FMP_g27": fmp_g27, "FH_g27": fh_g27, "Final_g27": g2027})
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
    print_quality_scorecard(qdf, fmp_enabled, fh_enabled, fmp_gated)
    save_fetch_cache(df, qdf, cache_dir)
    return df, qdf


def print_quality_scorecard(qdf, fmp_enabled, fh_enabled, fmp_gated=()):
    print(f"\n{'='*90}")
    print("  DATA QUALITY SCORECARD")
    print(f"{'='*90}")
    print(f"  Sources active: FMP={'YES' if fmp_enabled else 'NO (disabled)'} | "
          f"Finnhub={'YES' if fh_enabled else 'NO (disabled)'} | yfinance=YES")
    for label, col in [("2026 Growth", "G26_conf"), ("2027 Growth", "G27_conf")]:
        ct = qdf[col].value_counts()
        t = len(qdf)
        h, m, l = ct.get("HIGH", 0), ct.get("MED", 0), ct.get("LOW", 0)
        y, d, o = ct.get("YF", 0), ct.get("DEFAULT", 0), ct.get("OVERRIDE", 0)
        pct = (h + m + o) / t * 100
        print(f"  {label}: ✅HIGH:{h} ⚠M:{m} 🚩LOW:{l} 📊YF:{y} ❓DEFAULT:{d} 🔧OVR:{o} "
              f"| {pct:.0f}% reliable")
    bad = qdf[(qdf["G26_conf"].isin(["LOW", "DEFAULT", "NONE"]))
              | (qdf["G27_conf"].isin(["LOW", "DEFAULT", "NONE"]))]
    if len(bad) > 0:
        print(f"\n  ⚠️  Stocks needing overrides: {bad['Stock'].tolist()}")
    if fmp_gated:
        print(f"\n  💰 FMP free tier gated {len(fmp_gated)} symbols (fell back to yfinance): "
              f"{list(fmp_gated)}")

    # Loud gate: single-source runs are a materially different quality regime.
    if not fmp_enabled and not fh_enabled:
        print("\n  " + "!" * 86)
        print("  !!  WARNING: BOTH estimate sources are down — every growth number below")
        print("  !!  comes from yfinance alone. Cross-validation did NOT happen.")
        print("  " + "!" * 86)
