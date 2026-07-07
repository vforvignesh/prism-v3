"""Index scanner: flags stocks whose price growth trails net-income growth.

For each index constituent, compares net-income CAGR vs price CAGR over
trailing 1/3/5-year windows. A positive gap (earnings outgrowing price)
suggests potential undervaluation for swing entries.

yfinance is the only data source. Its annual income statements cover ~4
fiscal years, so the 5Y window is often unavailable — the signal classifier
averages whichever windows exist.
"""
import io
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

log = logging.getLogger("prism.scanner")

INDEX_SOURCES = {
    "S&P 500": {
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "attrs": {"id": "constituents"},
        "columns": {"Symbol": "Symbol", "Security": "Security",
                    "GICS Sector": "Sector"},
    },
    "Nasdaq 100": {
        "url": "https://en.wikipedia.org/wiki/Nasdaq-100",
        "attrs": {"id": "constituents"},
        "columns": {"Symbol": "Symbol", "Ticker": "Symbol",
                    "Company": "Security", "GICS Sector": "Sector"},
    },
}

NET_INCOME_LABELS = ["Net Income", "Net Income Common Stockholders",
                     "Net Income From Continuing Operations"]


def get_index_constituents(index_name, cache_dir=".cache", max_age_days=7):
    """Return DataFrame(Symbol, Security, Sector) for an index.

    Scrapes Wikipedia; keeps a cached copy so scans work offline.
    """
    src = INDEX_SOURCES[index_name]
    cache_file = Path(cache_dir) / f"index_{index_name.replace(' ', '').replace('&', '')}.csv"

    if cache_file.exists():
        age_days = (time.time() - cache_file.stat().st_mtime) / 86400
        if age_days <= max_age_days:
            return pd.read_csv(cache_file)

    try:
        # Wikipedia 403s the default urllib user agent
        resp = requests.get(src["url"], timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"})
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text), attrs=src["attrs"])
        raw = tables[0]
        df = pd.DataFrame()
        for src_col, dst_col in src["columns"].items():
            if src_col in raw.columns and dst_col not in df.columns:
                df[dst_col] = raw[src_col]
        if "Sector" not in df.columns:
            df["Sector"] = "Unknown"
        # Yahoo uses '-' for class shares (BRK.B -> BRK-B)
        df["Symbol"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False)
        df = df.dropna(subset=["Symbol"]).reset_index(drop=True)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=False)
        return df
    except Exception as e:
        log.warning("%s constituents fetch failed: %s", index_name, e)
        if cache_file.exists():
            return pd.read_csv(cache_file)  # stale cache beats nothing
        raise RuntimeError(
            f"Could not fetch {index_name} constituents (offline?) and no "
            f"cached copy exists at {cache_file}") from e


def cagr(start_val, end_val, years):
    """Compound annual growth rate; None for invalid/negative inputs."""
    if start_val is None or end_val is None or years <= 0:
        return None
    if np.isnan(start_val) or np.isnan(end_val):
        return None
    if start_val <= 0 or end_val <= 0:
        return None
    try:
        return (end_val / start_val) ** (1.0 / years) - 1.0
    except (ZeroDivisionError, ValueError, OverflowError):
        return None


def fetch_net_income_by_year(tk):
    """{calendar_year: net_income} from yfinance annual income statement."""
    try:
        fin = tk.income_stmt
        if fin is None or fin.empty:
            return None
        ni_row = None
        for label in NET_INCOME_LABELS:
            if label in fin.index:
                ni_row = fin.loc[label]
                break
        if ni_row is None:
            return None
        result = {}
        for date, val in ni_row.items():
            year = date.year if hasattr(date, "year") else int(str(date)[:4])
            if val is not None and not pd.isna(val):
                result[year] = float(val)
        return result or None
    except Exception:
        return None


def fetch_prices_by_year(tk):
    """{calendar_year: last close of that year} over the trailing 6 years."""
    try:
        hist = tk.history(period="6y", auto_adjust=True)
        if hist is None or hist.empty:
            return None
        closes = hist["Close"].dropna()
        result = {}
        for year in closes.index.year.unique():
            result[int(year)] = float(closes[closes.index.year == year].iloc[-1])
        # current year entry = most recent close
        result[datetime.now().year] = float(closes.iloc[-1])
        return result or None
    except Exception:
        return None


def compute_growths(ni_data, price_data, current_year, periods=(1, 3, 5)):
    """CAGRs and gaps (NI minus price growth) for each trailing period."""
    result = {}
    for period in periods:
        base_year = current_year - period

        ni_start = ni_data.get(base_year) if ni_data else None
        ni_end = None
        for y in (current_year, current_year - 1):
            if ni_data and y in ni_data:
                ni_end = ni_data[y]
                break
        ni_cagr = cagr(ni_start, ni_end, period)
        result[f"ni_cagr_{period}y"] = ni_cagr

        price_start = price_data.get(base_year) if price_data else None
        price_end = price_data.get(current_year) if price_data else None
        price_cagr = cagr(price_start, price_end, period)
        result[f"price_cagr_{period}y"] = price_cagr

        if ni_cagr is not None and price_cagr is not None:
            result[f"gap_{period}y"] = ni_cagr - price_cagr
        else:
            result[f"gap_{period}y"] = None
    return result


def classify_signal(growths):
    """UNDERVALUED .. OVERVALUED by average gap across available periods."""
    valid = [v for k, v in growths.items() if k.startswith("gap_") and v is not None]
    if not valid:
        return "NO DATA"
    avg_gap = float(np.mean(valid))
    if avg_gap > 0.10:
        return "UNDERVALUED"
    elif avg_gap > 0.02:
        return "MILD UNDER"
    elif avg_gap > -0.02:
        return "FAIR"
    elif avg_gap > -0.10:
        return "MILD OVER"
    else:
        return "OVERVALUED"


def scan_index(tickers_df, sectors=None, progress_cb=None, sleep=0.15):
    """Scan index constituents for earnings-vs-price growth gaps.

    Args:
        tickers_df: DataFrame(Symbol, Security, Sector)
        sectors: optional sector-name filter
        progress_cb: optional fn(done, total, symbol) for UI progress

    Returns a DataFrame sorted by average gap (most undervalued first).
    """
    df = tickers_df.copy()
    if sectors:
        df = df[df["Sector"].isin(sectors)].reset_index(drop=True)

    current_year = datetime.now().year
    results = []
    total = len(df)

    for idx, row in df.iterrows():
        symbol = row["Symbol"]
        if progress_cb:
            progress_cb(idx + 1, total, symbol)

        tk = yf.Ticker(symbol)
        ni_data = fetch_net_income_by_year(tk)
        price_data = fetch_prices_by_year(tk)
        try:
            info = tk.info or {}
            price = info.get("currentPrice") or info.get("regularMarketPrice") or np.nan
            mcap = info.get("marketCap") or np.nan
        except Exception:
            price = mcap = np.nan

        growths = compute_growths(ni_data, price_data, current_year)
        record = {
            "Symbol": symbol,
            "Security": row.get("Security", ""),
            "Sector": row.get("Sector", ""),
            "Price": price,
            "Mkt Cap ($B)": round(mcap / 1e9, 1) if pd.notna(mcap) else np.nan,
        }
        for period in (1, 3, 5):
            record[f"NI CAGR {period}Y"] = growths.get(f"ni_cagr_{period}y")
            record[f"Price CAGR {period}Y"] = growths.get(f"price_cagr_{period}y")
            record[f"Gap {period}Y"] = growths.get(f"gap_{period}y")
        record["Signal"] = classify_signal(growths)

        # Turnaround: NI negative in the base period, positive now
        if ni_data:
            recent = ni_data.get(current_year, ni_data.get(current_year - 1))
            oldest = ni_data.get(current_year - 5, ni_data.get(current_year - 3))
            if oldest is not None and recent is not None and oldest < 0 < recent:
                record["Signal"] = "TURNAROUND"

        results.append(record)
        time.sleep(sleep)

    out = pd.DataFrame(results)
    if not out.empty:
        gap_cols = [c for c in out.columns if c.startswith("Gap ")]
        out["Avg Gap"] = out[gap_cols].mean(axis=1, skipna=True)
        out = out.sort_values("Avg Gap", ascending=False).reset_index(drop=True)
    return out
