"""
PRISM v3 — Net Income vs Price Growth Discrepancy Scanner
Multi-source pipeline: yfinance → FMP → Alpha Vantage (with fallback)
"""

import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _safe(val, default=np.nan):
    if val is None:
        return default
    try:
        if np.isnan(val):
            return default
    except (TypeError, ValueError):
        pass
    return val


def _cagr(start_val: float, end_val: float, years: int) -> float | None:
    """Compound annual growth rate. Returns None for invalid inputs."""
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


# ---------------------------------------------------------------------------
#  Net Income — yfinance
# ---------------------------------------------------------------------------

def _fetch_net_income_yf(symbol: str) -> dict | None:
    """Return {year: net_income} from yfinance annual financials."""
    try:
        tk = yf.Ticker(symbol)
        fin = tk.financials  # columns = dates, rows = line items
        if fin is None or fin.empty:
            return None
        # Find net income row
        ni_row = None
        for label in ["Net Income", "Net Income Common Stockholders",
                       "Net Income From Continuing Operations"]:
            if label in fin.index:
                ni_row = fin.loc[label]
                break
        if ni_row is None:
            return None
        result = {}
        for date, val in ni_row.items():
            year = date.year if hasattr(date, "year") else int(str(date)[:4])
            v = _safe(val)
            if not np.isnan(v):
                result[year] = v
        return result if result else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  Net Income — FMP
# ---------------------------------------------------------------------------

def _fetch_net_income_fmp(symbol: str, api_key: str) -> dict | None:
    """Return {year: net_income} from FMP income statement."""
    if not api_key or api_key in ("FMP_API_KEY", ""):
        return None
    url = (f"https://financialmodelingprep.com/api/v3/income-statement/"
           f"{symbol}?period=annual&limit=6&apikey={api_key}")
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data or not isinstance(data, list):
            return None
        result = {}
        for item in data:
            year = int(str(item.get("calendarYear", item.get("date", "")[:4])))
            ni = item.get("netIncome")
            if ni is not None:
                result[year] = float(ni)
        return result if result else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  Net Income — Alpha Vantage
# ---------------------------------------------------------------------------

def _fetch_net_income_av(symbol: str, api_key: str) -> dict | None:
    """Return {year: net_income} from Alpha Vantage INCOME_STATEMENT."""
    if not api_key or api_key in ("ALPHA_VANTAGE_API_KEY", ""):
        return None
    url = (f"https://www.alphavantage.co/query?function=INCOME_STATEMENT"
           f"&symbol={symbol}&apikey={api_key}")
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        reports = data.get("annualReports", [])
        if not reports:
            return None
        result = {}
        for item in reports:
            year = int(item["fiscalDateEnding"][:4])
            ni = item.get("netIncome")
            if ni and ni != "None":
                result[year] = float(ni)
        return result if result else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  Price History — yfinance
# ---------------------------------------------------------------------------

def _fetch_price_yf(symbol: str) -> dict | None:
    """Return {year: price} — year-end (or latest) closing prices."""
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period="6y")
        if hist is None or hist.empty:
            return None
        # Get last trading day price for each calendar year
        result = {}
        hist.index = pd.to_datetime(hist.index)
        for year in hist.index.year.unique():
            yr_data = hist[hist.index.year == year]
            if not yr_data.empty:
                result[int(year)] = float(yr_data["Close"].iloc[-1])
        # Also add current price
        current_year = datetime.now().year
        if current_year not in result or True:
            # Overwrite with most recent
            result[current_year] = float(hist["Close"].iloc[-1])
        return result if result else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  Price History — FMP
# ---------------------------------------------------------------------------

def _fetch_price_fmp(symbol: str, api_key: str) -> dict | None:
    """Year-end prices from FMP historical endpoint."""
    if not api_key or api_key in ("FMP_API_KEY", ""):
        return None
    url = (f"https://financialmodelingprep.com/api/v3/historical-price-full/"
           f"{symbol}?apikey={api_key}")
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        prices = data.get("historical", [])
        if not prices:
            return None
        # Group by year, take last entry
        df = pd.DataFrame(prices)
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        result = {}
        for year, group in df.groupby("year"):
            result[int(year)] = float(group.sort_values("date").iloc[-1]["close"])
        return result if result else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  Price History — Alpha Vantage
# ---------------------------------------------------------------------------

def _fetch_price_av(symbol: str, api_key: str) -> dict | None:
    """Year-end prices from Alpha Vantage monthly adjusted."""
    if not api_key or api_key in ("ALPHA_VANTAGE_API_KEY", ""):
        return None
    url = (f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED"
           f"&symbol={symbol}&apikey={api_key}")
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        ts = data.get("Monthly Adjusted Time Series", {})
        if not ts:
            return None
        result = {}
        for date_str, vals in ts.items():
            year = int(date_str[:4])
            month = int(date_str[5:7])
            price = float(vals["5. adjusted close"])
            # Keep Dec or latest month per year
            if year not in result or month >= 12:
                result[year] = price
        return result if result else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  Market Cap / Info — yfinance
# ---------------------------------------------------------------------------

def _fetch_info_yf(symbol: str) -> dict:
    """Fetch current price, market cap from yfinance."""
    try:
        tk = yf.Ticker(symbol)
        info = tk.info or {}
        return {
            "price": _safe(info.get("currentPrice", info.get("regularMarketPrice")), 0),
            "market_cap": _safe(info.get("marketCap"), 0),
        }
    except Exception:
        return {"price": 0, "market_cap": 0}


# ---------------------------------------------------------------------------
#  Unified Fetch with Fallback
# ---------------------------------------------------------------------------

def _fetch_with_fallback(fetchers: list, label: str = "") -> tuple[dict | None, str]:
    """Try each fetcher in order, return (data, source_name) on first success."""
    for name, fn in fetchers:
        try:
            data = fn()
            if data:
                return data, name
        except Exception:
            continue
    return None, "NO_DATA"


# ---------------------------------------------------------------------------
#  Growth Calculation
# ---------------------------------------------------------------------------

def _compute_growths(ni_data: dict, price_data: dict, current_year: int) -> dict:
    """Compute CAGR for net income and price over 1, 3, 5 year trailing periods.

    Returns dict with keys like ni_cagr_1y, price_cagr_1y, gap_1y, etc.
    """
    result = {}
    for period in [1, 3, 5]:
        base_year = current_year - period

        # Net income CAGR
        ni_start = ni_data.get(base_year) if ni_data else None
        # For NI, use the most recent available year
        ni_end = None
        for y in [current_year, current_year - 1]:
            if ni_data and y in ni_data:
                ni_end = ni_data[y]
                break

        ni_cagr = _cagr(ni_start, ni_end, period)
        result[f"ni_cagr_{period}y"] = ni_cagr

        # Price CAGR
        price_start = price_data.get(base_year) if price_data else None
        price_end = price_data.get(current_year) if price_data else None
        price_cagr = _cagr(price_start, price_end, period)
        result[f"price_cagr_{period}y"] = price_cagr

        # Gap (positive = NI outpacing price = potential undervaluation)
        if ni_cagr is not None and price_cagr is not None:
            result[f"gap_{period}y"] = ni_cagr - price_cagr
        else:
            result[f"gap_{period}y"] = None

    return result


def _classify_signal(gaps: dict) -> str:
    """Classify stock based on average gap across available periods."""
    valid_gaps = [v for k, v in gaps.items() if k.startswith("gap_") and v is not None]
    if not valid_gaps:
        return "NO DATA"
    avg_gap = np.mean(valid_gaps)
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


# ---------------------------------------------------------------------------
#  Main Scanner
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def scan_sp500(
    tickers_df: pd.DataFrame,
    fmp_key: str = "",
    av_key: str = "",
    sectors: list[str] | None = None,
) -> pd.DataFrame:
    """Scan S&P 500 stocks for net income vs price growth discrepancies.

    Args:
        tickers_df: DataFrame with Symbol, Security, Sector columns
        fmp_key: FMP API key
        av_key: Alpha Vantage API key
        sectors: If provided, filter to these sectors only

    Returns:
        DataFrame with growth rates, gaps, and signals
    """
    df = tickers_df.copy()
    if sectors:
        df = df[df["Sector"].isin(sectors)].reset_index(drop=True)

    current_year = datetime.now().year
    results = []

    progress = st.progress(0, text="Scanning S&P 500...")
    total = len(df)

    for idx, row in df.iterrows():
        symbol = row["Symbol"]
        progress.progress(
            (idx + 1) / total,
            text=f"Scanning {symbol} ({idx + 1}/{total})"
        )

        # Fetch net income with fallback
        ni_fetchers = [
            ("YF", lambda s=symbol: _fetch_net_income_yf(s)),
            ("FMP", lambda s=symbol: _fetch_net_income_fmp(s, fmp_key)),
            ("AV", lambda s=symbol: _fetch_net_income_av(s, av_key)),
        ]
        ni_data, ni_source = _fetch_with_fallback(ni_fetchers, f"{symbol} NI")

        # Fetch price with fallback
        price_fetchers = [
            ("YF", lambda s=symbol: _fetch_price_yf(s)),
            ("FMP", lambda s=symbol: _fetch_price_fmp(s, fmp_key)),
            ("AV", lambda s=symbol: _fetch_price_av(s, av_key)),
        ]
        price_data, price_source = _fetch_with_fallback(price_fetchers, f"{symbol} Price")

        # Current info
        info = _fetch_info_yf(symbol)

        # Compute growths
        growths = _compute_growths(ni_data, price_data, current_year)

        # Build record
        record = {
            "Symbol": symbol,
            "Security": row["Security"],
            "Sector": row["Sector"],
            "Price": info["price"],
            "Mkt Cap ($B)": round(info["market_cap"] / 1e9, 1) if info["market_cap"] else 0,
        }

        for period in [1, 3, 5]:
            record[f"NI CAGR {period}Y"] = growths.get(f"ni_cagr_{period}y")
            record[f"Price CAGR {period}Y"] = growths.get(f"price_cagr_{period}y")
            record[f"Gap {period}Y"] = growths.get(f"gap_{period}y")

        record["Signal"] = _classify_signal(growths)
        record["NI Source"] = ni_source
        record["Price Source"] = price_source

        # Turnaround detection: NI was negative and is now positive
        if ni_data:
            recent = ni_data.get(current_year, ni_data.get(current_year - 1))
            oldest = ni_data.get(current_year - 5, ni_data.get(current_year - 3))
            if oldest is not None and recent is not None:
                if oldest < 0 < recent:
                    record["Signal"] = "TURNAROUND"

        results.append(record)

        # Rate limit
        time.sleep(0.15)

    progress.empty()
    return pd.DataFrame(results)
