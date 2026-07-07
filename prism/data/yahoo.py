"""yfinance wrapper: price/beta/market-cap snapshot plus estimate fallbacks."""
import logging

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger("prism.yahoo")

# Watchlist symbols whose Yahoo Finance spelling differs (class shares use '-').
YF_TICKER_MAP = {
    "BRK.B": "BRK-B",
    "BRK.A": "BRK-A",
}


def yf_symbol(symbol):
    return YF_TICKER_MAP.get(symbol, symbol)


def fetch_yahoo_snapshot(symbol):
    """Fetch price, market cap, beta, 52w range, industry, and analyst target.

    Returns (ticker_obj, info_dict) — ticker_obj may be reused for estimate calls.
    """
    fields = {
        "price": np.nan, "mcap": np.nan, "beta": np.nan,
        "high52": np.nan, "low52": np.nan,
        "industry": "Unknown", "target_mean": np.nan,
    }
    try:
        tk = yf.Ticker(yf_symbol(symbol))
        info = tk.info
        fields["price"] = info.get("currentPrice") or info.get("regularMarketPrice", np.nan)
        fields["mcap"] = info.get("marketCap", np.nan)
        fields["beta"] = info.get("beta", np.nan)
        fields["high52"] = info.get("fiftyTwoWeekHigh", np.nan)
        fields["low52"] = info.get("fiftyTwoWeekLow", np.nan)
        fields["industry"] = info.get("industry", "Unknown")
        fields["target_mean"] = info.get("targetMeanPrice", np.nan)
        fields["forward_pe_info"] = info.get("forwardPE", np.nan)
    except Exception as e:
        log.warning("yfinance %s: snapshot failed: %s", symbol, e)
        tk = None
        fields["forward_pe_info"] = np.nan
    return tk, fields


def fetch_yahoo_growth(tk, symbol):
    """Fallback growth estimates from yfinance (current year and next year)."""
    if tk is None:
        return np.nan, np.nan
    try:
        ge = tk.get_growth_estimates()
        if ge is None or ge.empty:
            return np.nan, np.nan
        ysym = yf_symbol(symbol)
        if ysym in ge.columns:
            col = ge[ysym]
        elif "stock" in ge.columns:
            col = ge["stock"]
        elif len(ge.columns) > 0:
            col = ge.iloc[:, 0]
        else:
            col = pd.Series(dtype=float)
        yf_cy = col.get("0y", np.nan)
        yf_ny = col.get("+1y", np.nan)
        # Yahoo sometimes reports percentages instead of decimals.
        if pd.notna(yf_cy) and abs(yf_cy) > 5:
            yf_cy /= 100
        if pd.notna(yf_ny) and abs(yf_ny) > 5:
            yf_ny /= 100
        return yf_cy, yf_ny
    except Exception as e:
        log.info("yfinance %s: growth estimates unavailable: %s", symbol, e)
        return np.nan, np.nan


def fetch_yahoo_forward_eps(tk, symbol):
    """Fallback current-year average EPS estimate from yfinance."""
    if tk is None:
        return np.nan
    try:
        ee = tk.get_earnings_estimate()
        if ee is not None and not ee.empty and "0y" in ee.index and "avg" in ee.columns:
            return ee.loc["0y", "avg"]
    except Exception as e:
        log.info("yfinance %s: earnings estimate unavailable: %s", symbol, e)
    return np.nan
