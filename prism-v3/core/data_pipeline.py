"""
PRISM v3 — Multi-Source Data Pipeline
FMP (primary) → Finnhub (validation) → yfinance (fallback)
Plus technical indicators via ta library
"""

import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta

try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def safe(val, default=np.nan):
    """Return default if val is None or NaN."""
    if val is None:
        return default
    try:
        if np.isnan(val):
            return default
    except (TypeError, ValueError):
        pass
    return val


def _fiscal_year_for_calendar(fy_end_month: int, cal_year: int) -> int:
    """Map FMP fiscal year to calendar year. FY ending Jan-May → prior calendar year."""
    return cal_year - 1 if fy_end_month <= 5 else cal_year


# ---------------------------------------------------------------------------
#  FMP — Primary source for EPS estimates / growth
# ---------------------------------------------------------------------------

def fetch_fmp_estimates(symbol: str, api_key: str) -> dict | None:
    """Fetch analyst EPS estimates from FMP stable endpoint."""
    url = f"https://financialmodelingprep.com/stable/analyst-estimates?symbol={symbol}&period=annual&apikey={api_key}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data or not isinstance(data, list):
            return None
        # Sort by year descending
        data.sort(key=lambda x: x.get("calendarYear", x.get("date", ""))[:4] if isinstance(x.get("date"), str) else str(x.get("calendarYear", 0)), reverse=True)
        return data
    except Exception:
        return None


def parse_fmp_growth(data: list, current_year: int = None) -> dict:
    """Extract growth rates from FMP estimates.
    Returns dict with keys: fwd_eps, g26, g27, rev_growth, confidence.
    Uses EPS growth as primary, revenue growth as sanity check / fallback.
    """
    if current_year is None:
        current_year = datetime.now().year

    result = {"fwd_eps": np.nan, "g26": np.nan, "g27": np.nan,
              "rev_growth": np.nan, "confidence": "NONE"}

    if not data:
        return result

    # Build year→eps and year→rev maps from FMP data
    year_eps = {}
    year_rev = {}
    for entry in data:
        yr = None
        for key in ["calendarYear", "date"]:
            val = entry.get(key)
            if val:
                yr_str = str(val)[:4]
                try:
                    yr = int(yr_str)
                except ValueError:
                    continue
                break
        if yr is None:
            continue

        eps = None
        for key in ["epsAvg", "estimatedEpsAvg", "epsConsensus"]:
            if key in entry and entry[key] is not None:
                try:
                    eps = float(entry[key])
                except (ValueError, TypeError):
                    pass
                break
        if eps is not None:
            year_eps[yr] = eps

        rev = None
        for key in ["revenueAvg", "estimatedRevenueAvg", "revenueConsensus"]:
            if key in entry and entry[key] is not None:
                try:
                    rev = float(entry[key])
                except (ValueError, TypeError):
                    pass
                break
        if rev is not None:
            year_rev[yr] = rev

    if not year_eps and not year_rev:
        return result

    # Current year and next year EPS
    cy_eps = year_eps.get(current_year)
    ny_eps = year_eps.get(current_year + 1)
    py_eps = year_eps.get(current_year - 1)

    # Forward EPS = next year estimate (used for forward P/E)
    if ny_eps:
        result["fwd_eps"] = ny_eps
    elif cy_eps:
        result["fwd_eps"] = cy_eps

    # EPS Growth rates
    if cy_eps and py_eps and py_eps > 0:
        result["g26"] = (cy_eps - py_eps) / abs(py_eps)
    if ny_eps and cy_eps and cy_eps > 0:
        result["g27"] = (ny_eps - cy_eps) / abs(cy_eps)

    # Revenue growth as sanity check / fallback
    cy_rev = year_rev.get(current_year)
    py_rev = year_rev.get(current_year - 1)
    if cy_rev and py_rev and py_rev > 0:
        result["rev_growth"] = (cy_rev - py_rev) / py_rev

    # If EPS growth is negative but revenue growth is strongly positive,
    # the EPS decline is likely transient (one-time items, share-based comp, etc.)
    # Use revenue growth as a more stable proxy
    if not np.isnan(result["g26"]) and result["g26"] < -0.1:
        if not np.isnan(result["rev_growth"]) and result["rev_growth"] > 0.10:
            result["g26"] = result["rev_growth"]
            result["confidence"] = "MED_REV"
            return result

    if not np.isnan(result["g26"]) or not np.isnan(result["g27"]):
        result["confidence"] = "HIGH"

    return result


# ---------------------------------------------------------------------------
#  Finnhub — Validation source
# ---------------------------------------------------------------------------

def fetch_finnhub_eps(symbol: str, api_key: str) -> dict | None:
    """Fetch EPS estimates from Finnhub."""
    url = f"https://finnhub.io/api/v1/stock/eps-estimate?symbol={symbol}&freq=annual&token={api_key}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data or "data" not in data:
            return None
        return data["data"]
    except Exception:
        return None


def fetch_finnhub_revenue(symbol: str, api_key: str) -> dict | None:
    """Fetch revenue estimates from Finnhub as fallback."""
    url = f"https://finnhub.io/api/v1/stock/revenue-estimate?symbol={symbol}&freq=annual&token={api_key}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data or "data" not in data:
            return None
        return data["data"]
    except Exception:
        return None


def parse_finnhub_growth(data: list, current_year: int = None) -> dict:
    """Extract growth from Finnhub estimates."""
    if current_year is None:
        current_year = datetime.now().year

    result = {"g26": np.nan, "g27": np.nan}
    if not data:
        return result

    year_eps = {}
    for entry in data:
        period = entry.get("period", "")[:4]
        try:
            yr = int(period)
        except (ValueError, TypeError):
            continue
        eps = entry.get("epsAvg")
        if eps is not None:
            try:
                year_eps[yr] = float(eps)
            except (ValueError, TypeError):
                pass

    py_eps = year_eps.get(current_year - 1)
    cy_eps = year_eps.get(current_year)
    ny_eps = year_eps.get(current_year + 1)

    if cy_eps and py_eps and py_eps > 0:
        result["g26"] = (cy_eps - py_eps) / abs(py_eps)
    if ny_eps and cy_eps and cy_eps > 0:
        result["g27"] = (ny_eps - cy_eps) / abs(cy_eps)

    return result


def parse_finnhub_rev_growth(data: list, current_year: int = None) -> float:
    """Extract revenue growth from Finnhub revenue estimates."""
    if current_year is None:
        current_year = datetime.now().year
    if not data:
        return np.nan

    year_rev = {}
    for entry in data:
        period = entry.get("period", "")[:4]
        try:
            yr = int(period)
        except (ValueError, TypeError):
            continue
        rev = entry.get("revenueAvg")
        if rev is not None:
            try:
                year_rev[yr] = float(rev)
            except (ValueError, TypeError):
                pass

    cy_rev = year_rev.get(current_year)
    py_rev = year_rev.get(current_year - 1)
    if cy_rev and py_rev and py_rev > 0:
        return (cy_rev - py_rev) / py_rev
    return np.nan


# ---------------------------------------------------------------------------
#  yfinance — Fallback + price/technicals
# ---------------------------------------------------------------------------

def fetch_yf_data(symbol: str) -> dict:
    """Fetch price, fundamentals, and historical data from yfinance."""
    result = {
        "price": np.nan, "market_cap": np.nan, "beta": np.nan,
        "pe_trailing": np.nan, "pe_forward": np.nan,
        "52w_high": np.nan, "52w_low": np.nan, "52w_position": np.nan,
        "sector": "", "industry": "",
        "analyst_target": np.nan, "recommendation": "",
        "dividend_yield": 0.0,
        "yf_growth": np.nan, "yf_fwd_eps": np.nan,
        "yf_rev_growth": np.nan, "yf_eps_growth": np.nan,
        # Technical data
        "history": None
    }
    try:
        tk = yf.Ticker(symbol)
        info = tk.info or {}

        result["price"] = safe(info.get("currentPrice") or info.get("regularMarketPrice"), np.nan)
        result["market_cap"] = safe(info.get("marketCap"), np.nan)
        result["beta"] = safe(info.get("beta"), np.nan)
        result["pe_trailing"] = safe(info.get("trailingPE"), np.nan)
        result["sector"] = info.get("sector", "")
        result["industry"] = info.get("industry", "")
        result["analyst_target"] = safe(info.get("targetMeanPrice"), np.nan)
        result["recommendation"] = info.get("recommendationKey", "")
        result["dividend_yield"] = safe(info.get("dividendYield"), 0.0)
        result["52w_high"] = safe(info.get("fiftyTwoWeekHigh"), np.nan)
        result["52w_low"] = safe(info.get("fiftyTwoWeekLow"), np.nan)

        # Forward P/E — calculate ourselves from yfinance forwardEps
        fwd_eps = safe(info.get("forwardEps"), np.nan)
        if not np.isnan(fwd_eps) and fwd_eps > 0 and not np.isnan(result["price"]):
            result["pe_forward"] = result["price"] / fwd_eps
            result["yf_fwd_eps"] = fwd_eps

        # 52w position
        h, l, p = result["52w_high"], result["52w_low"], result["price"]
        if not any(np.isnan(x) for x in [h, l, p]) and h > l:
            result["52w_position"] = (p - l) / (h - l)

        # Revenue growth (reliable, always available)
        rev_growth = safe(info.get("revenueGrowth"), np.nan)
        if not np.isnan(rev_growth):
            result["yf_rev_growth"] = rev_growth

        # EPS growth = forwardEps / trailingEps - 1
        trailing_eps = safe(info.get("trailingEps"), np.nan)
        if not np.isnan(fwd_eps) and not np.isnan(trailing_eps) and trailing_eps > 0:
            eps_g = (fwd_eps - trailing_eps) / abs(trailing_eps)
            # Cap extreme EPS growth from cyclical recovery (e.g., MU 500%+)
            # Use revenue growth as sanity check
            if eps_g > 1.0 and not np.isnan(rev_growth) and rev_growth < eps_g * 0.5:
                # EPS growth inflated vs revenue — use blended estimate
                result["yf_eps_growth"] = rev_growth * 1.5  # revenue growth + margin expansion
            else:
                result["yf_eps_growth"] = eps_g

        # Growth from yfinance analyst estimates (original fallback)
        try:
            ge = tk.get_growth_estimates()
            if ge is not None and not ge.empty and symbol in ge.columns:
                val = ge.loc["+1y", symbol] if "+1y" in ge.index else None
                if val is not None and not np.isnan(val):
                    result["yf_growth"] = float(val)
        except Exception:
            pass

        # Historical data (1 year for technicals)
        try:
            hist = tk.history(period="1y", auto_adjust=True)
            if hist is not None and not hist.empty:
                result["history"] = hist
        except Exception:
            pass

    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
#  Technical Indicators
# ---------------------------------------------------------------------------

def compute_technicals(hist: pd.DataFrame) -> dict:
    """Compute technical indicators from price history."""
    result = {
        "rsi_14": np.nan, "sma_50": np.nan, "sma_200": np.nan,
        "macd": np.nan, "macd_signal": np.nan, "macd_histogram": np.nan,
        "bb_upper": np.nan, "bb_middle": np.nan, "bb_lower": np.nan,
        "bb_pctb": np.nan,
        "avg_volume_20": np.nan, "current_volume": np.nan, "rel_volume": np.nan,
        "trend_score": 0, "momentum_score": 0, "volatility_score": 0,
        "technical_score": 0
    }

    if hist is None or len(hist) < 50:
        return result

    close = hist["Close"]
    volume = hist.get("Volume")

    # --- RSI(14) ---
    if HAS_TA:
        rsi_series = ta.momentum.RSIIndicator(close, window=14).rsi()
        result["rsi_14"] = rsi_series.iloc[-1] if len(rsi_series) > 0 else np.nan
    else:
        # Manual RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        result["rsi_14"] = rsi.iloc[-1] if len(rsi) > 0 else np.nan

    # --- Moving Averages ---
    result["sma_50"] = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
    if len(close) >= 200:
        result["sma_200"] = close.rolling(200).mean().iloc[-1]

    # --- MACD ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    result["macd"] = macd_line.iloc[-1]
    result["macd_signal"] = signal_line.iloc[-1]
    result["macd_histogram"] = result["macd"] - result["macd_signal"]

    # --- Bollinger Bands ---
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    result["bb_upper"] = (sma20 + 2 * std20).iloc[-1]
    result["bb_middle"] = sma20.iloc[-1]
    result["bb_lower"] = (sma20 - 2 * std20).iloc[-1]
    if result["bb_upper"] > result["bb_lower"]:
        result["bb_pctb"] = (close.iloc[-1] - result["bb_lower"]) / (result["bb_upper"] - result["bb_lower"])

    # --- Volume ---
    if volume is not None and len(volume) >= 20:
        result["avg_volume_20"] = volume.rolling(20).mean().iloc[-1]
        result["current_volume"] = volume.iloc[-1]
        if result["avg_volume_20"] > 0:
            result["rel_volume"] = result["current_volume"] / result["avg_volume_20"]

    # --- Composite Scores (0-100 scale) ---
    price = close.iloc[-1]

    # Trend Score: price vs SMA50/200
    trend = 50  # neutral
    if not np.isnan(result["sma_50"]):
        if price > result["sma_50"]:
            trend += 20
        else:
            trend -= 20
    if not np.isnan(result["sma_200"]):
        if price > result["sma_200"]:
            trend += 15
        else:
            trend -= 15
    # Golden/death cross
    if not np.isnan(result["sma_50"]) and not np.isnan(result["sma_200"]):
        if result["sma_50"] > result["sma_200"]:
            trend += 15  # golden cross
        else:
            trend -= 15  # death cross
    result["trend_score"] = max(0, min(100, trend))

    # Momentum Score: RSI + MACD
    mom = 50
    rsi_val = result["rsi_14"]
    if not np.isnan(rsi_val):
        if 40 <= rsi_val <= 60:
            mom += 5  # neutral, slight positive
        elif rsi_val > 60:
            mom += min(25, (rsi_val - 60) * 0.6)  # bullish but cap it
        elif rsi_val < 40:
            mom -= min(25, (40 - rsi_val) * 0.6)
        # Oversold bounce potential
        if rsi_val < 30:
            mom += 10  # contrarian signal
    if not np.isnan(result["macd_histogram"]):
        if result["macd_histogram"] > 0:
            mom += 15
        else:
            mom -= 10
    result["momentum_score"] = max(0, min(100, mom))

    # Volatility Score: BB position (higher = better positioned)
    vol = 50
    if not np.isnan(result["bb_pctb"]):
        if result["bb_pctb"] < 0.2:
            vol += 20  # near lower band = potential bounce
        elif result["bb_pctb"] > 0.8:
            vol -= 15  # near upper band = stretched
        else:
            vol += 5  # middle = healthy
    result["volatility_score"] = max(0, min(100, vol))

    # Overall Technical Score
    result["technical_score"] = int(
        result["trend_score"] * 0.40 +
        result["momentum_score"] * 0.40 +
        result["volatility_score"] * 0.20
    )

    return result


# ---------------------------------------------------------------------------
#  Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_growth(fmp_g: float, fh_g: float, yf_g: float) -> tuple[float, str]:
    """Cross-validate growth rates from 3 sources. Returns (best_estimate, confidence)."""
    sources = [(fmp_g, "FMP"), (fh_g, "FH"), (yf_g, "YF")]
    valid = [(g, s) for g, s in sources if not np.isnan(g)]

    if len(valid) == 0:
        return np.nan, "DEFAULT"

    if len(valid) == 1:
        return valid[0][0], f"MED ({valid[0][1]} only)"

    # Compare top two
    g1, s1 = valid[0]
    g2, s2 = valid[1]

    avg = (g1 + g2) / 2
    if avg == 0:
        gap_pct = 0
    else:
        gap_pct = abs(g1 - g2) / max(abs(g1), abs(g2), 0.01)

    if gap_pct <= 0.15:
        return g1, f"HIGH ({s1}≈{s2})"
    elif gap_pct <= 0.30:
        return avg, f"MED ({s1}~{s2})"
    else:
        return g1, f"LOW ({s1}≠{s2})"


# ---------------------------------------------------------------------------
#  Master Pipeline
# ---------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def fetch_all_data(tickers: list[str], fmp_key: str, fh_key: str,
                   growth_overrides: dict = None, default_growth: float = 0.04) -> pd.DataFrame:
    """
    Fetch all data for a list of tickers using the 3-layer pipeline.
    Returns a DataFrame with fundamentals, technicals, and data quality.
    """
    if growth_overrides is None:
        growth_overrides = {}

    current_year = datetime.now().year
    rows = []
    quality_log = []

    fmp_enabled = bool(fmp_key and fmp_key != "FMP_API_KEY")
    fh_enabled = bool(fh_key and fh_key != "FINNHUB_API_KEY")
    fmp_fails = 0

    for i, sym in enumerate(tickers):
        row = {"Ticker": sym}

        # --- Layer 1: yfinance (always, for price + technicals) ---
        yf_data = fetch_yf_data(sym)
        row["Price"] = yf_data["price"]
        row["Market Cap ($B)"] = yf_data["market_cap"] / 1e9 if not np.isnan(yf_data["market_cap"]) else np.nan
        row["Beta"] = yf_data["beta"]
        row["PE (TTM)"] = yf_data["pe_trailing"]
        row["52w High"] = yf_data["52w_high"]
        row["52w Low"] = yf_data["52w_low"]
        row["52w Position"] = yf_data["52w_position"]
        row["Sector"] = yf_data["sector"]
        row["Industry"] = yf_data["industry"]
        row["Analyst Target"] = yf_data["analyst_target"]
        row["Recommendation"] = yf_data["recommendation"]
        row["Dividend Yield"] = yf_data["dividend_yield"]

        # --- Technical indicators ---
        tech = compute_technicals(yf_data["history"])
        for k, v in tech.items():
            row[k] = v

        # --- Layer 2: FMP (primary for growth) ---
        fmp_g26 = fmp_g27 = fmp_fwd_eps = np.nan
        if fmp_enabled and fmp_fails < 3:
            fmp_data = fetch_fmp_estimates(sym, fmp_key)
            if fmp_data is None:
                fmp_fails += 1
            else:
                parsed = parse_fmp_growth(fmp_data, current_year)
                fmp_g26 = parsed["g26"]
                fmp_g27 = parsed["g27"]
                fmp_fwd_eps = parsed["fwd_eps"]
                if parsed["confidence"] != "NONE":
                    fmp_fails = max(0, fmp_fails - 1)  # reward success
            time.sleep(0.3)  # FMP rate limit

        # --- Layer 3: Finnhub (validation) ---
        fh_g26 = fh_g27 = np.nan
        if fh_enabled:
            fh_data = fetch_finnhub_eps(sym, fh_key)
            if fh_data:
                fh_parsed = parse_finnhub_growth(fh_data, current_year)
                fh_g26 = fh_parsed["g26"]
                fh_g27 = fh_parsed["g27"]
            time.sleep(0.15)  # Finnhub: 60 calls/min

            # Finnhub revenue as extra fallback if EPS is missing
            if np.isnan(fh_g26):
                fh_rev_data = fetch_finnhub_revenue(sym, fh_key)
                if fh_rev_data:
                    fh_rev_g = parse_finnhub_rev_growth(fh_rev_data, current_year)
                    if not np.isnan(fh_rev_g):
                        fh_g26 = fh_rev_g
                time.sleep(0.15)

        # --- Cross-validate ---
        # Priority: FMP EPS > Finnhub EPS > yfinance analyst growth >
        #           yfinance EPS growth > yfinance revenue growth > default
        yf_g = yf_data["yf_growth"]
        yf_eps_g = yf_data["yf_eps_growth"]
        yf_rev_g = yf_data["yf_rev_growth"]

        # Check for manual override first
        if sym in growth_overrides:
            row["Growth"] = growth_overrides[sym]
            row["Growth Confidence"] = "OVERRIDE"
        else:
            # Try multi-source cross-validation first
            best_g, conf = cross_validate_growth(fmp_g26, fh_g26, yf_g)

            if np.isnan(best_g):
                # Fallback: yfinance EPS growth (fwd vs trailing)
                if not np.isnan(yf_eps_g):
                    best_g = yf_eps_g
                    conf = "MED (YF EPS)"
                # Fallback: yfinance revenue growth
                elif not np.isnan(yf_rev_g):
                    best_g = yf_rev_g
                    conf = "LOW (YF Rev)"
                else:
                    best_g = default_growth
                    conf = "DEFAULT"

            row["Growth"] = best_g
            row["Growth Confidence"] = conf

        # Forward P/E: prefer FMP eps, fallback to yfinance
        # Guard against currency mismatch (e.g., TSM: FMP reports in TWD, price in USD)
        fwd_eps = fmp_fwd_eps if not np.isnan(fmp_fwd_eps) else np.nan
        yf_fwd_pe = yf_data["pe_forward"]

        if not np.isnan(fwd_eps) and fwd_eps > 0 and not np.isnan(row["Price"]):
            fmp_pe = row["Price"] / fwd_eps
            # Sanity check: if FMP PE < 1x, it's likely a currency mismatch
            if fmp_pe < 1.0 and not np.isnan(yf_fwd_pe) and yf_fwd_pe > 1.0:
                row["PE (Fwd)"] = yf_fwd_pe  # use yfinance (correct currency)
            else:
                row["PE (Fwd)"] = fmp_pe
        elif not np.isnan(yf_fwd_pe):
            row["PE (Fwd)"] = yf_fwd_pe
        else:
            row["PE (Fwd)"] = yf_data["pe_trailing"]  # last resort

        # Growth G27 (next year)
        best_g27, _ = cross_validate_growth(fmp_g27, fh_g27, np.nan)
        row["Growth G27"] = best_g27

        # Quality log
        quality_log.append({
            "Ticker": sym,
            "FMP": f"{fmp_g26:.1%}" if not np.isnan(fmp_g26) else "—",
            "Finnhub": f"{fh_g26:.1%}" if not np.isnan(fh_g26) else "—",
            "YF Analyst": f"{yf_g:.1%}" if not np.isnan(yf_g) else "—",
            "YF EPS": f"{yf_eps_g:.1%}" if not np.isnan(yf_eps_g) else "—",
            "YF Rev": f"{yf_rev_g:.1%}" if not np.isnan(yf_rev_g) else "—",
            "Used": row["Growth Confidence"],
            "Final": f"{row['Growth']:.1%}"
        })

        rows.append(row)

    df = pd.DataFrame(rows)

    # Store quality log as attribute
    df.attrs["quality_log"] = pd.DataFrame(quality_log)

    return df
