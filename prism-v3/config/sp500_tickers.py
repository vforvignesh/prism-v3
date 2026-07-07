"""
S&P 500 ticker list — dynamic fetch from Wikipedia with hardcoded fallback.
"""

import pandas as pd
import streamlit as st


@st.cache_data(ttl=86400)
def get_sp500_tickers() -> pd.DataFrame:
    """Return DataFrame with columns: Symbol, Security, Sector, Sub-Industry.

    Tries Wikipedia first, falls back to a hardcoded top-100 list.
    """
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            attrs={"id": "constituents"},
        )
        df = tables[0][["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
        df.columns = ["Symbol", "Security", "Sector", "SubIndustry"]
        # Fix tickers with dots (e.g. BRK.B → BRK-B for yfinance)
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        return df.reset_index(drop=True)
    except Exception:
        return _fallback_list()


def _fallback_list() -> pd.DataFrame:
    """Hardcoded subset — top ~120 stocks by market cap."""
    tickers = [
        ("AAPL", "Apple Inc.", "Information Technology"),
        ("MSFT", "Microsoft Corp.", "Information Technology"),
        ("AMZN", "Amazon.com Inc.", "Consumer Discretionary"),
        ("NVDA", "NVIDIA Corp.", "Information Technology"),
        ("GOOGL", "Alphabet Inc. A", "Communication Services"),
        ("META", "Meta Platforms", "Communication Services"),
        ("BRK-B", "Berkshire Hathaway B", "Financials"),
        ("TSLA", "Tesla Inc.", "Consumer Discretionary"),
        ("UNH", "UnitedHealth Group", "Health Care"),
        ("LLY", "Eli Lilly", "Health Care"),
        ("JPM", "JPMorgan Chase", "Financials"),
        ("V", "Visa Inc.", "Financials"),
        ("XOM", "Exxon Mobil", "Energy"),
        ("AVGO", "Broadcom Inc.", "Information Technology"),
        ("MA", "Mastercard", "Financials"),
        ("JNJ", "Johnson & Johnson", "Health Care"),
        ("PG", "Procter & Gamble", "Consumer Staples"),
        ("HD", "Home Depot", "Consumer Discretionary"),
        ("COST", "Costco Wholesale", "Consumer Staples"),
        ("ABBV", "AbbVie Inc.", "Health Care"),
        ("MRK", "Merck & Co.", "Health Care"),
        ("CRM", "Salesforce Inc.", "Information Technology"),
        ("CVX", "Chevron Corp.", "Energy"),
        ("NFLX", "Netflix Inc.", "Communication Services"),
        ("AMD", "Advanced Micro Devices", "Information Technology"),
        ("KO", "Coca-Cola Co.", "Consumer Staples"),
        ("PEP", "PepsiCo Inc.", "Consumer Staples"),
        ("WMT", "Walmart Inc.", "Consumer Staples"),
        ("TMO", "Thermo Fisher Scientific", "Health Care"),
        ("ADBE", "Adobe Inc.", "Information Technology"),
        ("BAC", "Bank of America", "Financials"),
        ("ACN", "Accenture plc", "Information Technology"),
        ("DIS", "Walt Disney Co.", "Communication Services"),
        ("MCD", "McDonald's Corp.", "Consumer Discretionary"),
        ("CSCO", "Cisco Systems", "Information Technology"),
        ("ABT", "Abbott Laboratories", "Health Care"),
        ("DHR", "Danaher Corp.", "Health Care"),
        ("INTC", "Intel Corp.", "Information Technology"),
        ("CMCSA", "Comcast Corp.", "Communication Services"),
        ("VZ", "Verizon Communications", "Communication Services"),
        ("IBM", "IBM Corp.", "Information Technology"),
        ("INTU", "Intuit Inc.", "Information Technology"),
        ("PM", "Philip Morris", "Consumer Staples"),
        ("TXN", "Texas Instruments", "Information Technology"),
        ("QCOM", "Qualcomm Inc.", "Information Technology"),
        ("NOW", "ServiceNow Inc.", "Information Technology"),
        ("GE", "GE Aerospace", "Industrials"),
        ("CAT", "Caterpillar Inc.", "Industrials"),
        ("AMGN", "Amgen Inc.", "Health Care"),
        ("ISRG", "Intuitive Surgical", "Health Care"),
        ("SPGI", "S&P Global", "Financials"),
        ("AMAT", "Applied Materials", "Information Technology"),
        ("GS", "Goldman Sachs", "Financials"),
        ("BKNG", "Booking Holdings", "Consumer Discretionary"),
        ("T", "AT&T Inc.", "Communication Services"),
        ("SYK", "Stryker Corp.", "Health Care"),
        ("BLK", "BlackRock Inc.", "Financials"),
        ("ADP", "Automatic Data Processing", "Industrials"),
        ("MDLZ", "Mondelez International", "Consumer Staples"),
        ("LMT", "Lockheed Martin", "Industrials"),
        ("PFE", "Pfizer Inc.", "Health Care"),
        ("DE", "Deere & Co.", "Industrials"),
        ("GILD", "Gilead Sciences", "Health Care"),
        ("MMC", "Marsh & McLennan", "Financials"),
        ("UNP", "Union Pacific", "Industrials"),
        ("AXP", "American Express", "Financials"),
        ("MS", "Morgan Stanley", "Financials"),
        ("ETN", "Eaton Corp.", "Industrials"),
        ("CB", "Chubb Ltd.", "Financials"),
        ("LOW", "Lowe's Companies", "Consumer Discretionary"),
        ("SCHW", "Charles Schwab", "Financials"),
        ("RTX", "RTX Corp.", "Industrials"),
        ("SO", "Southern Co.", "Utilities"),
        ("NEE", "NextEra Energy", "Utilities"),
        ("DUK", "Duke Energy", "Utilities"),
        ("TGT", "Target Corp.", "Consumer Discretionary"),
        ("BMY", "Bristol-Myers Squibb", "Health Care"),
        ("CI", "Cigna Group", "Health Care"),
        ("ELV", "Elevance Health", "Health Care"),
        ("PLD", "Prologis Inc.", "Real Estate"),
        ("AMT", "American Tower", "Real Estate"),
        ("CCI", "Crown Castle", "Real Estate"),
        ("PSA", "Public Storage", "Real Estate"),
        ("WM", "Waste Management", "Industrials"),
        ("APD", "Air Products", "Materials"),
        ("SHW", "Sherwin-Williams", "Materials"),
        ("ECL", "Ecolab Inc.", "Materials"),
        ("NEM", "Newmont Corp.", "Materials"),
        ("FCX", "Freeport-McMoRan", "Materials"),
        ("COP", "ConocoPhillips", "Energy"),
        ("SLB", "Schlumberger", "Energy"),
        ("EOG", "EOG Resources", "Energy"),
        ("MPC", "Marathon Petroleum", "Energy"),
        ("PSX", "Phillips 66", "Energy"),
        ("D", "Dominion Energy", "Utilities"),
        ("AEP", "American Electric Power", "Utilities"),
        ("EXC", "Exelon Corp.", "Utilities"),
        ("SRE", "Sempra", "Utilities"),
        ("XEL", "Xcel Energy", "Utilities"),
        ("MU", "Micron Technology", "Information Technology"),
        ("LRCX", "Lam Research", "Information Technology"),
        ("KLAC", "KLA Corp.", "Information Technology"),
        ("SNPS", "Synopsys Inc.", "Information Technology"),
        ("CDNS", "Cadence Design Systems", "Information Technology"),
        ("PANW", "Palo Alto Networks", "Information Technology"),
        ("CRWD", "CrowdStrike Holdings", "Information Technology"),
        ("FTNT", "Fortinet Inc.", "Information Technology"),
        ("ORCL", "Oracle Corp.", "Information Technology"),
        ("UBER", "Uber Technologies", "Industrials"),
        ("ABNB", "Airbnb Inc.", "Consumer Discretionary"),
        ("SNOW", "Snowflake Inc.", "Information Technology"),
        ("SQ", "Block Inc.", "Financials"),
        ("SHOP", "Shopify Inc.", "Information Technology"),
        ("MELI", "MercadoLibre", "Consumer Discretionary"),
        ("CME", "CME Group", "Financials"),
        ("ICE", "Intercontinental Exchange", "Financials"),
        ("AON", "Aon plc", "Financials"),
        ("FIS", "Fidelity National Info", "Financials"),
        ("GM", "General Motors", "Consumer Discretionary"),
        ("F", "Ford Motor", "Consumer Discretionary"),
        ("NKE", "Nike Inc.", "Consumer Discretionary"),
        ("SBUX", "Starbucks Corp.", "Consumer Discretionary"),
        ("LULU", "Lululemon Athletica", "Consumer Discretionary"),
    ]
    df = pd.DataFrame(tickers, columns=["Symbol", "Security", "Sector"])
    df["SubIndustry"] = ""
    return df


def get_sp500_sectors(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique GICS sectors."""
    return sorted(df["Sector"].dropna().unique().tolist())
