"""Shock scenarios, sector mapping, and portfolio stress testing."""
import numpy as np
import pandas as pd

SHOCK_SCENARIOS = {
    "Rate Shock":      {"probability": 0.10, "market_dd": -0.25, "valuation_matters": True,  "mean_reversion": True,  "profit_matters": True,  "liquidity_matters": False, "sector_multipliers": {"Semiconductor": 1.2, "Hardware": 1.1, "Software": 1.3, "Cybersecurity": 1.3, "Cloud Infra": 1.4, "Streaming": 1.1, "E-commerce": 1.2, "Financials": 0.8, "Banking": 0.9, "Pharma": 0.7, "Healthcare": 0.7, "Insurance": 0.7, "Oil Services": 0.8, "Auto": 0.9}},
    "COVID Crash":     {"probability": 0.05, "market_dd": -0.34, "valuation_matters": False, "mean_reversion": False, "profit_matters": True,  "liquidity_matters": True,  "sector_multipliers": {"Semiconductor": 1.0, "Hardware": 1.0, "Software": 0.9, "Cybersecurity": 0.9, "Cloud Infra": 0.9, "Streaming": 0.7, "E-commerce": 0.8, "Financials": 1.1, "Banking": 1.1, "Pharma": 0.8, "Healthcare": 0.8, "Insurance": 1.0, "Oil Services": 1.5, "Auto": 1.2}},
    "AI Bubble Pop":   {"probability": 0.15, "market_dd": -0.15, "valuation_matters": True,  "mean_reversion": True,  "profit_matters": True,  "liquidity_matters": False, "sector_multipliers": {"Semiconductor": 2.5, "Hardware": 1.8, "Software": 1.3, "Cybersecurity": 1.5, "Cloud Infra": 2.0, "Streaming": 0.8, "E-commerce": 0.8, "Financials": 0.3, "Banking": 0.4, "Pharma": 0.3, "Healthcare": 0.3, "Insurance": 0.3, "Oil Services": 0.2, "Auto": 0.3}},
    "Tariff War":      {"probability": 0.12, "market_dd": -0.20, "valuation_matters": False, "mean_reversion": False, "profit_matters": False, "liquidity_matters": False, "sector_multipliers": {"Semiconductor": 1.8, "Hardware": 1.6, "Software": 0.9, "Cybersecurity": 0.8, "Cloud Infra": 1.0, "Streaming": 0.8, "E-commerce": 1.2, "Financials": 1.0, "Banking": 1.0, "Pharma": 0.6, "Healthcare": 0.6, "Insurance": 0.7, "Oil Services": 1.3, "Auto": 1.4}},
    "Recession":       {"probability": 0.08, "market_dd": -0.30, "valuation_matters": True,  "mean_reversion": True,  "profit_matters": True,  "liquidity_matters": True,  "sector_multipliers": {"Semiconductor": 1.3, "Hardware": 1.3, "Software": 1.1, "Cybersecurity": 1.0, "Cloud Infra": 1.2, "Streaming": 0.9, "E-commerce": 1.1, "Financials": 1.4, "Banking": 1.3, "Pharma": 0.7, "Healthcare": 0.7, "Insurance": 0.9, "Oil Services": 1.2, "Auto": 1.3}},
    "Stagflation":     {"probability": 0.03, "market_dd": -0.40, "valuation_matters": True,  "mean_reversion": True,  "profit_matters": True,  "liquidity_matters": True,  "sector_multipliers": {"Semiconductor": 1.3, "Hardware": 1.2, "Software": 1.2, "Cybersecurity": 1.2, "Cloud Infra": 1.3, "Streaming": 1.0, "E-commerce": 1.1, "Financials": 1.1, "Banking": 1.1, "Pharma": 0.8, "Healthcare": 0.8, "Insurance": 0.9, "Oil Services": 0.7, "Auto": 1.1}},
    "Mild Correction": {"probability": 0.30, "market_dd": -0.10, "valuation_matters": False, "mean_reversion": True,  "profit_matters": False, "liquidity_matters": False, "sector_multipliers": {"Semiconductor": 1.1, "Hardware": 1.1, "Software": 1.0, "Cybersecurity": 1.1, "Cloud Infra": 1.1, "Streaming": 1.0, "E-commerce": 1.0, "Financials": 1.0, "Banking": 1.0, "Pharma": 0.9, "Healthcare": 0.9, "Insurance": 0.9, "Oil Services": 0.9, "Auto": 1.0}},
    "Semi Downturn":   {"probability": 0.15, "market_dd": -0.08, "valuation_matters": True,  "mean_reversion": True,  "profit_matters": False, "liquidity_matters": False, "sector_multipliers": {"Semiconductor": 4.0, "Hardware": 2.5, "Software": 0.5, "Cybersecurity": 0.5, "Cloud Infra": 0.7, "Streaming": 0.4, "E-commerce": 0.4, "Financials": 0.2, "Banking": 0.3, "Pharma": 0.2, "Healthcare": 0.2, "Insurance": 0.2, "Oil Services": 0.3, "Auto": 0.3}},
}

SECTOR_MAP = {
    "Semiconductors": "Semiconductor", "Semiconductor Equipment & Materials": "Semiconductor",
    "Software - Infrastructure": "Software", "Software - Application": "Software",
    "Information Technology Services": "Software", "Internet Content & Information": "Software",
    "Communication Equipment": "Hardware", "Computer Hardware": "Hardware",
    "Consumer Electronics": "Hardware", "Electronic Components": "Hardware",
    "Banks - Diversified": "Banking", "Banks - Regional": "Banking",
    "Capital Markets": "Financials", "Financial Data & Stock Exchanges": "Financials",
    "Credit Services": "Financials",
    "Drug Manufacturers - General": "Pharma", "Drug Manufacturers - Specialty & Generic": "Pharma",
    "Biotechnology": "Pharma",
    "Health Care Plans": "Healthcare", "Health Care Providers": "Healthcare",
    "Medical Instruments & Supplies": "Healthcare", "Medical Devices": "Healthcare",
    "Insurance - Diversified": "Insurance", "Insurance - Specialty": "Insurance",
    "Oil & Gas Equipment & Services": "Oil Services", "Oil & Gas Drilling": "Oil Services",
    "Auto Manufacturers": "Auto", "Auto Parts": "Auto",
    "Internet Retail": "E-commerce", "Specialty Retail": "E-commerce",
    "Entertainment": "Streaming", "Leisure": "E-commerce",
}

TICKER_SECTOR_OVERRIDE = {
    "RBRK": "Cybersecurity", "CRWV": "Cloud Infra", "GRAB": "E-commerce",
    "SE": "E-commerce", "MELI": "E-commerce", "OSCR": "Insurance",
    "SOFI": "Banking", "CNC": "Healthcare", "HROW": "Pharma",
    "NFLX": "Streaming", "SPOT": "Streaming", "UBER": "Software", "RDDT": "Software",
}


def safe_get(row, key, default):
    val = row.get(key, default)
    return default if pd.isna(val) else val


def shock_stock(row, scenario):
    """Model one stock's drawdown in a scenario (beta x sector x valuation x size x ...)."""
    bd = scenario["market_dd"]
    bm = 0.5 + 0.5 * safe_get(row, "Beta", 1.5)
    sm = scenario["sector_multipliers"].get(safe_get(row, "shock_sector", "default"), 1.0)
    if scenario.get("valuation_matters"):
        pe = safe_get(row, "P/E.1", 80)
        pe = pe if pe > 0 else 80
        vm = 1.3 if pe > 80 else (1.15 if pe > 50 else (1.0 if pe > 30 else 0.85))
    else:
        vm = 1.0
    if scenario.get("liquidity_matters"):
        mc = safe_get(row, "Market Cap", 1e10) / 1e9
        szm = 1.3 if mc < 10 else (1.1 if mc < 50 else 1.0)
    else:
        szm = 1.0
    pm = (0.8 + 0.4 * safe_get(row, "52w_position", 0.5)) if scenario.get("mean_reversion") else 1.0
    prm = 1.3 if (scenario.get("profit_matters") and not safe_get(row, "is_profitable", True)) else 1.0
    return max(bd * bm * sm * vm * szm * pm * prm, -0.95)


def run_shock(df, portfolios):
    """Stress-test each portfolio across all scenarios. Returns {name: metrics}."""
    results = {}
    for pn, sl in portfolios.items():
        pdf = df[df["Stock"].isin(sl)]
        n = len(pdf)
        if n == 0:
            continue
        r = {"N": n, "Growth": pdf["blended_growth"].mean(),
             "Beta": pdf["Beta"].fillna(1.5).mean(),
             "Stretch": pdf["stretch_score"].mean(), "52w": pdf["52w_position"].mean(),
             "R/R": pdf["Average Outcome"].mean()}
        pp = pdf[pdf["peg_proxy"].notna() & (pdf["peg_proxy"] > 0)]
        r["PEG"] = pp["peg_proxy"].mean() if len(pp) > 0 else np.nan
        r["Semi%"] = pdf["shock_sector"].eq("Semiconductor").mean() * 100
        r["PreProfit%"] = (~pdf["is_profitable"]).mean() * 100
        r["G/B"] = r["Growth"] / r["Beta"] if r["Beta"] > 0 else 0
        for sn, sp in SHOCK_SCENARIOS.items():
            r[sn] = np.nanmean([shock_stock(row, sp) for _, row in pdf.iterrows()]) * 100
        r["SHOCK Score"] = sum(SHOCK_SCENARIOS[s]["probability"] * r[s] for s in SHOCK_SCENARIOS)
        gp = r["Growth"] * 100
        r["Edge Ratio"] = gp / abs(r["SHOCK Score"]) if r["SHOCK Score"] != 0 else 0
        dd = r["SHOCK Score"] / 100
        rn = -dd / (1 + dd) * 100
        r["Recovery Needed"] = rn
        r["1Y Recovery"] = "YES" if rn < gp else ("MAYBE" if rn < gp * 2 else "NO")
        results[pn] = r
    return results
