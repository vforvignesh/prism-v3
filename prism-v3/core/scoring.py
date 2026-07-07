"""
PRISM v3 — Scoring Engine
Composite score = 40% Fundamental + 30% Risk + 30% Technical
"""

import numpy as np
import pandas as pd


def safe_get(val, default):
    """Return default if val is None, NaN, or missing."""
    if val is None:
        return default
    try:
        if np.isnan(val):
            return default
    except (TypeError, ValueError):
        pass
    return val


# ---------------------------------------------------------------------------
#  Fundamental Scoring
# ---------------------------------------------------------------------------

def compute_relative_peg(df: pd.DataFrame, index_pe: float = 22.0, index_growth: float = 0.13) -> pd.DataFrame:
    """
    Compute Relative PEG = (Stock PE / Stock Growth) / (Index PE / Index Growth)
    Lower = better value per unit of growth.
    Also computes Alpha Score = Growth Delta / Relative PE.
    """
    df = df.copy()

    index_peg = index_pe / (index_growth * 100) if index_growth > 0 else 1.0

    pegs = []
    alphas = []
    excess_returns = []

    for _, row in df.iterrows():
        pe = safe_get(row.get("PE (Fwd)"), safe_get(row.get("PE (TTM)"), 25.0))
        growth = safe_get(row.get("Growth"), 0.04)
        growth_pct = growth * 100

        # PEG ratio
        if growth_pct > 0 and pe > 0:
            stock_peg = pe / growth_pct
        else:
            stock_peg = 99.0  # penalty for negative/zero growth

        # Relative PEG (lower = better)
        rel_peg = stock_peg / index_peg if index_peg > 0 else stock_peg
        pegs.append(rel_peg)

        # Alpha Score = Growth Delta / Relative PE
        growth_delta = growth - index_growth
        rel_pe = pe / index_pe if index_pe > 0 else 1.0
        alpha = growth_delta / rel_pe if rel_pe > 0 else 0
        alphas.append(alpha)

        # Excess Return = Growth - (PE / Index PE) * Index Growth
        excess = growth_pct - rel_pe * (index_growth * 100)
        excess_returns.append(excess)

    df["Relative PEG"] = pegs
    df["Alpha Score"] = alphas
    df["Excess Return (%)"] = excess_returns

    return df


def score_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """Score fundamentals 0-100. Lower Rel PEG + higher Alpha = higher score."""
    df = df.copy()

    scores = []
    for _, row in df.iterrows():
        score = 50  # baseline

        # Relative PEG component (lower = better)
        rpeg = safe_get(row.get("Relative PEG"), 1.0)
        if rpeg < 0.3:
            score += 30
        elif rpeg < 0.5:
            score += 20
        elif rpeg < 0.8:
            score += 10
        elif rpeg < 1.0:
            score += 5
        elif rpeg < 1.5:
            score -= 5
        elif rpeg < 2.0:
            score -= 15
        else:
            score -= 25

        # Alpha Score component
        alpha = safe_get(row.get("Alpha Score"), 0)
        score += min(20, max(-20, alpha * 100))

        # Excess Return bonus
        excess = safe_get(row.get("Excess Return (%)"), 0)
        if excess > 10:
            score += 10
        elif excess > 5:
            score += 5
        elif excess < -5:
            score -= 10

        scores.append(max(0, min(100, score)))

    df["Fundamental Score"] = scores
    return df


# ---------------------------------------------------------------------------
#  Risk / SHOCK Scoring
# ---------------------------------------------------------------------------

def shock_stock(row: pd.Series, scenario: dict, sector_sensitivity: dict) -> float:
    """Compute expected drawdown for a single stock under a shock scenario."""
    base_dd = scenario["market_dd"]

    beta = safe_get(row.get("Beta"), 1.5)
    beta_mult = 0.5 + 0.5 * beta

    sector = safe_get(row.get("Sector"), "default")
    sector_mult = sector_sensitivity.get(sector, sector_sensitivity.get("default", 1.0))

    # Valuation penalty for expensive stocks
    val_mult = 1.0
    if scenario.get("valuation_matters"):
        pe = safe_get(row.get("PE (Fwd)"), safe_get(row.get("PE (TTM)"), 25.0))
        pe = pe if pe > 0 else 80
        if pe > 40:
            val_mult = 1.0 + (pe - 40) / 80
        elif pe < 15:
            val_mult = 0.8

    # 52w position — stocks near highs fall more
    pos = safe_get(row.get("52w Position"), 0.5)
    pos_mult = 0.8 + 0.4 * pos  # 0.8 at lows, 1.2 at highs

    dd = base_dd * beta_mult * sector_mult * val_mult * pos_mult
    return dd


def compute_shock_scores(df: pd.DataFrame, scenarios: dict, sector_sensitivity: dict) -> pd.DataFrame:
    """Compute SHOCK scores for all stocks."""
    df = df.copy()

    for sname, scenario in scenarios.items():
        dds = []
        for _, row in df.iterrows():
            dd = shock_stock(row, scenario, sector_sensitivity)
            dds.append(dd * 100)  # as percentage
        df[f"SHOCK: {sname}"] = dds

    # Weighted SHOCK Score
    shock_cols = [c for c in df.columns if c.startswith("SHOCK: ")]
    shock_scores = []
    for _, row in df.iterrows():
        ws = 0
        for sname, scenario in scenarios.items():
            col = f"SHOCK: {sname}"
            ws += scenario["probability"] * row[col]
        shock_scores.append(ws)
    df["SHOCK Score"] = shock_scores

    return df


def compute_edge_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Edge Ratio = Growth% / |SHOCK Score|. Higher = better risk/reward."""
    df = df.copy()

    edges = []
    recoveries = []
    recovery_flags = []

    for _, row in df.iterrows():
        growth_pct = safe_get(row.get("Growth"), 0.04) * 100
        shock = safe_get(row.get("SHOCK Score"), -10)

        # Edge Ratio
        edge = growth_pct / abs(shock) if shock != 0 else 0
        edges.append(edge)

        # Recovery needed
        dd_pct = shock / 100
        rn = -dd_pct / (1 + dd_pct) * 100 if dd_pct > -1 else 999
        recoveries.append(rn)

        # Can it recover in 1 year?
        if rn < growth_pct:
            recovery_flags.append("YES")
        elif rn < growth_pct * 2:
            recovery_flags.append("MAYBE")
        else:
            recovery_flags.append("NO")

    df["Edge Ratio"] = edges
    df["Recovery Needed (%)"] = recoveries
    df["1Y Recovery"] = recovery_flags

    return df


def score_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Score risk 0-100. Better Edge Ratio + lighter SHOCK = higher score."""
    df = df.copy()

    scores = []
    for _, row in df.iterrows():
        score = 50

        # SHOCK Score component (less negative = better)
        shock = safe_get(row.get("SHOCK Score"), -10)
        if shock > -5:
            score += 20
        elif shock > -10:
            score += 10
        elif shock > -15:
            score += 0
        elif shock > -20:
            score -= 10
        else:
            score -= 20

        # Edge Ratio component
        edge = safe_get(row.get("Edge Ratio"), 0)
        if edge > 3:
            score += 20
        elif edge > 2:
            score += 15
        elif edge > 1.5:
            score += 10
        elif edge > 1:
            score += 5
        elif edge < 0.5:
            score -= 15

        # Recovery flag
        rec = row.get("1Y Recovery", "NO")
        if rec == "YES":
            score += 10
        elif rec == "MAYBE":
            score += 0
        else:
            score -= 10

        scores.append(max(0, min(100, score)))

    df["Risk Score"] = scores
    return df


# ---------------------------------------------------------------------------
#  Fragile Flags
# ---------------------------------------------------------------------------

def compute_fragile_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag stocks with structural vulnerabilities."""
    df = df.copy()

    flags = []
    for _, row in df.iterrows():
        f = []
        pe = safe_get(row.get("PE (Fwd)"), safe_get(row.get("PE (TTM)"), 25))

        # Pre-profit or negative PE
        if pe < 0 or pe > 100:
            f.append("PRE-PROFIT")

        # Extreme valuation
        if pe > 50:
            f.append("EXPENSIVE")

        # High beta
        beta = safe_get(row.get("Beta"), 1.0)
        if beta > 1.8:
            f.append("HIGH-BETA")

        # Near 52w low
        pos = safe_get(row.get("52w Position"), 0.5)
        if pos < 0.2:
            f.append("NEAR-LOW")
        elif pos > 0.95:
            f.append("NEAR-HIGH")

        # Negative growth
        growth = safe_get(row.get("Growth"), 0)
        if growth < 0:
            f.append("NEG-GROWTH")

        # Low data confidence
        conf = row.get("Growth Confidence", "")
        if "DEFAULT" in str(conf):
            f.append("LOW-DATA")

        flags.append(", ".join(f) if f else "—")

    df["Fragile Flags"] = flags
    return df


# ---------------------------------------------------------------------------
#  Composite PRISM Score
# ---------------------------------------------------------------------------

def compute_prism_score(df: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """
    Composite PRISM Score = weighted sum of Fundamental, Risk, Technical scores.
    Default: 40% Fundamental + 30% Risk + 30% Technical
    """
    if weights is None:
        weights = {"fundamental": 0.40, "risk": 0.30, "technical": 0.30}

    df = df.copy()

    prism_scores = []
    for _, row in df.iterrows():
        fund = safe_get(row.get("Fundamental Score"), 50)
        risk = safe_get(row.get("Risk Score"), 50)
        tech = safe_get(row.get("technical_score"), 50)

        composite = (
            fund * weights["fundamental"] +
            risk * weights["risk"] +
            tech * weights["technical"]
        )
        prism_scores.append(round(composite, 1))

    df["PRISM Score"] = prism_scores

    # Rank
    df["Rank"] = df["PRISM Score"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("Rank")

    return df


# ---------------------------------------------------------------------------
#  Master Scoring Pipeline
# ---------------------------------------------------------------------------

def run_scoring(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Run the full PRISM v3 scoring pipeline."""
    weights = settings.get("scoring_weights", {})
    scenarios = settings.get("shock_scenarios", {})
    sector_sens = settings.get("sector_shock_sensitivity", {})

    # Step 1: Fundamentals
    df = compute_relative_peg(df)
    df = score_fundamentals(df)

    # Step 2: Risk / SHOCK
    df = compute_shock_scores(df, scenarios, sector_sens)
    df = compute_edge_ratio(df)
    df = score_risk(df)

    # Step 3: Fragile flags
    df = compute_fragile_flags(df)

    # Step 4: Composite
    df = compute_prism_score(df, weights)

    return df
