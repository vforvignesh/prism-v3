"""PRISM 4-dimension scoring engine: Growth / Value / Momentum / Resilience."""
import numpy as np
import pandas as pd

from .shock import SECTOR_MAP, TICKER_SECTOR_OVERRIDE


def pct_rank(series, ascending=True):
    if ascending:
        return series.rank(pct=True, na_option="keep") * 100
    return ((1 - series.rank(pct=True, na_option="keep")) * 100
            + (100 / len(series)))


def score_growth(df, cfg):
    cap = cfg["growth_cap"]
    df["g26_capped"] = df["2026 Growth Rate"].clip(upper=cap)
    df["g27_capped"] = df["2027 Growth Rate"].clip(upper=cap)
    df["blended_growth"] = (df["g26_capped"] * cfg["blend_2026"]
                            + df["g27_capped"] * cfg["blend_2027"])
    df["growth_durability"] = np.where(
        df["g26_capped"] > 0.05, (df["g27_capped"] / df["g26_capped"]).clip(0, 3), 0)
    return (pct_rank(df["g26_capped"]) * 0.40
            + pct_rank(df["g27_capped"]) * 0.15
            + pct_rank(df["blended_growth"]) * 0.30
            + pct_rank(df["growth_durability"]) * 0.15)


def score_value(df):
    df["is_profitable"] = (df["P/E.1"] > 0) & (df["P/E.1"].notna())
    df["peg_proxy"] = np.where(
        (df["P/E.1"] > 0) & (df["blended_growth"] > 0.01),
        df["P/E.1"] / (df["blended_growth"] * 100), np.nan)
    peg_s = pct_rank(df["peg_proxy"], ascending=False)
    pe_s = pct_rank(df["P/E.1"].where(df["is_profitable"]), ascending=False)
    return np.where(
        df["is_profitable"],
        peg_s.fillna(30) * 0.65 + pe_s.fillna(30) * 0.35,
        np.where(df["blended_growth"] > 0.5, 55,
                 np.where(df["blended_growth"] > 0.2, 35, 15)))


def score_momentum(df):
    df["52w_position"] = ((df["Price"] - df["52 Week Low"])
                          / (df["52 Week High"] - df["52 Week Low"]))
    df["52w_range_width"] = ((df["52 Week High"] - df["52 Week Low"])
                             / df["52 Week Low"])
    s = pd.Series(100.0, index=df.index)
    s -= np.where(df["52w_position"] > 0.90, 20,
                  np.where(df["52w_position"] > 0.80, 10,
                           np.where(df["52w_position"] > 0.70, 5, 0)))
    pf = df["peg_proxy"].fillna(1.0)
    s -= np.where(pf > 3.0, 15, np.where(pf > 2.0, 10, np.where(pf > 1.5, 5, 0)))
    bf = df["Beta"].fillna(1.5)
    s -= np.where(bf > 3.0, 10, np.where(bf > 2.5, 7, np.where(bf > 2.0, 4, 0)))
    s -= np.where(df["52w_range_width"] > 5.0, 10,
                  np.where(df["52w_range_width"] > 2.0, 5,
                           np.where(df["52w_range_width"] > 1.0, 2, 0)))
    s += np.where(df["52w_position"] < 0.30, 10,
                  np.where(df["52w_position"] < 0.50, 5, 0))
    df["stretch_score"] = s.clip(0, 100)
    return (pct_rank(df["Average Outcome"]) * 0.45
            + df["stretch_score"] * 0.35
            + pct_rank(df["52w_position"]) * 0.20)


def score_resilience(df):
    df["growth_per_beta"] = (df["blended_growth"]
                             / df["Beta"].fillna(1.5).clip(lower=0.3))
    np2 = np.where(df["2026 Growth Rate"] < 0, 0,
                   np.where(df["2026 Growth Rate"] < 0.05, 40, 100))
    return (pct_rank(df["growth_per_beta"]) * 0.45
            + pct_rank(np.log10(df["Market Cap"].clip(lower=1e8))) * 0.20
            + np2 * 0.35)


def run_prism(df, cfg):
    df["dim_growth"] = score_growth(df, cfg)
    df["dim_value"] = score_value(df)
    df["dim_momentum"] = score_momentum(df)
    df["dim_resilience"] = score_resilience(df)
    df["prism_score"] = (df["dim_growth"] * cfg["weight_growth"]
                         + df["dim_value"] * cfg["weight_value"]
                         + df["dim_momentum"] * cfg["weight_momentum"]
                         + df["dim_resilience"] * cfg["weight_resilience"]).fillna(0)
    df["prism_rank"] = df["prism_score"].rank(ascending=False, method="min").astype(int)

    # Fragility points: high beta, extended price, expensive/unprofitable, wide range
    fp = pd.Series(0.0, index=df.index)
    fp += np.where(df["Beta"].fillna(0) > 2.5, 2.0,
                   np.where(df["Beta"].fillna(0) > 2.0, 1.5,
                            np.where(df["Beta"].fillna(0) > 1.8, 1.0, 0)))
    fp += np.where(df["52w_position"] > 0.90, 2.0,
                   np.where(df["52w_position"] > 0.80, 1.5,
                            np.where(df["52w_position"] > 0.70, 1.0, 0)))
    pp = np.where(df["peg_proxy"].fillna(np.nan) > 3.0, 1.5,
                  np.where(df["peg_proxy"].fillna(np.nan) > 1.5, 1.0,
                           np.where(df["peg_proxy"].fillna(0) > 1.0, 0.5, 0)))
    prp = np.where(~df["is_profitable"], 1.0, 0)
    fp += np.maximum(pp, prp)
    rw = (df["52 Week High"] - df["52 Week Low"]) / df["52 Week Low"]
    fp += np.where(rw > 3.0, 1.0, np.where(rw > 1.5, 0.5, 0))
    df["fragile_points"] = fp
    df["fragile_flag"] = fp >= cfg.get("fragile_threshold", 3.0)

    df["shock_sector"] = df["Industry"].map(SECTOR_MAP).fillna("default")
    for t, sc in TICKER_SECTOR_OVERRIDE.items():
        df.loc[df["Stock"] == t, "shock_sector"] = sc
    return df.sort_values("prism_rank")
