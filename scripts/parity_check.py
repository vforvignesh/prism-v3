#!/usr/bin/env python3
"""Parity check: original notebook scoring vs the prism package, on identical input.

Embeds the notebook's scoring/portfolio math verbatim (as of Prism_Pilot_V03)
and compares scores, ranks, and portfolio membership against the package
implementation, using today's cached fetch data.

Run:  .venv/bin/python scripts/parity_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prism.config import load_config  # noqa: E402
from prism.data.fetch import load_cached_fetch  # noqa: E402
from prism.portfolio import build_portfolios  # noqa: E402
from prism.scoring import run_prism  # noqa: E402
from prism.shock import SECTOR_MAP, TICKER_SECTOR_OVERRIDE  # noqa: E402

# ----------------------------------------------------------------------
# ORIGINAL notebook implementation (verbatim math from Prism_Pilot_V03)
# ----------------------------------------------------------------------

def nb_pct_rank(series, ascending=True):
    return series.rank(pct=True, na_option="keep") * 100 if ascending else (1 - series.rank(pct=True, na_option="keep")) * 100 + (100 / len(series))


def nb_score_growth(df, cfg):
    cap = cfg["growth_cap"]
    df["g26_capped"] = df["2026 Growth Rate"].clip(upper=cap)
    df["g27_capped"] = df["2027 Growth Rate"].clip(upper=cap)
    df["blended_growth"] = df["g26_capped"]*cfg["blend_2026"] + df["g27_capped"]*cfg["blend_2027"]
    df["growth_durability"] = np.where(df["g26_capped"]>0.05, (df["g27_capped"]/df["g26_capped"]).clip(0,3), 0)
    return nb_pct_rank(df["g26_capped"])*0.40 + nb_pct_rank(df["g27_capped"])*0.15 + nb_pct_rank(df["blended_growth"])*0.30 + nb_pct_rank(df["growth_durability"])*0.15


def nb_score_value(df):
    df["is_profitable"] = (df["P/E.1"]>0) & (df["P/E.1"].notna())
    df["peg_proxy"] = np.where((df["P/E.1"]>0)&(df["blended_growth"]>0.01), df["P/E.1"]/(df["blended_growth"]*100), np.nan)
    peg_s = nb_pct_rank(df["peg_proxy"], ascending=False)
    pe_s = nb_pct_rank(df["P/E.1"].where(df["is_profitable"]), ascending=False)
    return np.where(df["is_profitable"], peg_s.fillna(30)*0.65+pe_s.fillna(30)*0.35, np.where(df["blended_growth"]>0.5,55,np.where(df["blended_growth"]>0.2,35,15)))


def nb_score_momentum(df):
    df["52w_position"] = (df["Price"]-df["52 Week Low"])/(df["52 Week High"]-df["52 Week Low"])
    df["52w_range_width"] = (df["52 Week High"]-df["52 Week Low"])/df["52 Week Low"]
    s = pd.Series(100.0, index=df.index)
    s -= np.where(df["52w_position"]>0.90,20,np.where(df["52w_position"]>0.80,10,np.where(df["52w_position"]>0.70,5,0)))
    pf = df["peg_proxy"].fillna(1.0); s -= np.where(pf>3.0,15,np.where(pf>2.0,10,np.where(pf>1.5,5,0)))
    bf = df["Beta"].fillna(1.5); s -= np.where(bf>3.0,10,np.where(bf>2.5,7,np.where(bf>2.0,4,0)))
    s -= np.where(df["52w_range_width"]>5.0,10,np.where(df["52w_range_width"]>2.0,5,np.where(df["52w_range_width"]>1.0,2,0)))
    s += np.where(df["52w_position"]<0.30,10,np.where(df["52w_position"]<0.50,5,0))
    df["stretch_score"] = s.clip(0,100)
    return nb_pct_rank(df["Average Outcome"])*0.45 + df["stretch_score"]*0.35 + nb_pct_rank(df["52w_position"])*0.20


def nb_score_resilience(df):
    df["growth_per_beta"] = df["blended_growth"]/df["Beta"].fillna(1.5).clip(lower=0.3)
    np2 = np.where(df["2026 Growth Rate"]<0,0,np.where(df["2026 Growth Rate"]<0.05,40,100))
    return nb_pct_rank(df["growth_per_beta"])*0.45 + nb_pct_rank(np.log10(df["Market Cap"].clip(lower=1e8)))*0.20 + np2*0.35


def nb_run_prism(df, cfg):
    df["dim_growth"] = nb_score_growth(df,cfg)
    df["dim_value"] = nb_score_value(df)
    df["dim_momentum"] = nb_score_momentum(df)
    df["dim_resilience"] = nb_score_resilience(df)
    df["prism_score"] = (df["dim_growth"]*cfg["weight_growth"]+df["dim_value"]*cfg["weight_value"]+df["dim_momentum"]*cfg["weight_momentum"]+df["dim_resilience"]*cfg["weight_resilience"]).fillna(0)
    df["prism_rank"] = df["prism_score"].rank(ascending=False, method="min").astype(int)
    fp = pd.Series(0.0, index=df.index)
    fp += np.where(df["Beta"].fillna(0)>2.5,2.0,np.where(df["Beta"].fillna(0)>2.0,1.5,np.where(df["Beta"].fillna(0)>1.8,1.0,0)))
    fp += np.where(df["52w_position"]>0.90,2.0,np.where(df["52w_position"]>0.80,1.5,np.where(df["52w_position"]>0.70,1.0,0)))
    pp = np.where(df["peg_proxy"].fillna(np.nan)>3.0,1.5,np.where(df["peg_proxy"].fillna(np.nan)>1.5,1.0,np.where(df["peg_proxy"].fillna(0)>1.0,0.5,0)))
    prp = np.where(~df["is_profitable"],1.0,0)
    fp += np.maximum(pp,prp)
    rw = (df["52 Week High"]-df["52 Week Low"])/df["52 Week Low"]
    fp += np.where(rw>3.0,1.0,np.where(rw>1.5,0.5,0))
    df["fragile_points"] = fp
    df["fragile_flag"] = fp >= cfg.get("fragile_threshold",3.0)
    df["shock_sector"] = df["Industry"].map(SECTOR_MAP).fillna("default")
    for t,sc in TICKER_SECTOR_OVERRIDE.items():
        df.loc[df["Stock"]==t,"shock_sector"] = sc
    return df.sort_values("prism_rank")


# ----------------------------------------------------------------------

def main():
    cached = load_cached_fetch()
    if cached is None:
        print("No cached fetch for today — run `python run.py` first.")
        sys.exit(2)
    raw, _ = cached
    _, cfg = load_config()

    nb_df = nb_run_prism(raw.copy(), cfg).set_index("Stock")
    pkg_df = run_prism(raw.copy(), cfg).set_index("Stock")

    failures = []
    for col in ["prism_score", "dim_growth", "dim_value", "dim_momentum",
                "dim_resilience", "fragile_points", "stretch_score"]:
        diff = (nb_df[col] - pkg_df[col]).abs()
        worst = diff.max()
        if worst > 1e-9:
            failures.append(f"{col}: max abs diff {worst:.6g} "
                            f"(worst: {diff.idxmax()})")
    if not nb_df["prism_rank"].equals(pkg_df["prism_rank"]):
        failures.append("prism_rank ordering differs")

    nb_ports, _, _ = build_portfolios(nb_df.reset_index(), cfg)
    pkg_ports, _, _ = build_portfolios(pkg_df.reset_index(), cfg)
    for name in nb_ports:
        if nb_ports[name] != pkg_ports[name]:
            failures.append(f"portfolio {name} differs: {nb_ports[name]} vs {pkg_ports[name]}")

    if failures:
        print("PARITY CHECK FAILED (expected if methodology changes have been applied):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print(f"PARITY CHECK PASSED — {len(nb_df)} stocks, scores/ranks/portfolios identical.")


if __name__ == "__main__":
    main()
