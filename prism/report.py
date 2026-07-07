"""Console report generation."""
from datetime import datetime

import numpy as np
import pandas as pd

from .shock import SHOCK_SCENARIOS


def print_header(text, width=120, char="="):
    print(f"\n{char*width}")
    print(f"  {text}")
    print(f"{char*width}")


def generate_report(df, portfolios, fragile_stocks, replacements, shock_results, cfg):
    print_header("PRISM — Scored Watchlist Report")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Stocks: {len(df)}")
    print(f"  Weights: G:{cfg['weight_growth']:.0%} V:{cfg['weight_value']:.0%} "
          f"M:{cfg['weight_momentum']:.0%} R:{cfg['weight_resilience']:.0%}")

    print_header("PRISM RANKINGS - Top 30")
    top30 = df.head(30)
    print(f"\n {'Rk':>3} {'Stock':>6} {'Score':>6} | {'Grw':>4} {'Val':>4} {'Mom':>4} {'Res':>4} | "
          f"{'26E':>5} {'27E':>5} {'Blnd':>5} | {'P/E':>6} {'PEG':>6} {'Beta':>5} | "
          f"{'52w%':>5} {'Stch':>4} {'R/R':>6} | {'Frag':>9}")
    print(f" {'-'*112}")
    for _, r in top30.iterrows():
        pe_s = f"{r['P/E.1']:6.1f}" if pd.notna(r['P/E.1']) and r['P/E.1'] > 0 else "   N/A"
        pg_s = f"{r['peg_proxy']:6.2f}" if pd.notna(r['peg_proxy']) else "   N/A"
        b_s = f"{r['Beta']:5.2f}" if pd.notna(r['Beta']) else "  N/A"
        fl = f"  F({r['fragile_points']:.1f})" if r['fragile_flag'] else f"  ({r['fragile_points']:.1f})"
        print(f" {int(r['prism_rank']):3d} {r['Stock']:>6} {r['prism_score']:6.1f} | "
              f"{r['dim_growth']:4.0f} {r['dim_value']:4.0f} {r['dim_momentum']:4.0f} "
              f"{r['dim_resilience']:4.0f} | {r['2026 Growth Rate']:4.0%} "
              f"{r['2027 Growth Rate']:4.0%} {r['blended_growth']:4.0%} | "
              f"{pe_s} {pg_s} {b_s} | {r['52w_position']:4.0%} {r['stretch_score']:4.0f} "
              f"{r['Average Outcome']:6.3f} | {fl}")

    if fragile_stocks:
        print_header("FRAGILE FLAGS", char="-")
        print(f"  Flagged ({len(fragile_stocks)}): {fragile_stocks}")
        if replacements:
            print(f"  Replacements ({len(replacements)}): {replacements}")

    for pname, sl in portfolios.items():
        nn = len(sl)
        wt = 100 / nn if nn > 0 else 0
        print_header(f"{pname} - {nn} stocks x {wt:.1f}%", char="-")
        pdf = df[df["Stock"].isin(sl)].sort_values("prism_score", ascending=False)
        for _, r in pdf.iterrows():
            pe_s = f"{r['P/E.1']:.0f}x" if pd.notna(r['P/E.1']) and r['P/E.1'] > 0 else " N/A"
            b_s = f"{r['Beta']:.2f}" if pd.notna(r['Beta']) else " N/A"
            print(f"    {r['Stock']:>6} | Score {r['prism_score']:5.1f} | "
                  f"G:{r['blended_growth']:4.0%} | PE:{pe_s:>5} | B:{b_s:>5} | "
                  f"52w:{r['52w_position']:3.0%} | Str:{r['stretch_score']:3.0f} | "
                  f"R/R:{r['Average Outcome']:.3f} | {r['shock_sector']}")

    print_header("SHOCK STRESS TEST & EDGE RATIO")
    pnames = list(shock_results.keys())
    print(f"\n {'Metric':20s}", end="")
    for pn in pnames:
        print(f" | {pn:>14}", end="")
    print(f" | {'SPY':>8}")
    print(f" {'-'*(25+17*len(pnames))}")
    for label, key, fmt in [("Positions", "N", "{:.0f}"), ("Blended Growth", "Growth", "{:.0%}"),
                            ("PEG", "PEG", "{:.2f}x"), ("Beta", "Beta", "{:.2f}"),
                            ("Stretch", "Stretch", "{:.0f}"), ("52w Position", "52w", "{:.0%}"),
                            ("R/R Ratio", "R/R", "{:.3f}"), ("Growth/Beta", "G/B", "{:.3f}"),
                            ("Semi%", "Semi%", "{:.0f}%"), ("Pre-Profit%", "PreProfit%", "{:.0f}%")]:
        print(f" {label:20s}", end="")
        for pn in pnames:
            v = shock_results[pn].get(key, np.nan)
            s = fmt.format(v) if pd.notna(v) else "N/A"
            print(f" | {s:>14}", end="")
        print(f" | {'':>8}")
    print()
    for sn in SHOCK_SCENARIOS:
        print(f" {sn:20s}", end="")
        vals = [shock_results[pn].get(sn, 0) for pn in pnames]
        best = max(vals)
        for v in vals:
            m = " <<" if v == best and len(pnames) > 1 else ""
            print(f" | {v:+10.0f}%{m:>3}", end="")
        print(f" | {SHOCK_SCENARIOS[sn]['market_dd']*100:+7.0f}%")
    print()
    spy_s = sum(s["probability"] * s["market_dd"] * 100 for s in SHOCK_SCENARIOS.values())
    print(f" {'SHOCK Score':20s}", end="")
    for pn in pnames:
        print(f" | {shock_results[pn]['SHOCK Score']:+10.1f}%    ", end="")
    print(f" | {spy_s:+7.1f}%")
    print(f" {'* EDGE RATIO *':20s}", end="")
    evs = [shock_results[pn]["Edge Ratio"] for pn in pnames]
    be = max(evs)
    for i, pn in enumerate(pnames):
        m = " <<" if evs[i] == be and len(pnames) > 1 else ""
        print(f" | {evs[i]:10.2f}x{m:>3}", end="")
    print(f" | {10/abs(spy_s):7.2f}x" if spy_s != 0 else " |    N/A")
    print(f"\n {'Recovery (1Y)':20s}", end="")
    for pn in pnames:
        print(f" | {shock_results[pn]['1Y Recovery']:>14}", end="")
    print()

    bp = max(shock_results, key=lambda x: shock_results[x]["Edge Ratio"])
    b = shock_results[bp]
    print_header(f"RECOMMENDATION: {bp}")
    print(f"  Edge Ratio: {b['Edge Ratio']:.2f}x | Growth: {b['Growth']:.0%} | "
          f"SHOCK: {b['SHOCK Score']:+.1f}% | Stocks: {b['N']}")
    print(f"  Holdings: {portfolios[bp]}")
