#!/usr/bin/env python3
"""Sanity-check the shock model against realized crash drawdowns.

Compares shock_stock() predictions for "COVID Crash" and "Rate Shock" against
each watchlist stock's actual max drawdown in Feb-Apr 2020 and Jan-Dec 2022.

Caveat: predictions use TODAY's beta/valuation/52w inputs, not the values the
stock had going into those crashes, and many watchlist names didn't trade
then — this is a directional calibration check, not a backtest.

Run:  .venv/bin/python scripts/calibrate_shocks.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from prism.data.fetch import load_cached_fetch  # noqa: E402
from prism.data.yahoo import yf_symbol  # noqa: E402
from prism.config import load_config  # noqa: E402
from prism.scoring import run_prism  # noqa: E402
from prism.shock import SHOCK_SCENARIOS, shock_stock  # noqa: E402

EPISODES = {
    "COVID Crash": ("2020-02-01", "2020-04-30"),
    "Rate Shock": ("2022-01-01", "2022-12-31"),
}


def max_drawdown(closes):
    peak = closes.cummax()
    return ((closes - peak) / peak).min()


def main():
    cached = load_cached_fetch()
    if cached is None:
        print("No cached fetch for today — run `python run.py` first.")
        sys.exit(2)
    raw, _ = cached
    _, cfg = load_config()
    df = run_prism(raw.copy(), cfg)

    print(f"\n{'='*84}")
    print("  SHOCK MODEL CALIBRATION — predicted vs realized crash drawdowns")
    print(f"{'='*84}")

    for scenario_name, (start, end) in EPISODES.items():
        scenario = SHOCK_SCENARIOS[scenario_name]
        rows = []
        for _, r in df.iterrows():
            sym = r["Stock"]
            try:
                closes = yf.Ticker(yf_symbol(sym)).history(
                    start=start, end=end, auto_adjust=True)["Close"].dropna()
            except Exception:
                closes = pd.Series(dtype=float)
            if len(closes) < 20:
                continue  # didn't trade through the episode
            actual = max_drawdown(closes)
            predicted = shock_stock(r, scenario)
            rows.append({"Stock": sym, "Sector": r["shock_sector"],
                         "Predicted": predicted, "Actual": actual,
                         "Error": predicted - actual})
        cal = pd.DataFrame(rows)
        if cal.empty:
            print(f"\n  {scenario_name}: no stocks with history in {start}..{end}")
            continue

        corr = cal["Predicted"].corr(cal["Actual"])
        spearman = cal["Predicted"].rank().corr(cal["Actual"].rank())
        bias = cal["Error"].mean()
        print(f"\n  {scenario_name} ({start} → {end}) — {len(cal)} stocks with history")
        print(f"  Rank correlation: {spearman:+.2f} "
              f"| Pearson: {corr:+.2f} | Mean bias: {bias:+.1%} "
              f"({'model too harsh' if bias < 0 else 'model too lenient'})")
        print(f"\n  {'Stock':>10} {'Sector':>14} {'Predicted':>10} {'Actual':>10} {'Error':>8}")
        print(f"  {'-'*56}")
        for _, c in cal.sort_values("Error").iterrows():
            print(f"  {c['Stock']:>10} {c['Sector']:>14} {c['Predicted']:>9.0%} "
                  f"{c['Actual']:>9.0%} {c['Error']:>+7.0%}")
        worst = cal.reindex(cal["Error"].abs().sort_values(ascending=False).index).head(3)
        print(f"\n  Largest misses: {worst['Stock'].tolist()} — check their sector "
              f"multipliers in SHOCK_SCENARIOS")


if __name__ == "__main__":
    main()
