"""CSV export and run-history tracking."""
from datetime import datetime
from pathlib import Path

import pandas as pd

WATCHLIST_COLS = [
    "Stock", "prism_rank", "prism_score", "dim_growth", "dim_value", "dim_momentum",
    "dim_resilience", "2026 Growth Rate", "2027 Growth Rate", "blended_growth",
    "P/E.1", "peg_proxy", "Beta", "52w_position", "stretch_score", "Average Outcome",
    "Market Cap", "fragile_flag", "fragile_points", "shock_sector", "Industry",
    "Price", "Target Mean",
]


def export_csvs(df, portfolios, shock_results, quality_df, cfg, run_date=None):
    """Write the four run CSVs into a dated outputs dir and append run history.

    Returns the output directory path.
    """
    run_date = run_date or datetime.now().strftime("%Y-%m-%d")
    out = Path(cfg["output_dir"]) / run_date
    out.mkdir(parents=True, exist_ok=True)

    vc = [c for c in WATCHLIST_COLS if c in df.columns]
    df[vc].sort_values("prism_rank").to_csv(out / "prism_scored_watchlist.csv", index=False)

    rows = []
    for pn, sl in portfolios.items():
        for s in sl:
            rows.append({"Portfolio": pn, "Stock": s, "Weight": 100 / len(sl)})
    pd.DataFrame(rows).to_csv(out / "prism_portfolios.csv", index=False)

    rr = [{"Portfolio": pn, **m} for pn, m in shock_results.items()]
    pd.DataFrame(rr).to_csv(out / "prism_report.csv", index=False)

    quality_df.to_csv(out / "prism_data_quality.csv", index=False)

    append_history(df, run_date, Path(cfg["output_dir"]) / "history.csv")

    print(f"\n  Exported to {out}/: prism_scored_watchlist.csv, prism_portfolios.csv, "
          f"prism_report.csv, prism_data_quality.csv")
    return out


def append_history(df, run_date, history_path):
    """Append this run's scores to the long-form history file (one row per stock)."""
    hist_cols = ["Stock", "prism_rank", "prism_score", "dim_growth", "dim_value",
                 "dim_momentum", "dim_resilience", "blended_growth", "Price"]
    snap = df[[c for c in hist_cols if c in df.columns]].copy()
    snap.insert(0, "Date", run_date)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        prev = pd.read_csv(history_path)
        prev = prev[prev["Date"] != run_date]  # re-runs on the same day replace
        snap = pd.concat([prev, snap], ignore_index=True)
    snap.to_csv(history_path, index=False)
