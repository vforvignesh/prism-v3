"""
PRISM v3 — Weekly Portfolio Health Check
Re-scores all holdings, flags below-threshold, saves snapshots.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import requests


def send_telegram(message: str):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print(f"[ALERT] {message}")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"})


def run_health_check():
    # Monkey-patch streamlit cache for non-streamlit context
    import streamlit as st
    st.cache_data = lambda **kwargs: lambda f: f

    from core.data_pipeline import fetch_all_data
    from core.scoring import run_scoring

    config_dir = Path(__file__).parent.parent / "config"
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    with open(config_dir / "settings.json") as f:
        settings = json.load(f)
    with open(config_dir / "watchlist.json") as f:
        watchlist = json.load(f)

    fmp_key = os.environ.get("FMP_API_KEY", settings["api_keys"]["fmp"])
    fh_key = os.environ.get("FINNHUB_API_KEY", settings["api_keys"]["finnhub"])

    tickers = list(watchlist["tickers"].keys())
    print(f"Running weekly health check for {len(tickers)} tickers...")

    # Fetch and score
    df = fetch_all_data(tickers, fmp_key, fh_key,
                        growth_overrides=watchlist.get("growth_overrides", {}))
    scored = run_scoring(df, settings)

    # Save snapshot
    today = datetime.now().strftime("%Y-%m-%d")
    snapshot = scored[["Ticker", "PRISM Score", "Fundamental Score", "Risk Score",
                       "technical_score", "Growth", "PE (Fwd)", "SHOCK Score", "Edge Ratio"]].copy()
    snapshot["Date"] = today

    snapshot_file = data_dir / "scores_history.json"
    history = []
    if snapshot_file.exists():
        with open(snapshot_file) as f:
            history = json.load(f)

    history.extend(snapshot.to_dict("records"))
    with open(snapshot_file, "w") as f:
        json.dump(history, f, indent=2, default=str)

    # Detect changes from last week
    alerts = []
    inclusion_threshold = 45  # PRISM score below this = flag

    for _, row in scored.iterrows():
        sym = row["Ticker"]
        prism = row["PRISM Score"]

        if prism < inclusion_threshold:
            alerts.append(
                f"🚩 <b>{sym}</b>: PRISM {prism:.1f} — below inclusion threshold ({inclusion_threshold})"
            )

    # Check for week-over-week drops
    last_week = [h for h in history if h.get("Date") != today]
    if last_week:
        last_scores = {}
        for h in last_week:
            tk = h.get("Ticker")
            if tk:
                last_scores[tk] = h.get("PRISM Score", 50)

        for _, row in scored.iterrows():
            sym = row["Ticker"]
            if sym in last_scores:
                delta = row["PRISM Score"] - last_scores[sym]
                if delta < -10:
                    alerts.append(
                        f"📉 <b>{sym}</b>: PRISM dropped {delta:.1f} pts this week "
                        f"({last_scores[sym]:.1f} → {row['PRISM Score']:.1f})"
                    )

    # Build summary
    avg_prism = scored["PRISM Score"].mean()
    avg_shock = scored["SHOCK Score"].mean()

    summary = (
        f"📊 <b>PRISM v3 — Weekly Health Check</b>\n"
        f"Date: {today}\n\n"
        f"Avg PRISM Score: {avg_prism:.1f}\n"
        f"Avg SHOCK Score: {avg_shock:.1f}%\n"
        f"Stocks tracked: {len(tickers)}\n"
    )

    if alerts:
        summary += f"\n⚠️ <b>{len(alerts)} flags:</b>\n" + "\n".join(alerts)
    else:
        summary += "\n✅ All holdings above threshold. Portfolio healthy."

    # Top 3 and bottom 3
    summary += "\n\n<b>Top 3:</b>"
    for _, row in scored.head(3).iterrows():
        summary += f"\n  #{int(row['Rank'])} {row['Ticker']} — PRISM {row['PRISM Score']:.1f}"

    summary += "\n\n<b>Bottom 3:</b>"
    for _, row in scored.tail(3).iterrows():
        summary += f"\n  #{int(row['Rank'])} {row['Ticker']} — PRISM {row['PRISM Score']:.1f}"

    send_telegram(summary)
    print(summary.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))


if __name__ == "__main__":
    run_health_check()
