"""
PRISM v3 — Macro Scenario Monitor
Tracks VIX, 10Y yield, SOX index for stress signals.
"""

import os
import sys
import json
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
import numpy as np


def send_telegram(message: str):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print(f"[ALERT] {message}")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"})


def check_macro():
    config_dir = Path(__file__).parent.parent / "config"
    with open(config_dir / "settings.json") as f:
        settings = json.load(f)

    thresholds = settings["alert_thresholds"]
    alerts = []

    # --- VIX ---
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d")
        if not hist.empty:
            vix_now = hist["Close"].iloc[-1]
            if vix_now >= thresholds["vix_stress"]:
                alerts.append(f"🚨 <b>VIX STRESS: {vix_now:.1f}</b> (threshold: {thresholds['vix_stress']})")
            elif vix_now >= thresholds["vix_elevated"]:
                alerts.append(f"⚠️ <b>VIX ELEVATED: {vix_now:.1f}</b> (threshold: {thresholds['vix_elevated']})")
    except Exception as e:
        print(f"VIX error: {e}")

    # --- 10Y Yield ---
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="10d")
        if len(hist) >= 5:
            yield_now = hist["Close"].iloc[-1]
            yield_week_ago = hist["Close"].iloc[-5]
            move_bps = (yield_now - yield_week_ago) * 100
            if abs(move_bps) >= thresholds["yield_10y_weekly_move_bps"]:
                direction = "UP" if move_bps > 0 else "DOWN"
                alerts.append(
                    f"📊 <b>10Y Yield {direction}: {yield_now:.2f}%</b> "
                    f"({move_bps:+.0f}bps this week, threshold: ±{thresholds['yield_10y_weekly_move_bps']}bps)"
                )
    except Exception as e:
        print(f"TNX error: {e}")

    # --- SOX (Semiconductor Index) ---
    try:
        sox = yf.Ticker("^SOX")
        hist = sox.history(period="1mo")
        if not hist.empty:
            sox_now = hist["Close"].iloc[-1]
            sox_peak = hist["Close"].max()
            drop_pct = (sox_now - sox_peak) / sox_peak * 100
            if abs(drop_pct) >= thresholds["sox_drop_from_peak_pct"]:
                alerts.append(
                    f"🔻 <b>SOX INDEX DOWN {drop_pct:.1f}% from 1-month peak</b> "
                    f"(now: {sox_now:.0f}, peak: {sox_peak:.0f})"
                )
    except Exception as e:
        print(f"SOX error: {e}")

    if alerts:
        # Map to SHOCK scenarios
        scenario_note = "\n\n📋 <i>Scenarios becoming more probable:</i>"
        for alert in alerts:
            if "VIX" in alert:
                scenario_note += "\n  → Rate Shock, Recession"
            if "10Y" in alert and "UP" in alert:
                scenario_note += "\n  → Rate Shock, Stagflation"
            if "SOX" in alert:
                scenario_note += "\n  → Semi Downturn, AI Bubble Pop"

        header = "🌍 <b>PRISM v3 — Macro Monitor</b>\n\n"
        message = header + "\n".join(alerts) + scenario_note
        send_telegram(message)
        print(f"Sent {len(alerts)} macro alerts")
    else:
        print("No macro alerts triggered")


if __name__ == "__main__":
    check_macro()
