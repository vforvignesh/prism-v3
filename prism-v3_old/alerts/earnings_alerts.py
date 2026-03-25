"""
PRISM v3 — Earnings Alert Monitor
Checks for upcoming earnings dates within 7-day window.
"""

import os
import sys
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf


def send_telegram(message: str):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print(f"[ALERT] {message}")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"})


def check_earnings_alerts():
    config_dir = Path(__file__).parent.parent / "config"
    with open(config_dir / "watchlist.json") as f:
        watchlist = json.load(f)

    tickers = list(watchlist["tickers"].keys())
    today = datetime.now()
    window = today + timedelta(days=7)

    alerts = []

    for sym in tickers:
        try:
            tk = yf.Ticker(sym)
            cal = tk.calendar
            if cal is not None and not cal.empty:
                # Earnings date
                if "Earnings Date" in cal.index:
                    dates = cal.loc["Earnings Date"]
                    for d in dates:
                        if isinstance(d, str):
                            d = datetime.strptime(d, "%Y-%m-%d")
                        if hasattr(d, 'date'):
                            d_date = d.date() if hasattr(d, 'date') else d
                            if today.date() <= d_date <= window.date():
                                days_out = (d_date - today.date()).days
                                alerts.append(
                                    f"📅 <b>{sym}</b> reports in {days_out} days ({d_date.strftime('%b %d')})"
                                )
        except Exception as e:
            print(f"Error checking {sym}: {e}")

    if alerts:
        header = "📈 <b>PRISM v3 — Earnings Calendar (Next 7 Days)</b>\n\n"
        message = header + "\n".join(alerts)
        send_telegram(message)
        print(f"Found {len(alerts)} upcoming earnings")
    else:
        print("No earnings in the next 7 days")


if __name__ == "__main__":
    check_earnings_alerts()
