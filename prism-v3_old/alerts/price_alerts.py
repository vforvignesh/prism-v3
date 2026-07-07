"""
PRISM v3 — Daily Price Alerts
Runs via GitHub Actions. Checks watchlist for buy-zone entries.
"""

import os
import sys
import json
import requests
import numpy as np
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf


def send_telegram(message: str):
    """Send alert via Telegram bot."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print(f"[ALERT - no Telegram configured] {message}")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"})


def check_price_alerts():
    config_dir = Path(__file__).parent.parent / "config"
    with open(config_dir / "watchlist.json") as f:
        watchlist = json.load(f)
    with open(config_dir / "settings.json") as f:
        settings = json.load(f)

    tickers = list(watchlist["tickers"].keys())
    thresholds = settings["alert_thresholds"]
    buy_zone_pct = thresholds["buy_zone_52w_pct"] / 100
    rsi_oversold = thresholds["rsi_oversold"]
    volume_spike = thresholds["volume_spike_multiplier"]

    alerts = []

    for sym in tickers:
        try:
            tk = yf.Ticker(sym)
            info = tk.info or {}
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            high52 = info.get("fiftyTwoWeekHigh")
            low52 = info.get("fiftyTwoWeekLow")

            if not price or not high52 or not low52 or high52 <= low52:
                continue

            position = (price - low52) / (high52 - low52)

            # Buy zone check
            if position < buy_zone_pct:
                alerts.append(f"🟢 <b>{sym}</b> in BUY ZONE: ${price:.2f} ({position:.0%} of 52w range)")

            # Near 52w low
            if position < 0.1:
                alerts.append(f"🔴 <b>{sym}</b> NEAR 52W LOW: ${price:.2f} (only {position:.0%} above low)")

            # Volume spike check
            hist = tk.history(period="5d", auto_adjust=True)
            if hist is not None and len(hist) >= 2:
                vol_today = hist["Volume"].iloc[-1]
                vol_avg = hist["Volume"].iloc[:-1].mean()
                if vol_avg > 0 and vol_today / vol_avg > volume_spike:
                    alerts.append(f"📊 <b>{sym}</b> VOLUME SPIKE: {vol_today/vol_avg:.1f}x average volume")

        except Exception as e:
            print(f"Error checking {sym}: {e}")

    if alerts:
        header = "🔔 <b>PRISM v3 — Daily Price Alerts</b>\n\n"
        message = header + "\n".join(alerts)
        send_telegram(message)
        print(f"Sent {len(alerts)} alerts")
    else:
        print("No price alerts triggered today")


if __name__ == "__main__":
    check_price_alerts()
