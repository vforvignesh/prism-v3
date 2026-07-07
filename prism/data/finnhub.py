"""Finnhub client: EPS estimates used to cross-validate FMP."""
import logging

import numpy as np
import requests

from .common import map_fy_to_calendar

log = logging.getLogger("prism.finnhub")


def fetch_finnhub_estimates(symbol, api_key, session=None):
    """Finnhub annual EPS estimates (may require paid plan). Returns list or None."""
    http = session or requests
    url = (
        "https://finnhub.io/api/v1/stock/eps-estimate"
        f"?symbol={symbol}&freq=annual&token={api_key}"
    )
    try:
        resp = http.get(url, timeout=10)
    except requests.RequestException as e:
        log.warning("Finnhub %s: request failed: %s", symbol, e)
        return None
    if resp.status_code != 200:
        log.warning("Finnhub %s: HTTP %s: %s", symbol, resp.status_code, resp.text[:200])
        return None
    try:
        data = resp.json()
    except ValueError:
        log.warning("Finnhub %s: non-JSON response: %s", symbol, resp.text[:200])
        return None
    if not data or "data" not in data or len(data.get("data", [])) == 0:
        log.info("Finnhub %s: empty result", symbol)
        return None
    return data["data"]


def calc_growth_from_finnhub(fh_data):
    """Derive calendar-year 2026/2027 EPS growth from Finnhub estimate entries."""
    if not fh_data:
        return np.nan, np.nan
    cal_eps = {}
    for entry in fh_data:
        cy = map_fy_to_calendar(entry.get("date", ""))
        eps = entry.get("epsAvg")
        if cy and eps is not None:
            cal_eps[cy] = eps
    e25, e26, e27 = cal_eps.get(2025), cal_eps.get(2026), cal_eps.get(2027)
    g26 = (e26 / e25) - 1 if (e25 and e26 and e25 > 0) else np.nan
    g27 = (e27 / e26) - 1 if (e26 and e27 and e26 > 0) else np.nan
    return g26, g27
