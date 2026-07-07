"""Financial Modeling Prep client: analyst EPS estimates (primary growth source).

The /stable/ endpoints gate some symbols behind paid tiers (HTTP 402) while
serving others on the free tier, so a failure for one symbol says nothing
about the next — callers should keep trying per symbol.
"""
import logging

import numpy as np
import requests

from .common import map_fy_to_calendar

log = logging.getLogger("prism.fmp")


def fetch_fmp_estimates(symbol, api_key, session=None):
    """FMP analyst estimates.

    Returns (data, status) where status is one of:
      ok      - data is a non-empty list of estimate dicts
      gated   - this symbol requires a paid plan (HTTP 402)
      auth    - API key invalid/expired (HTTP 401/403) -> disable the source
      empty   - valid response, no estimates for this symbol
      error   - network/HTTP/parse failure
    """
    http = session or requests
    url = (
        "https://financialmodelingprep.com/stable/analyst-estimates"
        f"?symbol={symbol}&period=annual&page=0&limit=10&apikey={api_key}"
    )
    try:
        resp = http.get(url, timeout=10)
    except requests.RequestException as e:
        log.warning("FMP %s: request failed: %s", symbol, e)
        return None, "error"
    if resp.status_code == 402:
        log.info("FMP %s: symbol gated behind paid plan", symbol)
        return None, "gated"
    if resp.status_code in (401, 403):
        log.warning("FMP %s: auth failure HTTP %s: %s", symbol, resp.status_code, resp.text[:200])
        return None, "auth"
    if resp.status_code != 200:
        log.warning("FMP %s: HTTP %s: %s", symbol, resp.status_code, resp.text[:200])
        return None, "error"
    try:
        data = resp.json()
    except ValueError:
        log.warning("FMP %s: non-JSON response: %s", symbol, resp.text[:200])
        return None, "error"
    if isinstance(data, dict) and ("Error" in str(data) or "message" in data):
        log.warning("FMP %s: API error: %s", symbol, str(data)[:200])
        return None, "error"
    if not isinstance(data, list) or len(data) == 0:
        return None, "empty"
    return data, "ok"


def calc_growth_from_fmp(fmp_data):
    """Derive calendar-year 2026/2027 EPS growth from FMP estimate entries."""
    if not fmp_data:
        return np.nan, np.nan, np.nan, "no_data"
    cal_eps = {}
    for entry in fmp_data:
        cy = map_fy_to_calendar(entry.get("date", ""))
        # /stable/ uses "epsAvg"; the retired v3 API used "estimatedEpsAvg"
        eps = entry.get("epsAvg", entry.get("estimatedEpsAvg"))
        if cy and eps is not None:
            cal_eps[cy] = eps
    e25, e26, e27 = cal_eps.get(2025), cal_eps.get(2026), cal_eps.get(2027)
    g26 = g27 = np.nan
    if e25 and e26 and e25 > 0:
        g26 = (e26 / e25) - 1
    elif e25 and e26 and e25 < 0 and e26 > 0:
        g26 = abs(e26 - e25) / abs(e25)
    if e26 and e27 and e26 > 0:
        g27 = (e27 / e26) - 1
    elif e26 and e27 and e26 < 0 and e27 > 0:
        g27 = abs(e27 - e26) / abs(e26)
    return g26, g27, e26, "ok"
