"""Finnhub client: EPS estimates used to cross-validate FMP."""
import logging

import numpy as np
import requests

from .common import map_fy_to_calendar

log = logging.getLogger("prism.finnhub")


def fetch_finnhub_estimates(symbol, api_key, session=None):
    """Finnhub annual EPS estimates (premium-only endpoint on free keys).

    Returns (data, status) where status is one of:
      ok      - data is a non-empty list of estimate dicts
      auth    - endpoint not accessible on this key (HTTP 401/403) -> disable
      empty   - valid response, no estimates for this symbol
      error   - network/HTTP/parse failure
    """
    http = session or requests
    url = (
        "https://finnhub.io/api/v1/stock/eps-estimate"
        f"?symbol={symbol}&freq=annual&token={api_key}"
    )
    try:
        resp = http.get(url, timeout=10)
    except requests.RequestException as e:
        log.warning("Finnhub %s: request failed: %s", symbol, e)
        return None, "error"
    if resp.status_code in (401, 403):
        log.warning("Finnhub %s: no access (HTTP %s) — eps-estimate needs a paid plan",
                    symbol, resp.status_code)
        return None, "auth"
    if resp.status_code != 200:
        log.warning("Finnhub %s: HTTP %s: %s", symbol, resp.status_code, resp.text[:200])
        return None, "error"
    try:
        data = resp.json()
    except ValueError:
        log.warning("Finnhub %s: non-JSON response: %s", symbol, resp.text[:200])
        return None, "error"
    if not data or "data" not in data or len(data.get("data", [])) == 0:
        return None, "empty"
    return data["data"], "ok"


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
