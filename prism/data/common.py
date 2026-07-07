"""Shared helpers for the data layer."""
from datetime import datetime

import numpy as np
import pandas as pd


def map_fy_to_calendar(date_str):
    """Map a fiscal-year-end date string to the calendar year the estimate covers."""
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return dt.year - 1 if dt.month <= 5 else dt.year
    except Exception:
        return None


def validate_growth(fmp_g, fh_g):
    """Cross-validate FMP vs Finnhub growth estimates.

    Returns (growth, confidence) where confidence is HIGH/MED/LOW/NONE.
    """
    has_fmp, has_fh = pd.notna(fmp_g), pd.notna(fh_g)
    if has_fmp and has_fh:
        if fmp_g == 0 and fh_g == 0:
            return 0.0, "HIGH"
        avg = (abs(fmp_g) + abs(fh_g)) / 2
        if avg == 0:
            return fmp_g, "MED"
        diff = abs(fmp_g - fh_g) / max(avg, 0.01)
        if diff <= 0.15:
            return fmp_g, "HIGH"
        elif diff <= 0.30:
            return (fmp_g + fh_g) / 2, "MED"
        else:
            return fmp_g, "LOW"
    elif has_fmp:
        return fmp_g, "MED"
    elif has_fh:
        return fh_g, "MED"
    else:
        return np.nan, "NONE"
