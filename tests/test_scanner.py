import numpy as np
import pytest

from prism.scanner import cagr, classify_signal, compute_growths


class TestCagr:
    def test_basic(self):
        assert cagr(100, 121, 2) == pytest.approx(0.10)

    def test_invalid_inputs(self):
        assert cagr(None, 100, 3) is None
        assert cagr(100, None, 3) is None
        assert cagr(np.nan, 100, 3) is None
        assert cagr(-50, 100, 3) is None  # negative base has no CAGR
        assert cagr(100, 150, 0) is None


class TestComputeGrowths:
    def test_positive_gap_when_earnings_outrun_price(self):
        ni = {2021: 100, 2023: 150, 2025: 300, 2026: 330}
        px = {2021: 50, 2023: 55, 2025: 60, 2026: 61}
        g = compute_growths(ni, px, 2026)
        assert g["gap_1y"] > 0
        assert g["gap_3y"] > 0
        assert g["gap_5y"] > 0

    def test_missing_years_yield_none(self):
        g = compute_growths({2026: 100}, {2026: 50}, 2026)
        assert g["gap_1y"] is None
        assert g["gap_5y"] is None

    def test_ni_end_falls_back_to_prior_year(self):
        # No current-year NI reported yet — uses last fiscal year
        ni = {2023: 100, 2025: 121}
        px = {2023: 10, 2026: 10}
        g = compute_growths(ni, px, 2026)
        assert g["ni_cagr_3y"] == pytest.approx(0.0656, abs=1e-3)


class TestClassifySignal:
    def test_bands(self):
        assert classify_signal({"gap_1y": 0.2, "gap_3y": 0.2}) == "UNDERVALUED"
        assert classify_signal({"gap_1y": 0.05}) == "MILD UNDER"
        assert classify_signal({"gap_1y": 0.0}) == "FAIR"
        assert classify_signal({"gap_1y": -0.05}) == "MILD OVER"
        assert classify_signal({"gap_1y": -0.3}) == "OVERVALUED"
        assert classify_signal({"gap_1y": None}) == "NO DATA"
