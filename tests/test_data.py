import numpy as np
import pytest

from prism.data.common import map_fy_to_calendar, validate_growth
from prism.data.finnhub import calc_growth_from_finnhub
from prism.data.fmp import calc_growth_from_fmp


class TestMapFyToCalendar:
    def test_fy_ending_late_in_year_maps_to_same_year(self):
        assert map_fy_to_calendar("2026-12-31") == 2026
        assert map_fy_to_calendar("2026-09-27") == 2026

    def test_fy_ending_early_in_year_maps_to_prior_year(self):
        # e.g. NVDA FY ends late January: FY2031 covers calendar 2030
        assert map_fy_to_calendar("2031-01-25") == 2030
        assert map_fy_to_calendar("2026-05-31") == 2025

    def test_invalid_date_returns_none(self):
        assert map_fy_to_calendar("") is None
        assert map_fy_to_calendar("not-a-date") is None


class TestValidateGrowth:
    def test_agreeing_sources_high_confidence(self):
        g, conf = validate_growth(0.20, 0.21)
        assert g == 0.20
        assert conf == "HIGH"

    def test_moderate_disagreement_averages(self):
        g, conf = validate_growth(0.20, 0.25)
        assert g == pytest.approx(0.225)
        assert conf == "MED"

    def test_large_disagreement_flags_low(self):
        g, conf = validate_growth(0.20, 0.40)
        assert g == 0.20  # keeps primary source
        assert conf == "LOW"

    def test_single_source_is_med(self):
        assert validate_growth(0.20, np.nan) == (0.20, "MED")
        assert validate_growth(np.nan, 0.30) == (0.30, "MED")

    def test_no_sources(self):
        g, conf = validate_growth(np.nan, np.nan)
        assert np.isnan(g)
        assert conf == "NONE"


class TestCalcGrowthFromFmp:
    def test_stable_api_field_name(self):
        # /stable/ endpoint uses epsAvg (v3 used estimatedEpsAvg)
        data = [
            {"date": "2025-12-31", "epsAvg": 10.0},
            {"date": "2026-12-31", "epsAvg": 12.0},
            {"date": "2027-12-31", "epsAvg": 15.0},
        ]
        g26, g27, e26, status = calc_growth_from_fmp(data)
        assert g26 == pytest.approx(0.20)
        assert g27 == pytest.approx(0.25)
        assert e26 == 12.0
        assert status == "ok"

    def test_legacy_field_name_still_supported(self):
        data = [
            {"date": "2025-12-31", "estimatedEpsAvg": 10.0},
            {"date": "2026-12-31", "estimatedEpsAvg": 11.0},
        ]
        g26, _, _, _ = calc_growth_from_fmp(data)
        assert g26 == pytest.approx(0.10)

    def test_negative_to_positive_transition(self):
        data = [
            {"date": "2025-12-31", "epsAvg": -2.0},
            {"date": "2026-12-31", "epsAvg": 1.0},
        ]
        g26, _, _, _ = calc_growth_from_fmp(data)
        assert g26 == pytest.approx(1.5)  # abs(1 - -2) / abs(-2)

    def test_empty(self):
        g26, g27, e26, status = calc_growth_from_fmp([])
        assert status == "no_data"
        assert np.isnan(g26)


class TestCalcGrowthFromFinnhub:
    def test_basic(self):
        data = [
            {"date": "2025-12-31", "epsAvg": 10.0},
            {"date": "2026-12-31", "epsAvg": 13.0},
            {"date": "2027-12-31", "epsAvg": 14.3},
        ]
        g26, g27 = calc_growth_from_finnhub(data)
        assert g26 == pytest.approx(0.30)
        assert g27 == pytest.approx(0.10)
