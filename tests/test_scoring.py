import numpy as np
import pandas as pd
import pytest

from prism.portfolio import build_portfolios
from prism.scoring import pct_rank, run_prism
from prism.shock import SHOCK_SCENARIOS, run_shock, shock_stock


def make_fixture(n=12, seed=7):
    rng = np.random.RandomState(seed)
    low = rng.uniform(20, 100, n)
    high = low * rng.uniform(1.2, 3.0, n)
    price = low + (high - low) * rng.uniform(0.1, 0.95, n)
    return pd.DataFrame({
        "Stock": [f"T{i:02d}" for i in range(n)],
        "Price": price,
        "Market Cap": rng.uniform(5e9, 2e12, n),
        "Beta": rng.uniform(0.5, 2.8, n),
        "52 Week High": high,
        "52 Week Low": low,
        "P/E.1": rng.uniform(8, 90, n),
        "Industry": ["Semiconductors", "Software - Application"] * (n // 2),
        "Target Mean": price * rng.uniform(0.9, 1.6, n),
        "Average Outcome": rng.uniform(-0.1, 0.5, n),
        "2026 Growth Rate": rng.uniform(-0.1, 1.2, n),
        "2027 Growth Rate": rng.uniform(0.0, 0.8, n),
    })


CFG = {
    "weight_growth": 0.35, "weight_value": 0.25,
    "weight_momentum": 0.20, "weight_resilience": 0.20,
    "blend_2026": 0.75, "blend_2027": 0.25, "growth_cap": 2.00,
    "edge_size": 5, "fragile_threshold": 3.0,
    "growth_overrides": {}, "force_include": [], "force_exclude": [],
    "output_dir": ".",
}


class TestPctRank:
    def test_ascending_bounds(self):
        s = pd.Series([1, 2, 3, 4])
        r = pct_rank(s)
        assert r.iloc[0] == 25.0
        assert r.iloc[-1] == 100.0

    def test_descending_reverses_order(self):
        s = pd.Series([1, 2, 3, 4])
        r = pct_rank(s, ascending=False)
        assert r.iloc[0] > r.iloc[-1]
        assert r.iloc[0] == 100.0


class TestMomentum:
    def test_higher_returns_score_higher_all_else_equal(self):
        df = make_fixture()
        df["Ret 3M"] = np.linspace(-0.2, 0.6, len(df))
        df["Ret 6M"] = np.linspace(-0.3, 0.9, len(df))
        df["Ret 12M"] = np.linspace(-0.4, 1.2, len(df))
        base = df.copy()
        # equalize everything else that feeds momentum
        for col in ["Average Outcome", "Beta", "P/E.1"]:
            base[col] = base[col].mean()
        base["52 Week Low"] = 50.0
        base["52 Week High"] = 100.0
        base["Price"] = 75.0
        scored = run_prism(base, CFG)
        assert scored.set_index("Stock")["dim_momentum"].loc["T11"] > \
            scored.set_index("Stock")["dim_momentum"].loc["T00"]

    def test_missing_return_columns_neutral(self):
        df = run_prism(make_fixture(), CFG)  # fixture has no Ret columns
        assert df["dim_momentum"].notna().all()

    def test_sector_relative_flag(self):
        df = make_fixture()
        cfg = dict(CFG, sector_relative=True)
        scored = run_prism(df, cfg)
        assert scored["dim_growth"].between(0, 100).all()


class TestRunPrism:
    def test_scores_bounded_and_complete(self):
        df = run_prism(make_fixture(), CFG)
        assert df["prism_score"].between(0, 100).all()
        for dim in ["dim_growth", "dim_value", "dim_momentum", "dim_resilience"]:
            assert df[dim].notna().all(), dim
        assert sorted(df["prism_rank"]) == list(range(1, len(df) + 1))

    def test_ranking_sorted_by_score(self):
        df = run_prism(make_fixture(), CFG)
        assert df["prism_score"].is_monotonic_decreasing

    def test_sector_mapping(self):
        df = run_prism(make_fixture(), CFG)
        assert set(df["shock_sector"]) == {"Semiconductor", "Software"}


class TestPortfolios:
    def test_structure(self):
        df = run_prism(make_fixture(), CFG)
        portfolios, frag, reps = build_portfolios(df, CFG)
        assert set(portfolios) == {"PRISM Full", "PRISM Select", "PRISM Edge", "Custom"}
        assert len(portfolios["PRISM Edge"]) <= CFG["edge_size"]
        # Select swaps fragile names out
        assert not set(frag) & set(portfolios["PRISM Select"])

    def test_force_include_exclude(self):
        df = run_prism(make_fixture(), CFG)
        cfg = dict(CFG, force_include=["T00"], force_exclude=["T01"])
        portfolios, _, _ = build_portfolios(df, cfg)
        assert "T00" in portfolios["Custom"]
        assert "T01" not in portfolios["Custom"]


class TestShock:
    def test_shock_stock_bounded(self):
        row = {"Beta": 2.0, "shock_sector": "Semiconductor", "P/E.1": 100,
               "Market Cap": 5e9, "52w_position": 0.95, "is_profitable": False}
        for scenario in SHOCK_SCENARIOS.values():
            dd = shock_stock(row, scenario)
            assert -0.95 <= dd < 0

    def test_high_beta_semis_fall_more_in_semi_downturn(self):
        semi = {"Beta": 2.0, "shock_sector": "Semiconductor", "P/E.1": 40,
                "Market Cap": 1e11, "52w_position": 0.5, "is_profitable": True}
        insurer = dict(semi, shock_sector="Insurance", Beta=0.8)
        sc = SHOCK_SCENARIOS["Semi Downturn"]
        assert shock_stock(semi, sc) < shock_stock(insurer, sc)

    def test_run_shock_metrics(self):
        df = run_prism(make_fixture(), CFG)
        portfolios, _, _ = build_portfolios(df, CFG)
        results = run_shock(df, portfolios)
        for pn, r in results.items():
            assert r["SHOCK Score"] < 0, pn
            assert r["Edge Ratio"] >= 0, pn
            assert r["1Y Recovery"] in {"YES", "MAYBE", "NO"}
