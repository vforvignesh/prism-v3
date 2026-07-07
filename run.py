#!/usr/bin/env python3
"""PRISM CLI entry point.

Usage:
    python run.py                    # run with same-day fetch cache if available
    python run.py --refresh          # force refetch of all market data
    python run.py --config other.yaml
"""
import argparse
import logging
import warnings

from prism.config import load_api_keys, load_config
from prism.data.fetch import fetch_all_data
from prism.export import export_csvs
from prism.portfolio import build_portfolios
from prism.report import generate_report
from prism.scoring import run_prism
from prism.shock import run_shock

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="PRISM stock screener")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--refresh", action="store_true",
                        help="ignore same-day cache and refetch market data")
    parser.add_argument("--verbose", action="store_true",
                        help="show per-request data-source diagnostics")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="  %(levelname)s %(name)s: %(message)s")

    tickers, cfg = load_config(args.config)
    fmp_key, fh_key = load_api_keys()

    df, quality_df = fetch_all_data(tickers, cfg, fmp_key, fh_key, refresh=args.refresh)
    print("\n  Running PRISM scoring engine...")
    df = run_prism(df, cfg)
    print("  Building portfolios...")
    portfolios, fragile, reps = build_portfolios(df, cfg)
    print("  Running SHOCK stress tests...")
    shock = run_shock(df, portfolios)
    generate_report(df, portfolios, fragile, reps, shock, cfg)
    export_csvs(df, portfolios, shock, quality_df, cfg)
    print("\n  PRISM complete!")


if __name__ == "__main__":
    main()
