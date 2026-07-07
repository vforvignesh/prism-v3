# PRISM

Quantitative stock screener, portfolio constructor, and macro-shock
stress-tester for a growth watchlist. Successor to the `Prism_Pilot_V03`
Colab notebook (archived in `notebooks/`), restructured as a local Python
package.

## What it does

1. **Fetches** market data and analyst growth estimates from Yahoo Finance
   (yfinance) for the watchlist in `config.yaml` — no API keys needed.
2. **Scores** each stock 0–100 on Growth / Value / Momentum / Resilience and
   blends them into a PRISM score. See [docs/methodology.md](docs/methodology.md).
3. **Builds portfolios** — Full (top half), Select (fragile names replaced),
   Edge (concentrated), Custom (force-include/exclude).
4. **Stress-tests** them against 8 probability-weighted shock scenarios and
   recommends the portfolio with the best growth-per-drawdown Edge Ratio.
5. **Exports** dated CSVs, an HTML report, and appends to a score history.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Usage

```bash
.venv/bin/python run.py            # uses same-day fetch cache when present
.venv/bin/python run.py --refresh  # force refetch of market data
.venv/bin/python run.py --verbose  # per-request data-source diagnostics
```

Outputs land in `outputs/<date>/` (CSVs + `prism_report.html`); score history
accumulates in `outputs/history.csv`.

Edit `config.yaml` to change the watchlist, dimension weights, growth
overrides, or enable `sector_relative` ranking.

## Development

```bash
.venv/bin/python -m pytest tests/         # unit tests
.venv/bin/python scripts/parity_check.py  # notebook-vs-package scoring parity
.venv/bin/python scripts/calibrate_shocks.py  # shock model vs 2020/2022 reality
```

## Data source notes

Yahoo Finance is the single data source (no API keys required). Stocks with
no analyst growth estimate get a flat 4% and are flagged in the data-quality
scorecard — pin better numbers via `growth_overrides` in `config.yaml`.
FMP/Finnhub were removed in v3; see [docs/methodology.md](docs/methodology.md)
for the rationale.
