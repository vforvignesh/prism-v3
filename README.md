# PRISM

Quantitative stock screener, portfolio constructor, and macro-shock
stress-tester for a growth watchlist. Successor to the `Prism_Pilot_V03`
Colab notebook (archived in `notebooks/`), restructured as a local Python
package.

## What it does

1. **Fetches** analyst EPS estimates (FMP, cross-validated by Finnhub when
   available) and market data (yfinance) for the watchlist in `config.yaml`.
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
cp .env.example .env   # add your FMP_API_KEY and FINNHUB_API_KEY
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

## Data source caveats

- FMP free tier gates many symbols (HTTP 402) — those fall back to yfinance,
  shown in the data-quality scorecard.
- Finnhub's `eps-estimate` endpoint is premium-only; on a free key it is
  auto-disabled.
- Foreign listings may report FMP EPS in local currency; implausible implied
  P/Es (<3x) are discarded in favour of yfinance.
