# PRISM Methodology

PRISM scores a watchlist of growth stocks on four dimensions, blends them into
a composite score, constructs portfolios, and stress-tests those portfolios
against probability-weighted macro shock scenarios.

## Data sources

| Source | Role | Notes |
|---|---|---|
| FMP `/stable/analyst-estimates` | Primary EPS growth estimates | Free tier gates many symbols (HTTP 402) per symbol; EPS field is `epsAvg`. Foreign listings may report EPS in local currency — implied P/E under 3x is discarded as a currency mismatch. |
| Finnhub `eps-estimate` | Cross-validation of FMP growth | Premium-only endpoint; disabled automatically on HTTP 403. |
| yfinance | Price, beta, market cap, 52w range, analyst targets, price history; growth fallback | Class-share tickers are remapped (`BRK.B` → `BRK-B`). |

Cross-validation confidence: **HIGH** = FMP and Finnhub agree within 15%,
**MED** = within 30% (averaged) or single-source, **LOW** = >30% disagreement
(FMP kept), **YF** = yfinance fallback, **DEFAULT** = no source (4% assumed).

Fiscal years ending Jan–May are attributed to the *prior* calendar year
(e.g. NVDA's FY ending Jan 2027 counts as calendar 2026).

## Dimensions (default weights)

- **Growth 35%** — percentile ranks of capped 2026 growth (40%), 2027 growth
  (15%), blended growth (30%, 75/25 blend), and durability = g27/g26 (15%).
  Growth is capped at 200% so one hyper-grower doesn't dominate.
  With `sector_relative: true`, growth ranks are blended 50% global /
  50% within-sector to reduce structural sector bias.
- **Value 25%** — for profitable names: PEG proxy rank (65%) + forward P/E rank
  (35%). Pre-profit names get a flat score keyed to growth (55/35/15).
- **Momentum 20%** — real price momentum rank (30%; trailing returns blended
  3M×0.30 + 6M×0.50 + 12M×0.20, neutral 50 if history is missing), analyst
  upside rank (30%), stretch score (25%), 52-week-position rank (15%).
  The *stretch score* starts at 100 and deducts for being near 52w highs,
  expensive PEG, high beta, and wide 52w ranges; it adds back for being
  near 52w lows.
- **Resilience 20%** — growth-per-beta rank (45%), log market cap rank (20%),
  and a 2026-growth floor (0/40/100 for negative / <5% / ≥5%) (35%).

## Fragility flags

Points accumulate for: beta (>1.8/2.0/2.5 → 1/1.5/2), 52w position
(>0.70/0.80/0.90 → 1/1.5/2), max(PEG penalty, pre-profit penalty), and 52w
range width. At ≥3.0 points a stock is *fragile* and excluded from
PRISM Select (replaced by the best non-fragile candidates).

## Portfolios

- **PRISM Full** — top ~50% by score.
- **PRISM Select** — Full with fragile names swapped for sturdier candidates.
- **PRISM Edge** — up to `edge_size` names maximizing an edge score
  (score 40%, upside×stretch 25%, growth 25%, distance from 52w high 10%).
- **Custom** — Edge plus `force_include` minus `force_exclude`.

## Shock model

Eight scenarios, each with a probability, a market drawdown, and sector
multipliers. A stock's modeled drawdown is:

```
dd = market_dd × beta_mult × sector_mult × valuation_mult × size_mult × mean_reversion_mult × profit_mult
```

capped at −95%. The **SHOCK Score** is the probability-weighted expected
drawdown; the **Edge Ratio** is blended growth ÷ |SHOCK Score| — growth earned
per unit of crash risk. The portfolio with the best Edge Ratio is recommended.

`scripts/calibrate_shocks.py` compares model predictions against realized
2020 and 2022 drawdowns for the current watchlist (directional check only —
it uses today's beta/valuation inputs).

Calibration snapshot (2026-07-07 watchlist): COVID Crash is well calibrated
(mean bias −1.4%, rank correlation +0.57). **Rate Shock is ~20pp too lenient**
(mean bias +19.6% vs 2022 reality; SE, NU, NFLX were the biggest misses) —
consider raising its `market_dd` or the growth-sector multipliers before
leaning on that scenario.

## Changes vs notebook v2.3

1. FMP field rename (`estimatedEpsAvg` → `epsAvg`) fixed — v2.3 parsed NaN
   from every successful FMP response.
2. FMP is no longer disabled after 3 failed tickers — the free tier gates
   per symbol, so each symbol is tried.
3. `pct_rank(descending)` now uses a true descending percentile rank; the old
   formula mis-ranked series containing NaNs (divided by the NaN-inclusive
   length).
4. Momentum now includes actual trailing returns; previously "momentum" was
   entirely analyst upside + positioning heuristics. Weights rebalanced
   45/35/20 → 30/30/25/15.
5. Optional sector-relative growth ranking (`sector_relative` in config.yaml).
