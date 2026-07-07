# PRISM v3 — Portfolio Risk & Intelligence Scoring Model

A quantitative portfolio intelligence dashboard that scores and ranks stocks using a composite of fundamental, risk, and technical indicators.

## Architecture

```
Data Sources:  FMP → Finnhub → yfinance (3-layer pipeline with cross-validation)
Scoring:       40% Fundamental (Relative PEG + Alpha Score)
               30% Risk (SHOCK scenarios + Edge Ratio)
               30% Technical (RSI, MACD, SMA, Bollinger Bands)
Dashboard:     Streamlit (dark theme)
Deployment:    Render.com (free tier) or Streamlit Cloud
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys as environment variables
export FMP_API_KEY=your_fmp_key
export FINNHUB_API_KEY=your_finnhub_key

# Run locally
streamlit run app.py
```

## Deploy to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service → Connect your repo
3. Render auto-detects `render.yaml` and configures everything
4. Add environment variables: `FMP_API_KEY` and `FINNHUB_API_KEY`
5. Deploy

## Tabs

| Tab | Description |
|-----|-------------|
| **Watchlist** | Scored & ranked table, KPIs, Risk vs Growth scatter |
| **Stock Detail** | Deep-dive: metrics, score breakdown, SHOCK scenarios, technicals |
| **Stress Test** | Portfolio-level SHOCK drawdowns, recovery analysis |
| **Data Quality** | Multi-source cross-validation scorecard, growth overrides |

## Scoring Methodology

### Fundamental Score (0-100)
- **Relative PEG** = (Stock PE / Stock Growth) / (Index PE / Index Growth)
- **Alpha Score** = Growth Delta / Relative PE
- Lower PEG + higher alpha = higher score

### Risk Score (0-100)
- **SHOCK Score** = probability-weighted drawdown across 5 scenarios
- **Edge Ratio** = Growth% / |SHOCK Score| (risk-reward)
- **1Y Recovery** = Can the stock recover from SHOCK drawdown in 1 year?

### Technical Score (0-100)
- Trend (40%): price vs SMA(50), SMA(200), golden/death cross
- Momentum (40%): RSI(14), MACD histogram
- Volatility (20%): Bollinger Band %B position

## Data Pipeline

Priority: FMP analyst estimates → Finnhub EPS → yfinance analyst growth → yfinance EPS (fwd/trailing) → yfinance revenue growth → 4% default

Cross-validation: when multiple sources agree within 15% → HIGH confidence. 15-30% gap → MEDIUM. >30% → LOW.
