# TradeSignal Scanner v2.0

Automated multi-asset trading signal scanner with Jedi Green Lights and live market sentiment.

## What It Does
- Scans **35 stocks** + **15 crypto pairs** daily
- Outputs the **top 3–5 setups** with full trade parameters
- Minimum **2.5:1 R/R** threshold (targets 5:1)
- Deployed via **GitHub Actions** — runs weekdays at 9am EST
- Dashboard hosted on **GitHub Pages**

## Indicators
| Indicator | Weight |
|---|---|
| EMA 5/9 Crossover | Primary signal (3pts) |
| EMA Trend (5 > 9 > 50) | Trend confirmation (2pts) |
| MACD Crossover + Histogram | Momentum (2pts) |
| Jedi Green Lights (Gaussian Kernel) | Slope direction (2pts) |
| StochRSI K/D | Overbought/oversold filter (1pt) |
| Volume Surge (≥1.5x avg) | Conviction filter (1pt) |
| Bollinger Band Squeeze | Volatility setup (1pt) |

## Market Sentiment
- **CNN Fear & Greed Index** (live via alternative.me API)
- **VIX** (live via Yahoo Finance)
- Composite **RISK-ON / NEUTRAL / RISK-OFF** bias displayed on dashboard
- Manual macro note shown for geopolitical/Fed event awareness

## Trade Parameters
- **Entry**: Current price at signal
- **Stop Loss**: `max(EMA9 × 0.995, price − 1.5 × ATR)`
- **T1**: 2:1 R/R
- **T2**: 3.5:1 R/R
- **T3**: 5:1 R/R

## Setup
```bash
pip install numpy pandas yfinance ta python-dotenv
python scanner.py
```

## Files
| File | Purpose |
|---|---|
| `scanner.py` | Main scanner engine |
| `dashboard.html` | Live results dashboard |
| `scan_results.json` | Latest scan output |
| `.github/workflows/scanner.yml` | Automated daily run |

## Dashboard
Enable GitHub Pages on the `main` branch root to view the live dashboard.
