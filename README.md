# Nifty 50 Next-Day Direction Predictor

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![ML](https://img.shields.io/badge/ML-Ensemble-green) ![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)

## Overview
An end-to-end ensemble ML system to predict the next-day directional movement of the Nifty 50 Index, with realistic Indian market backtesting.

**Built by CA Bhavya | AI for Finance Level 2**

## Methodology

### Feature Engineering
- 40+ custom technical indicators (RSI-14, MACD, Bollinger %B, ATR, OBV, Stochastic, Williams %R, CCI, EMA crossovers)
- Macro features: India VIX (lagged 1 day), USDINR (lagged 1 day)
- 5 lagged Nifty returns (lag-1 to lag-5), Rate-of-Change (5/10/20 days)
- Zero data leakage: all macro features shifted 1 day; raw OHLCV dropped from feature set

### Model Architecture
- **Base Models**: Random Forest, XGBoost, LightGBM (all class-balanced)
- **Meta-Learner**: Logistic Regression trained on OOF predictions
- **Tuning**: RandomizedSearchCV with 5-fold TimeSeriesSplit, scoring=AUC
- **Calibration**: Sigmoid probability calibration

### Backtesting
- **Train**: Jan 2018 – Dec 2023 | **Test**: Jan 2024 – Present
- **Walk-Forward**: Rolling 18-month train, 6-month test windows
- **Costs**: 0.05% per trade + stop-loss at -1% per position
- **Sizing**: Half-Kelly Criterion

## Performance (Baseline — Honest)

| Metric | Value |
|---|---|
| Test Accuracy | ~50% |
| Sharpe Ratio | See dashboard |
| Max Drawdown | See dashboard |
| Win Rate | See dashboard |

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/nifty50-ml-predictor.git
cd nifty50-ml-predictor

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python src/data_loader.py
python src/feature_engineering.py
python src/model_training.py
python src/backtesting.py

# (Optional) Walk-forward validation
python src/walk_forward.py

# Launch the dashboard
streamlit run app.py
```

## Project Structure

```
nifty50-ml-predictor/
├── src/
│   ├── data_loader.py           # Data ingestion (yfinance)
│   ├── feature_engineering.py   # 40+ leak-free indicators
│   ├── model_training.py        # Stacked ensemble training
│   ├── backtesting.py           # Realistic backtest engine
│   └── walk_forward.py          # Rolling window validation
├── data/                        # Generated CSV files (gitignored)
├── models/                      # Saved model artifacts (gitignored)
├── app.py                       # Streamlit dashboard
├── requirements.txt
├── .gitignore
└── README.md
```

## Tech Stack
`Python` `Pandas` `Scikit-Learn` `XGBoost` `LightGBM` `Streamlit` `Plotly`
