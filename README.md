# Nifty 50 Next-Day Direction Predictor

> An end-to-end ensemble machine learning system to predict the next-day directional movement of the Nifty 50 Index with realistic Indian market costs and walk-forward validation.

**Built by CA Bhavya** • AI for Finance Level 2

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![ML](https://img.shields.io/badge/ML-Stacked_Ensemble-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?style=flat-square)
![Backtesting](https://img.shields.io/badge/Backtesting-Realistic_Costs-orange?style=flat-square)

---

## Dashboard

![Nifty 50 ML Predictor Dashboard](dashboard)

> Interactive Streamlit dashboard showing live predictions, feature importance, and backtest equity curve.

---

## Overview

This project is a production-style ML system that predicts whether the Nifty 50 will close **higher or lower** the next trading day. 

It goes beyond basic notebooks by incorporating:
- Proper time-series validation
- Realistic transaction costs (STT + slippage)
- Probability calibration
- Walk-forward testing
- Deployed interactive dashboard

---

## Key Highlights

- **Stacked Ensemble** — Random Forest + XGBoost + LightGBM with Logistic Regression meta-learner
- **No Data Leakage** — All features and macro variables properly lagged
- **Realistic Backtesting** — Includes Indian market costs + Half-Kelly position sizing
- **Walk-Forward Validation** — Rolling 18-month training windows
- **Probability Calibration** — Sigmoid calibration for more reliable confidence scores
- **Feature Selection** — Top features selected using Random Forest importance

---

## Methodology

### 1. Feature Engineering
- 40+ technical indicators (RSI, MACD, Bollinger Bands, ATR, Stochastic, Williams %R, CCI, etc.)
- Lagged macro features (India VIX, USDINR)
- Multiple lagged returns and rate-of-change features
- All indicators calculated in a **leak-free** manner

### 2. Model Architecture
- Base models: Random Forest, XGBoost, LightGBM (with SMOTE balancing)
- Meta-learner: Logistic Regression trained on out-of-fold predictions
- Hyperparameter tuning using `RandomizedSearchCV` + `TimeSeriesSplit`
- Final model calibrated using `CalibratedClassifierCV`

### 3. Backtesting Framework
- **Train Period**: Jan 2018 – Dec 2023
- **Test Period**: Jan 2024 – Present
- Transaction costs + slippage modeled
- Position sizing using Half-Kelly Criterion
- Stop-loss applied at -1%

---

## Current Performance (Baseline)

| Metric              | Value          | Notes                     |
|---------------------|----------------|---------------------------|
| Test Accuracy       | ~50%           | Near random               |
| Win Rate            | ~43%           | Below 50%                 |
| Sharpe Ratio        | Negative       | Strategy currently losing |
| Max Drawdown        | High           | Needs improvement         |

**Note**: This is the **honest baseline** result. The model shows a downward bias and trades too frequently. Significant improvements are planned in the next version (probability thresholding + regime filtering).

---

## Project Structure

```
ai-ml-nifty/
├── src/
│   ├── data_loader.py           # Data ingestion from yfinance
│   ├── feature_engineering.py   # 40+ leak-free technical indicators
│   ├── model_training.py        # Stacked ensemble + calibration
│   ├── backtesting.py           # Realistic cost-aware backtester
│   └── walk_forward.py          # Rolling window validation
├── data/                        # Generated datasets (gitignored)
├── models/                      # Saved model artifacts (gitignored)
├── app.py                       # Streamlit dashboard
├── dashboard                    # Dashboard screenshot
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tech Stack

`Python` `Pandas` `NumPy` `Scikit-Learn` `XGBoost` `LightGBM` `SMOTE` `Streamlit` `Plotly` `Joblib`

---

## How to Run

```bash
# 1. Clone the repository

```