"""
walk_forward.py
===============
Walk-forward validation with rolling 18-month train, 6-month test windows.
Outputs per-window accuracy and Sharpe to detect regime drift.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_predict, TimeSeriesSplit, KFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_walk_forward(
    features_path: str = None,
    raw_path: str = None,
    train_months: int = 18,
    test_months: int = 6,
) -> pd.DataFrame:
    """
    Rolling walk-forward validation.

    For each window:
      - Train on `train_months` of data
      - Test on next `test_months`
      - Report accuracy, AUC, and naive Sharpe for each window

    Parameters
    ----------
    features_path : str
        Path to features.csv.
    raw_path : str
        Path to raw_data.csv (for target construction and prices).
    train_months : int
        Size of the training window in months (default 18).
    test_months : int
        Size of the test window in months (default 6).

    Returns
    -------
    pd.DataFrame  - Results table per window.
    """
    if features_path is None:
        features_path = os.path.join(BASE_DIR, "data", "features.csv")
    if raw_path is None:
        raw_path = os.path.join(BASE_DIR, "data", "raw_data.csv")

    feat = pd.read_csv(features_path, index_col="Date", parse_dates=True)
    feat.index = feat.index.normalize()

    raw = pd.read_csv(raw_path, index_col="Date", parse_dates=True)
    raw.index = raw.index.normalize()

    target = (raw["Nifty_Close"].shift(-1) > raw["Nifty_Close"]).astype(int)
    target.name = "Target"

    df = feat.join(target, how="inner").dropna(subset=["Target"])
    df["Target"] = df["Target"].astype(int)

    X = df.drop(columns=["Target"])
    y = df["Target"]

    # Use top-20 features if saved, else use all
    feat_path = os.path.join(BASE_DIR, "models", "feature_cols.pkl")
    if os.path.exists(feat_path):
        top_cols = joblib.load(feat_path)
        top_cols = [c for c in top_cols if c in X.columns]
        X = X[top_cols]

    dates  = df.index
    start  = dates[0]
    end    = dates[-1]

    results = []
    window_start = start

    while True:
        train_end = window_start + pd.DateOffset(months=train_months)
        test_end  = train_end   + pd.DateOffset(months=test_months)

        if test_end > end:
            break

        train_mask = (dates >= window_start) & (dates < train_end)
        test_mask  = (dates >= train_end)    & (dates < test_end)

        if train_mask.sum() < 50 or test_mask.sum() < 10:
            window_start = window_start + pd.DateOffset(months=test_months)
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask],  y[test_mask]

        scale_pos = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)

        # Lightweight models for speed
        rf  = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                      random_state=42, n_jobs=1)
        xgb = XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos,
                             eval_metric="logloss", random_state=42, n_jobs=1)
        lgb = LGBMClassifier(n_estimators=100, class_weight="balanced",
                              verbose=-1, random_state=42, n_jobs=1)

        oof_cv  = KFold(n_splits=3, shuffle=False)
        oof_rf  = cross_val_predict(rf,  X_tr, y_tr, cv=oof_cv, method="predict_proba")[:, 1]
        oof_xgb = cross_val_predict(xgb, X_tr, y_tr, cv=oof_cv, method="predict_proba")[:, 1]
        oof_lgb = cross_val_predict(lgb, X_tr, y_tr, cv=oof_cv, method="predict_proba")[:, 1]

        meta = LogisticRegression(C=1.0, max_iter=300, random_state=42)
        meta.fit(np.column_stack([oof_rf, oof_xgb, oof_lgb]), y_tr)

        rf.fit(X_tr, y_tr);  xgb.fit(X_tr, y_tr);  lgb.fit(X_tr, y_tr)

        p_rf  = rf.predict_proba(X_te)[:, 1]
        p_xgb = xgb.predict_proba(X_te)[:, 1]
        p_lgb = lgb.predict_proba(X_te)[:, 1]
        proba = meta.predict_proba(np.column_stack([p_rf, p_xgb, p_lgb]))[:, 1]
        preds = (proba >= 0.5).astype(int)

        acc = accuracy_score(y_te, preds)
        auc = roc_auc_score(y_te, proba) if len(np.unique(y_te)) > 1 else 0.5

        # Simple strategy Sharpe
        prices = raw.loc[X_te.index, "Nifty_Close"]
        rets   = prices.pct_change().fillna(0)
        signal = pd.Series(proba >= 0.5, index=X_te.index).shift(1).fillna(False)
        strat  = rets * signal.astype(float)
        cost   = signal.astype(float).diff().abs().fillna(0) * 0.0005 * 2
        strat  = strat - cost
        sharpe = np.sqrt(252) * strat.mean() / strat.std() if strat.std() > 0 else 0.0

        results.append({
            "Window": f"{train_end.strftime('%b %Y')} -> {test_end.strftime('%b %Y')}",
            "Train Start": window_start.strftime("%Y-%m"),
            "Test Start":  train_end.strftime("%Y-%m"),
            "Test End":    test_end.strftime("%Y-%m"),
            "Train Size":  int(train_mask.sum()),
            "Test Size":   int(test_mask.sum()),
            "Accuracy":    round(acc, 4),
            "AUC":         round(auc, 4),
            "Sharpe":      round(sharpe, 4),
        })

        label = results[-1]["Window"]
        print(f"  {label} | Acc: {acc:.4f} | AUC: {auc:.4f} | Sharpe: {sharpe:.4f}")

        window_start = window_start + pd.DateOffset(months=test_months)

    results_df = pd.DataFrame(results)
    out_path   = os.path.join(BASE_DIR, "data", "walk_forward_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nWalk-Forward complete. {len(results)} windows. Saved -> {out_path}")
    print(f"  Mean Accuracy: {results_df['Accuracy'].mean():.4f}")
    print(f"  Mean AUC:      {results_df['AUC'].mean():.4f}")
    print(f"  Mean Sharpe:   {results_df['Sharpe'].mean():.4f}")
    return results_df


if __name__ == "__main__":
    run_walk_forward()
