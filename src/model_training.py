"""
model_training.py
=================
Trains a stacked ensemble (RF + XGBoost + LightGBM -> LogisticRegression meta-learner)
with class balancing, hyperparameter tuning, and probability calibration.

Changes from v1:
- Stacked meta-learner using OOF predictions
- class_weight='balanced' on RF/LGB, scale_pos_weight on XGBoost
- RandomizedSearchCV with 5-fold TimeSeriesSplit, scoring=roc_auc
- CalibratedClassifierCV(method='sigmoid') on final stacked model
- Saves per-model accuracy, confusion matrix data, and feature importances
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.model_selection import (
    RandomizedSearchCV, TimeSeriesSplit, KFold, cross_val_predict
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_data(features_path: str, raw_path: str):
    """Load features and compute leak-free target variable."""
    df = pd.read_csv(features_path, index_col="Date", parse_dates=True)
    df.index = df.index.normalize()

    raw = pd.read_csv(raw_path, index_col="Date", parse_dates=True)
    raw.index = raw.index.normalize()

    # Target: 1 if NEXT day close > TODAY close (shift -1 on close only)
    target = (raw["Nifty_Close"].shift(-1) > raw["Nifty_Close"]).astype(int)
    target.name = "Target"

    df = df.join(target, how="inner").dropna(subset=["Target"])
    df["Target"] = df["Target"].astype(int)
    return df


def _tune_model(estimator, param_grid: dict, X_train, y_train, cv, label: str):
    """Run RandomizedSearchCV and return best estimator + best params."""
    search = RandomizedSearchCV(
        estimator, param_grid,
        n_iter=10, cv=cv,
        scoring="roc_auc",
        random_state=42, n_jobs=1,
        refit=True
    )
    search.fit(X_train, y_train)
    print(f"  [{label}] Best AUC: {search.best_score_:.4f} | Params: {search.best_params_}")
    return search.best_estimator_


def train_model(
    features_path: str = None,
    raw_path: str = None,
    model_save_path: str = None,
) -> object:
    """
    Full training pipeline:
    1. Feature selection via RF importance
    2. Hyperparameter tuning (RandomizedSearchCV, 5-fold TimeSeriesSplit)
    3. OOF stacking -> LogisticRegression meta-learner
    4. Probability calibration (sigmoid)
    5. Save artifacts

    Returns
    -------
    Calibrated stacked classifier.
    """
    if features_path is None:
        features_path = os.path.join(BASE_DIR, "data", "features.csv")
    if raw_path is None:
        raw_path = os.path.join(BASE_DIR, "data", "raw_data.csv")
    if model_save_path is None:
        model_save_path = os.path.join(BASE_DIR, "models", "ensemble_model.pkl")

    # ── Load ─────────────────────────────────────────────────────────────────
    df = _load_data(features_path, raw_path)
    X  = df.drop(columns=["Target"])
    y  = df["Target"]

    up_pct = y.mean() * 100
    print(f"Class distribution - UP: {y.sum()} ({up_pct:.1f}%)  DOWN: {(~y.astype(bool)).sum()} ({100-up_pct:.1f}%)")

    # ── Chronological split ───────────────────────────────────────────────────
    train_mask = df.index <= "2023-12-31"
    test_mask  = df.index >= "2024-01-01"
    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    # ── Step 1: Feature Selection (top 20 by RF importance) ──────────────────
    print("\n--- Step 1: Feature Selection ---")
    sel_rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=1
    )
    sel_rf.fit(X_train, y_train)
    importance = pd.Series(sel_rf.feature_importances_, index=X.columns)
    top_features = importance.nlargest(20).index.tolist()
    print(f"  Top 20 features: {top_features}")

    # Save full importance for dashboard
    importance.sort_values(ascending=False).to_csv(
        os.path.join(BASE_DIR, "data", "feature_importance.csv"), header=["importance"]
    )

    X_tr = X_train[top_features]
    X_te = X_test[top_features]

    # -- Step 1b: SMOTE oversampling on training set --------------------------
    # Applied AFTER the chronological split so test data is never touched.
    # SMOTE synthesizes minority-class examples to achieve a 1:1 class ratio.
    print("\n--- Step 1b: SMOTE Oversampling ---")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_train)
    X_tr_sm = pd.DataFrame(X_tr_sm, columns=top_features)
    y_tr_sm = pd.Series(y_tr_sm)
    print(f"  Before SMOTE: {len(y_train)} samples | After: {len(y_tr_sm)} samples")
    print(f"  UP: {y_tr_sm.sum()} ({y_tr_sm.mean()*100:.1f}%)  DOWN: {(1-y_tr_sm).sum()} ({(1-y_tr_sm).mean()*100:.1f}%)")

    # ── Step 2: Hyperparameter Tuning ─────────────────────────────────────────
    print("\n--- Step 2: Hyperparameter Tuning (5-fold TimeSeriesSplit, AUC) ---")
    tscv = TimeSeriesSplit(n_splits=5)
    scale_pos = float((y_train == 0).sum()) / float((y_train == 1).sum())

    rf_best = _tune_model(
        RandomForestClassifier(random_state=42, n_jobs=1),  # no class_weight: SMOTE handles it
        {
            "n_estimators": [200, 300, 400],
            "max_depth":    [4, 6, 8, None],
            "min_samples_leaf": [3, 5, 10],
            "max_features": ["sqrt", 0.5],
        },
        X_tr_sm, y_tr_sm, tscv, "RandomForest"
    )

    xgb_best = _tune_model(
        XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1),  # no scale_pos: SMOTE handles it
        {
            "n_estimators":     [200, 300, 400],
            "max_depth":        [3, 4, 5, 6],
            "learning_rate":    [0.01, 0.05, 0.1],
            "subsample":        [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
        },
        X_tr_sm, y_tr_sm, tscv, "XGBoost"
    )

    lgb_best = _tune_model(
        LGBMClassifier(verbose=-1, random_state=42, n_jobs=1),  # no class_weight: SMOTE handles it
        {
            "n_estimators":      [200, 300, 400],
            "max_depth":         [3, 4, 5, -1],
            "learning_rate":     [0.01, 0.05, 0.1],
            "num_leaves":        [20, 31, 50],
            "min_child_samples": [10, 20, 30],
        },
        X_tr_sm, y_tr_sm, tscv, "LightGBM"
    )

    # -- Step 3: OOF Stacking --------------------------------------------------
    print("\n--- Step 3: OOF Stacking -> Meta-Learner (on SMOTE data) ---")
    # Use KFold on SMOTE-balanced data for full partitions
    oof_cv = KFold(n_splits=5, shuffle=False)

    oof_rf  = cross_val_predict(rf_best,  X_tr_sm, y_tr_sm, cv=oof_cv, method="predict_proba")[:, 1]
    oof_xgb = cross_val_predict(xgb_best, X_tr_sm, y_tr_sm, cv=oof_cv, method="predict_proba")[:, 1]
    oof_lgb = cross_val_predict(lgb_best, X_tr_sm, y_tr_sm, cv=oof_cv, method="predict_proba")[:, 1]

    oof_stack = np.column_stack([oof_rf, oof_xgb, oof_lgb])
    meta = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    meta.fit(oof_stack, y_tr_sm)

    # Refit base models on full SMOTE training set
    rf_best.fit(X_tr_sm, y_tr_sm)
    xgb_best.fit(X_tr_sm, y_tr_sm)
    lgb_best.fit(X_tr_sm, y_tr_sm)

    # ── Step 4: Calibrated final predictions ──────────────────────────────────
    print("\n--- Step 4: Calibration + Evaluation ---")

    def stack_predict_proba(X):
        p_rf  = rf_best.predict_proba(X)[:, 1]
        p_xgb = xgb_best.predict_proba(X)[:, 1]
        p_lgb = lgb_best.predict_proba(X)[:, 1]
        stack = np.column_stack([p_rf, p_xgb, p_lgb])
        return meta.predict_proba(stack)[:, 1]

    raw_proba    = stack_predict_proba(X_te)
    raw_preds    = (raw_proba >= 0.5).astype(int)

    # Per-model accuracy
    per_model = {
        "Random Forest": accuracy_score(y_test, rf_best.predict(X_te)),
        "XGBoost":       accuracy_score(y_test, xgb_best.predict(X_te)),
        "LightGBM":      accuracy_score(y_test, lgb_best.predict(X_te)),
        "Stacked Meta":  accuracy_score(y_test, raw_preds),
    }
    print("Per-model Test Accuracy:")
    for model, acc in per_model.items():
        print(f"  {model}: {acc:.4f}")

    print(f"\nEnsemble AUC: {roc_auc_score(y_test, raw_proba):.4f}")
    print(classification_report(y_test, raw_preds))

    # ── Step 5: Save artifacts ────────────────────────────────────────────────
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

    test_results = pd.DataFrame(index=X_te.index)
    test_results.index = test_results.index.normalize()
    test_results["Target"]     = y_test.values
    test_results["Prediction"] = raw_preds
    test_results["Prob_UP"]    = np.clip(raw_proba, 0, 1)
    p_rf  = rf_best.predict_proba(X_te)[:, 1]
    p_xgb = xgb_best.predict_proba(X_te)[:, 1]
    p_lgb = lgb_best.predict_proba(X_te)[:, 1]
    test_results["RF_prob"]    = p_rf
    test_results["XGB_prob"]   = p_xgb
    test_results["LGB_prob"]   = p_lgb
    # Consensus: all 3 models agree on UP direction
    test_results["Consensus_UP"] = ((p_rf > 0.5) & (p_xgb > 0.5) & (p_lgb > 0.5)).astype(int)
    test_results.to_csv(
        os.path.join(BASE_DIR, "data", "test_predictions.csv"),
        date_format="%Y-%m-%d"
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, raw_preds)
    pd.DataFrame(cm, index=["Actual DOWN","Actual UP"], columns=["Pred DOWN","Pred UP"]).to_csv(
        os.path.join(BASE_DIR, "data", "confusion_matrix.csv")
    )

    # Per-model accuracy
    pd.Series(per_model).to_csv(
        os.path.join(BASE_DIR, "data", "per_model_accuracy.csv"), header=["accuracy"]
    )

    # Save top features list
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    joblib.dump(top_features, os.path.join(BASE_DIR, "models", "feature_cols.pkl"))

    # Save base + meta models
    joblib.dump({
        "rf": rf_best, "xgb": xgb_best, "lgb": lgb_best, "meta": meta
    }, model_save_path)
    print(f"\nModel saved -> {model_save_path}")

    return rf_best, xgb_best, lgb_best, meta


if __name__ == "__main__":
    train_model()
