#!/usr/bin/env python3
"""
XGBoost gradient-boosted tree model for review score prediction.

Requires: pip install xgboost

TODO (team): choose regression vs classification, tune hyperparameters.
Framing as regression (same target as DNN) allows direct RMSE/MAE comparison.
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score)
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("XGBoost not installed. Run: pip install xgboost")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dnn_pipeline import _load_raw, _fit_preprocessor, _apply_preprocessor, SEED

RESULTS_DIR = os.path.join('results', 'xgboost')
os.makedirs(RESULTS_DIR, exist_ok=True)

N_FOLDS = 3


def main():
    print("Loading data...")
    df, y = _load_raw()                # y: review_score 1–5 (regression target)

    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=SEED, stratify=y.astype(int)
    )

    print("Preprocessing...")
    ohe, medians = _fit_preprocessor(df_train)
    X_train, feature_names = _apply_preprocessor(df_train, ohe, medians)
    X_test,  _             = _apply_preprocessor(df_test,  ohe, medians)
    # Note: XGBoost handles mixed scales internally; StandardScaler optional here
    print(f"  Features: {X_train.shape[1]}  |  Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── TODO: hyperparameter search ────────────────────────────────────────
    # Recommended: optuna.create_study + XGBRegressor inside objective, or
    # sklearn RandomizedSearchCV. Key parameters to tune:
    #   n_estimators  : [100, 300, 500]
    #   max_depth     : [3, 5, 7]
    #   learning_rate : [0.01, 0.05, 0.1, 0.3]
    #   subsample     : [0.6, 0.8, 1.0]
    #   colsample_bytree: [0.6, 0.8, 1.0]
    #   reg_alpha / reg_lambda: log-uniform [1e-3, 10]
    # Use early stopping with eval_set on a validation fold to avoid over-fitting.
    # ───────────────────────────────────────────────────────────────────────

    print("\nTraining XGBoost...")
    # TODO: replace with tuned estimator from the search above
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )
    # TODO: add eval_set + early_stopping_rounds once tuning is in place
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    print(f"\n── Test Set ──")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")

    # Per-score breakdown (mirrors DNN output for easy comparison)
    print("\nPer-score breakdown (mean predicted | count):")
    for score in range(1, 6):
        mask = y_test == score
        if mask.sum():
            print(f"  {score}★  pred_mean={y_pred[mask].mean():.2f}  n={mask.sum()}")

    # Feature importance (top 20)
    importances = pd.Series(model.feature_importances_, index=feature_names)
    print("\nTop 20 features by importance:")
    print(importances.nlargest(20).to_string())

    # Save artifacts
    results = {'test_rmse': rmse, 'test_mae': mae, 'test_r2': r2}
    with open(os.path.join(RESULTS_DIR, 'xgb_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'xgb_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(RESULTS_DIR, 'ohe.pkl'), 'wb') as f:
        pickle.dump(ohe, f)
    medians.to_json(os.path.join(RESULTS_DIR, 'feature_medians.json'))
    print(f"\nArtifacts saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
