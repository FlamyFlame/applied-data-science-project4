#!/usr/bin/env python3
"""
Logistic Regression baseline for review score classification.

TODO (team): decide target, tune hyperparameters, add evaluation metrics.
Suggested target: binary is_bad_review (review_score <= 2) for interpretability,
or 5-class review_score for direct comparison with the DNN.
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              classification_report)
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dnn_pipeline import _load_raw, _fit_preprocessor, _apply_preprocessor, SEED

RESULTS_DIR = os.path.join('results', 'logistic_regression')
os.makedirs(RESULTS_DIR, exist_ok=True)

N_FOLDS = 3


def main():
    print("Loading data...")
    df, y_score = _load_raw()          # y_score: review_score 1–5

    # ── TODO: choose target ────────────────────────────────────────────────
    # Option A — binary (recommended for LR baseline)
    y = (y_score <= 2).astype(int)     # is_bad_review
    # Option B — 5-class
    # y = (y_score - 1).astype(int)    # 0-indexed classes 0..4
    # ───────────────────────────────────────────────────────────────────────

    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=SEED, stratify=y
    )

    print("Preprocessing...")
    ohe, medians = _fit_preprocessor(df_train)
    X_train, feature_names = _apply_preprocessor(df_train, ohe, medians)
    X_test,  _             = _apply_preprocessor(df_test,  ohe, medians)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    print(f"  Features: {X_train.shape[1]}  |  Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"  Class balance — 0: {(y_train==0).sum()}  1: {(y_train==1).sum()}")

    # ── TODO: hyperparameter search ────────────────────────────────────────
    # Suggested: GridSearchCV or RandomizedSearchCV over:
    #   C          : [0.001, 0.01, 0.1, 1, 10]
    #   penalty    : ['l1', 'l2']   (use solver='saga' for l1)
    #   class_weight: [None, 'balanced']   (important for imbalanced classes)
    # K-fold: StratifiedKFold(n_splits=N_FOLDS) — use stratified for binary
    # ───────────────────────────────────────────────────────────────────────

    print("\nTraining Logistic Regression...")
    # TODO: replace with tuned estimator from the search above
    model = LogisticRegression(
        C=1.0, penalty='l2', solver='lbfgs',
        class_weight='balanced', max_iter=1000, random_state=SEED,
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n── Test Set ──")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 macro  : {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")
    print(classification_report(y_test, y_pred, target_names=['good', 'bad']))

    # Save artifacts
    results = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'f1_macro': float(f1_score(y_test, y_pred, average='macro')),
        'roc_auc':  float(roc_auc_score(y_test, y_proba)),
    }
    with open(os.path.join(RESULTS_DIR, 'lr_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'lr_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(RESULTS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(RESULTS_DIR, 'ohe.pkl'), 'wb') as f:
        pickle.dump(ohe, f)
    medians.to_json(os.path.join(RESULTS_DIR, 'feature_medians.json'))
    print(f"\nArtifacts saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
