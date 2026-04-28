#!/usr/bin/env python3
"""
Supervised Model 1: Logistic Regression
Target: is_bad_review (1 = review_score <= 2, 0 = good review)

Feature set: 13 numeric + 3 logistics-cluster dummies + top-15 category dummies (~31 total).
Deliberately narrower than the DNN feature set — more defensible for a linear model
and produces interpretable coefficients.

Original analysis: teammate (root logistic_regression.py).
Integrated here: relative paths, artifact saving, pipeline compatibility.
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, accuracy_score
)
from sklearn.impute import SimpleImputer

PROCESSED_DIR = 'processed'
RESULTS_DIR   = os.path.join('results', 'logistic_regression')
FIGURES_DIR   = 'figures'
SEED          = 42
TOP_N_CATS    = 15

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def main():
    # ── 0. Load data ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("Loading data...")
    df = pd.read_csv(os.path.join(PROCESSED_DIR, 'clean_df_final.csv'))
    print(f"  Dataset shape: {df.shape}")

    # ── 1. Feature selection ──────────────────────────────────────────────────
    numeric_features = [
        'price', 'freight_value', 'freight_ratio',
        'delivery_days', 'delay_days', 'distance_km',
        'seller_recent_delay_avg', 'product_volume_cm3',
        'product_weight_g', 'product_photos_qty',
        'product_name_lenght', 'product_description_lenght',
        'order_item_id',
    ]
    cluster_features = ['logistics_cluster_1', 'logistics_cluster_2', 'logistics_cluster_3']

    top_cats = df['product_category_name'].value_counts().nlargest(TOP_N_CATS).index.tolist()
    df['product_category_grouped'] = df['product_category_name'].apply(
        lambda x: x if x in top_cats else 'other'
    )
    cat_dummies = pd.get_dummies(df['product_category_grouped'], prefix='cat', drop_first=True)

    X = pd.concat([df[numeric_features], df[cluster_features].astype(int), cat_dummies], axis=1)
    y = (df['review_score'] <= 2).astype(int)   # is_bad_review

    print(f"\n  Features used: {X.shape[1]}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")
    print(f"  Class imbalance ratio: {(y == 0).sum() / (y == 1).sum():.1f}:1")

    # ── 2. Handle remaining nulls ─────────────────────────────────────────────
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # ── 3. Train / test split (stratified) ───────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"\n  Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # ── 4. Cross-validation (5-fold, stratified) ──────────────────────────────
    print("\n" + "=" * 60)
    print("Running 5-fold Stratified Cross-Validation...")

    lr = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        solver='lbfgs',
        C=1.0,
        random_state=SEED,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_results = cross_validate(
        lr, X_train, y_train, cv=cv,
        scoring=['accuracy', 'f1', 'roc_auc', 'average_precision'],
        return_train_score=True,
    )

    metrics_cv = {
        'Accuracy':      ('test_accuracy',          'train_accuracy'),
        'F1 Score':      ('test_f1',                'train_f1'),
        'ROC-AUC':       ('test_roc_auc',           'train_roc_auc'),
        'Avg Precision': ('test_average_precision', 'train_average_precision'),
    }
    print("\n  Cross-Validation Results (mean ± std):")
    for name, (test_key, train_key) in metrics_cv.items():
        tr = cv_results[train_key]
        te = cv_results[test_key]
        print(f"  {name:<18} train: {tr.mean():.4f} ± {tr.std():.4f}  |  "
              f"val: {te.mean():.4f} ± {te.std():.4f}")

    # ── 5. Fit on full train, evaluate on test ────────────────────────────────
    print("\n" + "=" * 60)
    print("Fitting on full train set, evaluating on held-out test set...")

    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    ap  = average_precision_score(y_test, y_prob)

    print(f"\n  Test Accuracy:       {acc:.4f}")
    print(f"  Test F1 Score:       {f1:.4f}")
    print(f"  Test ROC-AUC:        {auc:.4f}")
    print(f"  Test Avg Precision:  {ap:.4f}")
    print()
    print("  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Good Review', 'Bad Review']))

    # ── 6. Top feature coefficients ───────────────────────────────────────────
    coef_df = pd.DataFrame({
        'feature':     X.columns,
        'coefficient': lr.coef_[0],
    }).sort_values('coefficient', key=abs, ascending=False)

    print("  Top 15 Features by |Coefficient|:")
    print(coef_df.head(15).to_string(index=False))

    # ── 7. Visualizations ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Logistic Regression – Bad Review Prediction',
                 fontsize=15, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # CV metric bar chart
    ax0 = fig.add_subplot(gs[0, 0])
    cv_means = {k: cv_results[v[0]].mean() for k, v in metrics_cv.items()}
    cv_stds  = {k: cv_results[v[0]].std()  for k, v in metrics_cv.items()}
    bars = ax0.bar(list(cv_means.keys()), list(cv_means.values()),
                   yerr=list(cv_stds.values()), capsize=5,
                   color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
    ax0.set_ylim(0, 1.05)
    ax0.set_ylabel('Score')
    ax0.set_title('5-Fold CV Validation Metrics')
    ax0.tick_params(axis='x', rotation=20)
    for bar, val in zip(bars, cv_means.values()):
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Confusion matrix
    ax1 = fig.add_subplot(gs[0, 1])
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
    ax1.set_xlabel('Predicted'); ax1.set_ylabel('Actual')
    ax1.set_title('Confusion Matrix (Test Set)')

    # ROC curve
    ax2 = fig.add_subplot(gs[0, 2])
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax2.plot(fpr, tpr, color='#C44E52', lw=2, label=f'AUC = {auc:.3f}')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1)
    ax2.set_xlabel('False Positive Rate'); ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve (Test Set)'); ax2.legend(loc='lower right')

    # Precision-Recall curve
    ax3 = fig.add_subplot(gs[1, 0])
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ax3.plot(recall, precision, color='#55A868', lw=2, label=f'AP = {ap:.3f}')
    ax3.axhline(y=(y_test == 1).mean(), color='gray', linestyle='--', lw=1, label='Baseline')
    ax3.set_xlabel('Recall'); ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Curve (Test Set)'); ax3.legend()

    # Feature coefficients
    ax4 = fig.add_subplot(gs[1, 1:])
    top_feats = coef_df.head(15).sort_values('coefficient')
    colors = ['#C44E52' if c > 0 else '#4C72B0' for c in top_feats['coefficient']]
    ax4.barh(top_feats['feature'], top_feats['coefficient'], color=colors)
    ax4.axvline(0, color='black', lw=0.8)
    ax4.set_xlabel('Coefficient Value')
    ax4.set_title('Top 15 Feature Coefficients\n(red = increases bad-review risk)')

    plot_path = os.path.join(FIGURES_DIR, 'logistic_regression_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved to {plot_path}")

    # ── 8. Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary = pd.DataFrame({
        'Metric':   ['Accuracy', 'F1 Score', 'ROC-AUC', 'Avg Precision'],
        'CV (val)': [f"{cv_results['test_accuracy'].mean():.4f}",
                     f"{cv_results['test_f1'].mean():.4f}",
                     f"{cv_results['test_roc_auc'].mean():.4f}",
                     f"{cv_results['test_average_precision'].mean():.4f}"],
        'Test Set': [f'{acc:.4f}', f'{f1:.4f}', f'{auc:.4f}', f'{ap:.4f}'],
    })
    print(summary.to_string(index=False))

    # ── 9. Save artifacts ─────────────────────────────────────────────────────
    results = {
        'accuracy': float(acc), 'f1': float(f1),
        'roc_auc':  float(auc), 'avg_precision': float(ap),
        'cv': {k: float(cv_results[v[0]].mean()) for k, v in metrics_cv.items()},
    }
    with open(os.path.join(RESULTS_DIR, 'lr_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'lr_model.pkl'), 'wb') as f:
        pickle.dump(lr, f)
    with open(os.path.join(RESULTS_DIR, 'imputer.pkl'), 'wb') as f:
        pickle.dump(imputer, f)
    with open(os.path.join(RESULTS_DIR, 'feature_columns.json'), 'w') as f:
        json.dump(list(X.columns), f, indent=2)
    print(f"\nArtifacts saved to {RESULTS_DIR}/")
    print("\nDone.")


if __name__ == '__main__':
    main()
