#!/usr/bin/env python3
"""
End-to-end pipeline: data cleaning/EDA → supervised models.
Run from project root: python pipeline.py

Steps
-----
1. Data_clean_EDA_unsupervised.py  — merges raw CSVs, engineers features,
                                     writes processed/clean_df_final.csv
2. src/logistic_regression.py      — LR baseline, writes results/logistic_regression/
3. src/xgboost_model.py            — XGBoost, writes results/xgboost/
4. src/dnn_pipeline.py             — MLP with Optuna search, writes results/

Requirements: pip install torch optuna xgboost scikit-learn pandas numpy
"""
import subprocess
import sys
import os

env = os.environ.copy()
env['MPLBACKEND'] = 'Agg'   # suppress plt.show() calls in non-interactive runs

steps = [
    ('Data cleaning & EDA',  [sys.executable, 'Data_clean_EDA_unsupervised.py']),
    ('Logistic Regression',  [sys.executable, os.path.join('src', 'logistic_regression.py')]),
    ('XGBoost',              [sys.executable, os.path.join('src', 'xgboost_model.py')]),
    ('DNN (MLP)',            [sys.executable, os.path.join('src', 'dnn_pipeline.py')]),
]

for name, cmd in steps:
    print(f'\n{"─" * 60}\n  {name}\n{"─" * 60}')
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f'\nERROR: "{name}" failed (exit {result.returncode}). Aborting.')
        sys.exit(result.returncode)

print('\n' + '─' * 60)
print('  Pipeline complete.')
print('  Results: results/logistic_regression/  results/xgboost/  results/')
print('─' * 60)
