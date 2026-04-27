#!/usr/bin/env python3
"""
End-to-end pipeline: data cleaning/EDA → DNN training.
Run from project root: python pipeline.py

Expects:
  - processed/ containing the raw Olist CSVs
  - Data_clean_EDA_unsupervised.py  (writes processed/clean_df_final.csv)
  - src/dnn_pipeline.py             (reads clean_df_final.csv, writes results/)
"""
import subprocess
import sys
import os

env = os.environ.copy()
env['MPLBACKEND'] = 'Agg'   # suppress plt.show() calls in EDA script

steps = [
    ('Data cleaning & EDA',  [sys.executable, 'Data_clean_EDA_unsupervised.py']),
    ('DNN training',          [sys.executable, os.path.join('src', 'dnn_pipeline.py')]),
]

for name, cmd in steps:
    print(f'\n{"─" * 60}\n  {name}\n{"─" * 60}')
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f'\nERROR: "{name}" failed (exit {result.returncode}). Aborting.')
        sys.exit(result.returncode)

print('\n' + '─' * 60)
print('  Pipeline complete. Artifacts in results/')
print('─' * 60)
