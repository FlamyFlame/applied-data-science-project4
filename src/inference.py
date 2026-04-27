#!/usr/bin/env python3
"""
Inference: predict review_score for new orders in clean_df_final format.

Usage:
    python src/inference.py processed/clean_df_final.csv
    python src/inference.py my_new_orders.csv --out predictions.csv
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch

# Allow running from project root or from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dnn_pipeline import MLP, _DROP, _CAT, TARGET, RESULTS_DIR


def load_artifacts(results_dir=RESULTS_DIR):
    """Load all saved artifacts and reconstruct the model."""
    paths = {k: os.path.join(results_dir, v) for k, v in {
        'results':  'dnn_results.json',
        'ohe':      'ohe.pkl',
        'scaler':   'scaler.pkl',
        'medians':  'feature_medians.json',
        'features': 'feature_names.npy',
        'weights':  'best_model.pt',
    }.items()}

    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts: {missing}. Run src/dnn_pipeline.py first."
        )

    with open(paths['results']) as f:
        results = json.load(f)
    with open(paths['ohe'], 'rb') as f:
        ohe = pickle.load(f)
    with open(paths['scaler'], 'rb') as f:
        scaler = pickle.load(f)
    medians      = pd.read_json(paths['medians'], typ='series')
    feature_names = np.load(paths['features'], allow_pickle=True).tolist()

    p = results['best_params']
    model = MLP(len(feature_names), [p['hidden_dim']] * p['n_layers'],
                p['dropout'], p['activation'])
    model.load_state_dict(torch.load(paths['weights'], map_location='cpu'))
    model.eval()

    return model, ohe, scaler, medians, feature_names


def _prepare(df, ohe, scaler, medians, feature_names):
    """Transform a DataFrame (clean_df_final format) to the scaled input the model expects."""
    df = df.copy()
    for col in [TARGET, 'is_bad_review']:
        if col in df.columns:
            df = df.drop(columns=[col])

    df['product_category_name'] = df.get(
        'product_category_name', pd.Series(['unknown'] * len(df), index=df.index)
    ).fillna('unknown')
    df = df.drop(columns=[c for c in _DROP if c in df.columns], errors='ignore')
    df = df.apply(pd.to_numeric, errors='coerce')

    cat_cols = [c for c in _CAT if c in df.columns]
    num_cols = [c for c in df.columns if c not in cat_cols]

    num_part = df[num_cols].fillna(medians.reindex(num_cols))
    bool_cols = num_part.select_dtypes(include='bool').columns
    num_part[bool_cols] = num_part[bool_cols].astype(np.float32)

    cat_part  = ohe.transform(df[cat_cols].astype(str)).astype(np.float32)
    cat_names = list(ohe.get_feature_names_out(cat_cols))

    X = np.concatenate([num_part.values.astype(np.float32), cat_part], axis=1)
    current_names = num_cols + cat_names

    # Align to training column order; fill any unseen columns with 0
    X_df = pd.DataFrame(X, columns=current_names).reindex(columns=feature_names, fill_value=0.0)
    return scaler.transform(X_df.values.astype(np.float32))


def predict(df, results_dir=RESULTS_DIR):
    """Return predicted review scores (float32 array, clipped to [1, 5])."""
    model, ohe, scaler, medians, feature_names = load_artifacts(results_dir)
    X = _prepare(df, ohe, scaler, medians, feature_names)
    with torch.no_grad():
        preds = model(torch.FloatTensor(X)).numpy()
    return np.clip(preds, 1.0, 5.0)


def main():
    parser = argparse.ArgumentParser(description='Predict review scores for new orders.')
    parser.add_argument('input',  help='CSV file in clean_df_final format')
    parser.add_argument('--out',  default=None, help='Output CSV path (default: print to stdout)')
    parser.add_argument('--results', default=RESULTS_DIR, help='Directory with saved artifacts')
    args = parser.parse_args()

    df   = pd.read_csv(args.input)
    preds = predict(df, results_dir=args.results)

    out_df = pd.DataFrame({'predicted_review_score': preds.round(3)})

    if args.out:
        out_df.to_csv(args.out, index=False)
        print(f"Saved {len(preds)} predictions to {args.out}")
    else:
        print(out_df.to_string(index=False))

    print(f"\nSummary — mean: {preds.mean():.3f}  std: {preds.std():.3f}  "
          f"min: {preds.min():.3f}  max: {preds.max():.3f}")


if __name__ == '__main__':
    main()
