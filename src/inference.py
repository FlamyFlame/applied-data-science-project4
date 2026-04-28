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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dnn_pipeline import (
    MLP, _DROP, _OHE_CATS, _EMB_CAT, _ALL_CATS, TARGET, RESULTS_DIR, EMBED_DIM,
)


def load_artifacts(results_dir=RESULTS_DIR):
    paths = {k: os.path.join(results_dir, v) for k, v in {
        'results':   'dnn_results.json',
        'ohe':       'ohe.pkl',
        'scaler':    'scaler.pkl',
        'medians':   'feature_medians.json',
        'features':  'feature_names.npy',
        'weights':   'best_model.pt',
        'cat_index': 'cat_index_map.json',
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
    with open(paths['cat_index']) as f:
        cat_index_map = json.load(f)
    medians       = pd.read_json(paths['medians'], typ='series')
    feature_names = np.load(paths['features'], allow_pickle=True).tolist()

    p            = results['best_params']
    n_categories = len(cat_index_map)
    model = MLP(
        len(feature_names), [p['hidden_dim']] * p['n_layers'],
        p['dropout'], p['activation'],
        n_categories=n_categories, embed_dim=EMBED_DIM,
    )
    model.load_state_dict(torch.load(paths['weights'], map_location='cpu'))
    model.eval()

    return model, ohe, scaler, medians, feature_names, cat_index_map


def _prepare(df, ohe, scaler, medians, feature_names, cat_index_map):
    df = df.copy()
    for col in [TARGET, 'is_bad_review']:
        if col in df.columns:
            df = df.drop(columns=[col])

    df[_EMB_CAT] = df.get(
        _EMB_CAT, pd.Series(['unknown'] * len(df), index=df.index)
    ).fillna('unknown')
    df = df.drop(columns=[c for c in _DROP if c in df.columns], errors='ignore')

    ohe_cols = [c for c in _OHE_CATS if c in df.columns]
    num_cols = [c for c in df.columns if c not in _ALL_CATS]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    num_part  = df[num_cols].fillna(medians.reindex(num_cols))
    bool_cols = num_part.select_dtypes(include='bool').columns
    num_part  = num_part.copy()
    num_part[bool_cols] = num_part[bool_cols].astype(np.float32)

    ohe_part  = ohe.transform(df[ohe_cols].astype(str)).astype(np.float32)
    ohe_names = list(ohe.get_feature_names_out(ohe_cols))

    X_dense       = np.concatenate([num_part.values.astype(np.float32), ohe_part], axis=1)
    current_names = list(num_cols) + ohe_names

    # Align to training column order; fill any unseen OHE columns with 0
    X_df     = pd.DataFrame(X_dense, columns=current_names).reindex(
        columns=feature_names, fill_value=0.0,
    )
    X_scaled = scaler.transform(X_df.values.astype(np.float32))

    # Category indices for embedding
    unk_idx = cat_index_map.get('unknown', 0)
    cat_idx = df[_EMB_CAT].astype(str).map(
        lambda x: cat_index_map.get(x, unk_idx)
    ).values.astype(np.int64)

    return X_scaled, cat_idx


def predict(df, results_dir=RESULTS_DIR, return_probs=False):
    """
    Return predicted review star class (int array, values 1-5).
    If return_probs=True, also return (n, 5) softmax probability matrix.
    """
    import torch.nn.functional as F
    model, ohe, scaler, medians, feature_names, cat_index_map = load_artifacts(results_dir)
    X_scaled, cat_idx = _prepare(df, ohe, scaler, medians, feature_names, cat_index_map)
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_scaled), torch.LongTensor(cat_idx))
        probs  = F.softmax(logits, dim=1).numpy()
        preds  = logits.argmax(dim=1).numpy() + 1   # 1-5
    if return_probs:
        return preds, probs
    return preds


def main():
    parser = argparse.ArgumentParser(description='Predict review scores for new orders.')
    parser.add_argument('input',     help='CSV file in clean_df_final format')
    parser.add_argument('--out',     default=None, help='Output CSV path (default: print to stdout)')
    parser.add_argument('--results', default=RESULTS_DIR, help='Directory with saved artifacts')
    args = parser.parse_args()

    df    = pd.read_csv(args.input)
    preds = predict(df, results_dir=args.results)   # integer 1-5

    out_df = pd.DataFrame({'predicted_review_score': preds})

    if args.out:
        out_df.to_csv(args.out, index=False)
        print(f"Saved {len(preds)} predictions to {args.out}")
    else:
        print(out_df.to_string(index=False))

    counts = np.bincount(preds, minlength=6)
    print("\nPrediction distribution:")
    for s in range(1, 6):
        print(f"  {s}★ : {counts[s]:>6}  ({100*counts[s]/len(preds):.1f}%)")


if __name__ == '__main__':
    main()
