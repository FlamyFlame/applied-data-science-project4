#!/usr/bin/env python3
"""
DNN pipeline: feature prep -> 80/20 split -> Optuna K=3 CV (20 trials) -> best model -> test eval.
Target: review_score regression (1-5).
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from optuna.pruners import MedianPruner
import warnings
warnings.filterwarnings('ignore')

PROCESSED_DIR = 'processed'
RESULTS_DIR   = 'results'
SEED          = 42
N_FOLDS       = 3     # 3-fold: each fold ~58K train rows; faster than 5-fold on CPU
N_TRIALS      = 20    # Bayesian search is sample-efficient; 20 trials sufficient
PATIENCE      = 10
MAX_EPOCHS    = 100

os.makedirs(RESULTS_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Feature Preparation ───────────────────────────────────────────────────────

_DROP = [
    'order_id', 'customer_id', 'product_id', 'seller_id', 'review_id',
    'customer_unique_id', 'order_item_id', 'payment_sequential',
    'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
    'order_delivered_customer_date', 'order_estimated_delivery_date', 'shipping_limit_date',
    'geolocation_zip_code_prefix_x', 'geolocation_zip_code_prefix_y',
    'customer_zip_code_prefix', 'seller_zip_code_prefix',
    'customer_city', 'seller_city', 'order_status', 'is_bad_review',
]
_CAT   = ['payment_type', 'seller_state', 'customer_state', 'product_category_name']
TARGET = 'review_score'


def _load_raw():
    """Load clean_df_final, drop irrelevant columns. Returns (DataFrame, y)."""
    df = pd.read_csv(os.path.join(PROCESSED_DIR, 'clean_df_final.csv'))
    df['product_category_name'] = df['product_category_name'].fillna('unknown')
    df = df.drop(columns=[c for c in _DROP if c in df.columns])
    y  = df.pop(TARGET).values.astype(np.float32)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df, y


def _fit_preprocessor(df_train):
    """Fit OHE on categorical cols and compute medians on numeric cols from training data only."""
    cat_cols = [c for c in _CAT if c in df_train.columns]
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(df_train[cat_cols].astype(str))
    num_cols = [c for c in df_train.columns if c not in cat_cols]
    medians  = df_train[num_cols].median()
    return ohe, medians


def _apply_preprocessor(df, ohe, medians):
    """Transform a DataFrame to a fixed-width float32 numpy array."""
    cat_cols = [c for c in _CAT if c in df.columns]
    num_cols = [c for c in df.columns if c not in cat_cols]

    num_part = df[num_cols].fillna(medians.reindex(num_cols))
    bool_cols = num_part.select_dtypes(include='bool').columns
    num_part = num_part.copy()
    num_part[bool_cols] = num_part[bool_cols].astype(np.float32)

    cat_part  = ohe.transform(df[cat_cols].astype(str)).astype(np.float32)
    cat_names = list(ohe.get_feature_names_out(cat_cols))

    X = np.concatenate([num_part.values.astype(np.float32), cat_part], axis=1)
    return X, num_cols + cat_names


# ── Model ─────────────────────────────────────────────────────────────────────

_ACT = {'relu': nn.ReLU, 'gelu': nn.GELU, 'elu': nn.ELU}


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout, activation='relu'):
        super().__init__()
        act = _ACT[activation]
        layers, in_dim = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), act(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Training ──────────────────────────────────────────────────────────────────

def _make_loaders(X_tr, y_tr, X_val, y_val, batch_size):
    def dl(X, y, shuffle):
        return DataLoader(TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y)),
                          batch_size=batch_size, shuffle=shuffle)
    return dl(X_tr, y_tr, True), dl(X_val, y_val, False)


def train_model(model, train_dl, val_dl, lr, weight_decay,
                patience=PATIENCE, max_epochs=MAX_EPOCHS):
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    loss_fn = nn.MSELoss()
    best_val, no_improve, best_state = float('inf'), 0, None

    for _ in range(max_epochs):
        model.train()
        for Xb, yb in train_dl:
            opt.zero_grad(); loss_fn(model(Xb), yb).backward(); opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_mse = np.mean([loss_fn(model(Xb), yb).item() for Xb, yb in val_dl])

        if val_mse < best_val:
            best_val   = val_mse
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return best_val


# ── Optuna Objective ──────────────────────────────────────────────────────────

def _make_objective(X_train, y_train):
    def objective(trial):
        n_layers   = trial.suggest_int('n_layers', 2, 5)
        hidden_dim = trial.suggest_int('hidden_dim', 64, 512, log=True)
        dropout    = trial.suggest_float('dropout', 0.0, 0.5)
        lr         = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        wd         = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [1024, 2048, 4096])
        activation = trial.suggest_categorical('activation', ['relu', 'gelu', 'elu'])

        hidden_dims = [hidden_dim] * n_layers
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_rmses = []

        for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
            scaler = StandardScaler()
            X_tr   = scaler.fit_transform(X_train[tr_idx])
            X_val  = scaler.transform(X_train[val_idx])
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            train_dl, val_dl = _make_loaders(X_tr, y_tr, X_val, y_val, batch_size)
            model  = MLP(X_tr.shape[1], hidden_dims, dropout, activation)
            mse    = train_model(model, train_dl, val_dl, lr, wd)
            fold_rmses.append(np.sqrt(mse))

            trial.report(float(np.mean(fold_rmses)), fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_rmses))
    return objective


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df, y = _load_raw()
    print(f"  Rows: {len(df)}  |  target: [{y.min():.0f}, {y.max():.0f}]")

    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=SEED, stratify=y.astype(int)
    )

    print("Fitting preprocessor on training data...")
    ohe, medians = _fit_preprocessor(df_train)
    X_train, feature_names = _apply_preprocessor(df_train, ohe, medians)
    X_test,  _             = _apply_preprocessor(df_test,  ohe, medians)
    print(f"  Feature matrix: {X_train.shape}  |  Train: {len(X_train)}  |  Test: {len(X_test)}")

    print(f"\nOptuna search: {N_TRIALS} trials, K={N_FOLDS} folds each (~25 min on CPU)...")
    study = optuna.create_study(
        direction='minimize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(_make_objective(X_train, y_train), n_trials=N_TRIALS, show_progress_bar=True)

    best_p = study.best_trial.params
    print(f"\nBest CV RMSE : {study.best_trial.value:.4f}")
    print(f"Best params  : {best_p}")

    print("\nRetraining on full train set...")
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_train)
    X_te_s  = scaler.transform(X_test)

    hidden_dims = [best_p['hidden_dim']] * best_p['n_layers']
    model = MLP(X_tr_s.shape[1], hidden_dims, best_p['dropout'], best_p['activation'])
    train_dl, val_dl = _make_loaders(X_tr_s, y_train, X_te_s, y_test, best_p['batch_size'])
    train_model(model, train_dl, val_dl, best_p['lr'], best_p['weight_decay'])

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_te_s)).numpy()

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    print(f"\n── Test Set ──")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")

    print("\nPer-score breakdown (mean predicted | count):")
    for score in range(1, 6):
        mask = y_test == score
        if mask.sum():
            print(f"  {score}★  pred_mean={y_pred[mask].mean():.2f}  n={mask.sum()}")

    # Save artifacts
    results = {
        'best_params': best_p,
        'cv_rmse': study.best_trial.value,
        'test_rmse': rmse, 'test_mae': mae, 'test_r2': r2,
    }
    with open(os.path.join(RESULTS_DIR, 'dnn_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'best_model.pt'))
    with open(os.path.join(RESULTS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(RESULTS_DIR, 'ohe.pkl'), 'wb') as f:
        pickle.dump(ohe, f)
    medians.to_json(os.path.join(RESULTS_DIR, 'feature_medians.json'))
    np.save(os.path.join(RESULTS_DIR, 'feature_names.npy'), np.array(feature_names))
    print(f"\nArtifacts saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
