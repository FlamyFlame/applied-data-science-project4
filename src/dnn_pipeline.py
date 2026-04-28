#!/usr/bin/env python3
"""
DNN pipeline: feature prep -> 80/20 split -> Optuna K=5 StratifiedKFold CV (20 trials)
             -> best model retrained on 90% train / 10% val for early stopping -> test eval.
Target: ordinal-regularized multi-class classification over review_score (1-5).

Architecture: standard 5-class softmax head (not a true ordinal threshold model like
CORAL/CORN).  Ordinal structure is encouraged via a regularization term in the loss —
no architectural constraint guarantees monotone cumulative probabilities.
Loss: class-weighted cross-entropy + lambda * ordinal MSE on expected score.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    mean_absolute_error, accuracy_score, confusion_matrix, cohen_kappa_score,
)
import optuna
from optuna.pruners import MedianPruner
import warnings
warnings.filterwarnings('ignore')

PROCESSED_DIR   = 'processed'
RESULTS_DIR     = 'results'
FIGURES_DIR     = 'figures'
SEED            = 42
N_FOLDS         = 5
N_TRIALS        = 20
PATIENCE        = 10
MAX_EPOCHS      = 100
EMBED_DIM       = 8
ORDINAL_LAMBDA  = 0.5   # weight on MSE(expected_score, true_score) vs cross-entropy

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
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
_OHE_CATS = ['seller_state', 'customer_state']   # OHE: 22 + 27 = 49 cols
_EMB_CAT  = 'product_category_name'              # embedding: EMBED_DIM cols
_ALL_CATS = _OHE_CATS + [_EMB_CAT]
TARGET    = 'review_score'
N_CLASSES = 5


def _load_raw():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, 'clean_df_final.csv'))
    df[_EMB_CAT] = df[_EMB_CAT].fillna('unknown')
    df = df.drop(columns=[c for c in _DROP if c in df.columns])
    y  = df.pop(TARGET).values.astype(np.float32)
    num_cols = [c for c in df.columns if c not in _ALL_CATS]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    return df, y


def _fit_preprocessor(df_train):
    ohe_cols = [c for c in _OHE_CATS if c in df_train.columns]
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(df_train[ohe_cols].astype(str))

    cats = sorted(df_train[_EMB_CAT].astype(str).unique().tolist())
    if 'unknown' not in cats:
        cats = ['unknown'] + cats
    cat_index_map = {c: i for i, c in enumerate(cats)}

    num_cols = [c for c in df_train.columns if c not in _ALL_CATS]
    medians  = df_train[num_cols].median()
    return ohe, cat_index_map, medians


def _apply_preprocessor(df, ohe, cat_index_map, medians):
    ohe_cols = [c for c in _OHE_CATS if c in df.columns]
    num_cols = [c for c in df.columns if c not in _ALL_CATS]

    num_part  = df[num_cols].fillna(medians.reindex(num_cols))
    bool_cols = num_part.select_dtypes(include='bool').columns
    num_part  = num_part.copy()
    num_part[bool_cols] = num_part[bool_cols].astype(np.float32)

    ohe_part  = ohe.transform(df[ohe_cols].astype(str)).astype(np.float32)
    ohe_names = list(ohe.get_feature_names_out(ohe_cols))

    X_dense     = np.concatenate([num_part.values.astype(np.float32), ohe_part], axis=1)
    dense_names = list(num_cols) + ohe_names

    unk_idx = cat_index_map.get('unknown', 0)
    cat_idx = df[_EMB_CAT].astype(str).map(
        lambda x: cat_index_map.get(x, unk_idx)
    ).values.astype(np.int64)

    return X_dense, cat_idx, dense_names


def _class_weights_tensor(y):
    """5-element inverse-frequency weight tensor for F.cross_entropy(weight=...)."""
    y_int  = y.astype(int)
    counts = np.bincount(y_int, minlength=6)   # index 0 unused; scores are 1-5
    total  = len(y)
    w = np.array([total / (5.0 * counts[s]) for s in range(1, 6)], dtype=np.float32)
    return torch.FloatTensor(w)


# ── Model ─────────────────────────────────────────────────────────────────────

_ACT = {'relu': nn.ReLU, 'gelu': nn.GELU, 'elu': nn.ELU}

# Reusable score tensor for expected-value computation; populated in train_model
_SCORE_TENSOR = torch.arange(1, N_CLASSES + 1, dtype=torch.float32)


class MLP(nn.Module):
    def __init__(self, dense_input_dim, hidden_dims, dropout,
                 activation='relu', n_categories=1, embed_dim=EMBED_DIM):
        super().__init__()
        self.embedding = nn.Embedding(n_categories, embed_dim)
        act    = _ACT[activation]
        in_dim = dense_input_dim + embed_dim
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), act(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, N_CLASSES))   # 5 logits, one per star rating
        self.net = nn.Sequential(*layers)

    def forward(self, x_dense, cat_idx):
        emb = self.embedding(cat_idx)
        return self.net(torch.cat([x_dense, emb], dim=1))   # (batch, 5) logits


# ── Loss ──────────────────────────────────────────────────────────────────────

def _ordinal_ce_loss(logits, y_float, class_weights_t):
    """
    Class-weighted cross-entropy + ordinal MSE on expected score.

    Cross-entropy (class-weighted): treats each star as a discrete class;
        class weights fix the 57.5%-fives imbalance, same role as in regression.
    Ordinal MSE: pulls E[predicted score] = sum_k k*P(k) toward the true score,
        penalising distant mispredictions (predicting 5★ for a 1★ order hurts more
        than predicting 2★).
    """
    y_int    = (y_float - 1).long()                                  # 0-indexed class
    ce       = F.cross_entropy(logits, y_int, weight=class_weights_t)
    probs    = F.softmax(logits, dim=1)
    scores_t = _SCORE_TENSOR.to(logits.device)
    expected = (probs * scores_t).sum(dim=1)                         # (batch,)
    ord_mse  = F.mse_loss(expected, y_float)
    return ce + ORDINAL_LAMBDA * ord_mse


# ── Training ──────────────────────────────────────────────────────────────────

def _make_loaders(X_tr, c_tr, y_tr, X_val, c_val, y_val, batch_size):
    train_ds = TensorDataset(
        torch.FloatTensor(X_tr), torch.LongTensor(c_tr), torch.FloatTensor(y_tr),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(c_val), torch.FloatTensor(y_val),
    )
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False))


def train_model(model, train_dl, val_dl, lr, weight_decay, class_weights_t,
                patience=PATIENCE, max_epochs=MAX_EPOCHS):
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    best_val, no_improve, best_state = float('inf'), 0, None

    for _ in range(max_epochs):
        model.train()
        for Xb, cb, yb in train_dl:
            loss = _ordinal_ce_loss(model(Xb, cb), yb, class_weights_t)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        # Monitor combined loss for early stopping (aligned with training objective)
        model.eval()
        with torch.no_grad():
            val_ce = np.mean([
                _ordinal_ce_loss(model(Xb, cb), yb, class_weights_t).item()
                for Xb, cb, yb in val_dl
            ])

        if val_ce < best_val:
            best_val   = val_ce
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return best_val


# ── Optuna Objective ──────────────────────────────────────────────────────────

def _make_objective(df_train, y_train):
    """
    Optuna objective.  OHE, median imputation, and embedding vocabulary are all fitted
    inside each fold on the fold's training partition only — no leakage from validation
    rows into any preprocessing step.
    """
    def objective(trial):
        n_layers   = trial.suggest_int('n_layers', 2, 5)
        hidden_dim = trial.suggest_int('hidden_dim', 64, 512, log=True)
        dropout    = trial.suggest_float('dropout', 0.0, 0.5)
        lr         = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        wd         = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [1024, 2048, 4096])
        activation = trial.suggest_categorical('activation', ['relu', 'gelu', 'elu'])

        hidden_dims  = [hidden_dim] * n_layers
        skf          = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_losses  = []
        idx          = np.arange(len(df_train))

        for fold_i, (tr_idx, val_idx) in enumerate(skf.split(idx, y_train.astype(int))):
            df_tr  = df_train.iloc[tr_idx]
            df_val = df_train.iloc[val_idx]
            y_tr   = y_train[tr_idx]
            y_val  = y_train[val_idx]

            # Fit OHE and medians on fold train only
            ohe_cols  = [c for c in _OHE_CATS if c in df_tr.columns]
            fold_ohe  = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            fold_ohe.fit(df_tr[ohe_cols].astype(str))
            num_cols  = [c for c in df_tr.columns if c not in _ALL_CATS]
            fold_meds = df_tr[num_cols].median()

            # Build embedding vocabulary from fold train only
            cats = sorted(df_tr[_EMB_CAT].astype(str).unique().tolist())
            if 'unknown' not in cats:
                cats = ['unknown'] + cats
            fold_cat_map = {c: i for i, c in enumerate(cats)}

            X_tr_raw, c_tr, _ = _apply_preprocessor(df_tr,  fold_ohe, fold_cat_map, fold_meds)
            X_val_raw, c_val, _ = _apply_preprocessor(df_val, fold_ohe, fold_cat_map, fold_meds)

            scaler = StandardScaler()
            X_tr   = scaler.fit_transform(X_tr_raw)
            X_val  = scaler.transform(X_val_raw)
            cw_t   = _class_weights_tensor(y_tr)

            train_dl, val_dl = _make_loaders(X_tr, c_tr, y_tr, X_val, c_val, y_val, batch_size)
            model    = MLP(X_tr.shape[1], hidden_dims, dropout, activation,
                           n_categories=len(fold_cat_map))
            val_loss = train_model(model, train_dl, val_dl, lr, wd, cw_t)
            fold_losses.append(val_loss)

            trial.report(float(np.mean(fold_losses)), fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_losses))
    return objective


# ── Evaluation Plots ──────────────────────────────────────────────────────────

def _plot_per_score(y_true, y_pred_class, per_score):
    scores = sorted(per_score.keys())
    accs   = [per_score[s]['accuracy'] for s in scores]
    labels = [f'{s}★' for s in scores]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Bar: per-class accuracy
    bars = ax1.bar(labels, accs, color='#4C72B0')
    ax1.set_ylim(0, 1.0)
    ax1.set_xlabel('True Review Score')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Per-Class Accuracy')
    for bar, a in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{a:.2f}', ha='center', va='bottom', fontsize=9)

    # Confusion matrix
    cm     = confusion_matrix(y_true, y_pred_class, labels=scores)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_pct, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
                xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title('Confusion Matrix (row-normalised)')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'dnn_per_score.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df, y = _load_raw()
    print(f"  Rows: {len(df)}  |  target: [{y.min():.0f}, {y.max():.0f}]")
    print(f"  Score distribution: { {int(s): int((y == s).sum()) for s in range(1, 6)} }")

    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=SEED, stratify=y.astype(int),
    )

    # Split raw DataFrames BEFORE fitting preprocessor — val_f rows must not influence
    # the OHE category lists, median values, or scaler statistics.
    print("Splitting train → 90% train_f / 10% val_f (raw DataFrames, before preprocessing)...")
    df_tr_f, df_val_f, y_tr_f, y_val_f = train_test_split(
        df_train, y_train,
        test_size=0.1, random_state=SEED, stratify=y_train.astype(int),
    )

    print("Fitting preprocessor on 90% train_f partition only...")
    ohe, cat_index_map, medians = _fit_preprocessor(df_tr_f)
    X_tr_f_raw,  c_tr_f,  feature_names = _apply_preprocessor(df_tr_f,  ohe, cat_index_map, medians)
    X_val_f_raw, c_val_f, _             = _apply_preprocessor(df_val_f, ohe, cat_index_map, medians)
    X_test_raw,  c_test,  _             = _apply_preprocessor(df_test,  ohe, cat_index_map, medians)
    n_categories = len(cat_index_map)
    print(f"  Dense features: {X_tr_f_raw.shape[1]}  |  Categories (embedding): {n_categories}  "
          f"|  Train_f: {len(X_tr_f_raw)}  |  Val_f: {len(X_val_f_raw)}  |  Test: {len(X_test_raw)}")

    print(f"\nOptuna search: {N_TRIALS} trials, K={N_FOLDS} stratified folds each...")
    study = optuna.create_study(
        direction='minimize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(
        _make_objective(df_train, y_train),
        n_trials=N_TRIALS, show_progress_bar=True,
    )

    best_p = study.best_trial.params
    print(f"\nBest CV combined loss : {study.best_trial.value:.4f}")
    print(f"Best params     : {best_p}")

    # Final retraining: scaler fitted on train_f only; val_f and test are transform-only
    print("\nRetraining on 90% train_f partition (10% val_f for early stopping, test locked)...")
    scaler    = StandardScaler()
    X_tr_f_s  = scaler.fit_transform(X_tr_f_raw)
    X_val_f_s = scaler.transform(X_val_f_raw)
    X_te_s    = scaler.transform(X_test_raw)
    cw_t      = _class_weights_tensor(y_tr_f)

    hidden_dims = [best_p['hidden_dim']] * best_p['n_layers']
    model = MLP(X_tr_f_s.shape[1], hidden_dims, best_p['dropout'], best_p['activation'],
                n_categories=n_categories)
    train_dl, val_dl = _make_loaders(
        X_tr_f_s, c_tr_f, y_tr_f, X_val_f_s, c_val_f, y_val_f, best_p['batch_size'],
    )
    train_model(model, train_dl, val_dl, best_p['lr'], best_p['weight_decay'], cw_t)

    model.eval()
    with torch.no_grad():
        logits      = model(torch.FloatTensor(X_te_s), torch.LongTensor(c_test))
        probs       = F.softmax(logits, dim=1).numpy()       # (n, 5)
        y_pred_class = logits.argmax(dim=1).numpy() + 1      # 1-5 predicted star

    y_test_int = y_test.astype(int)

    acc  = float(accuracy_score(y_test_int, y_pred_class))
    mae  = float(mean_absolute_error(y_test_int, y_pred_class))
    qwk  = float(cohen_kappa_score(y_test_int, y_pred_class, weights='quadratic'))

    print(f"\n── Test Set ──")
    print(f"Accuracy : {acc:.4f}")
    print(f"MAE      : {mae:.4f}  (stars off on average)")
    print(f"QWK      : {qwk:.4f}  (quadratic weighted kappa)")

    print("\nPer-score breakdown:")
    per_score = {}
    for score in range(1, 6):
        mask = y_test_int == score
        if mask.sum() == 0:
            continue
        ps_acc  = float(accuracy_score(y_test_int[mask], y_pred_class[mask]))
        ps_mae  = float(mean_absolute_error(y_test_int[mask], y_pred_class[mask]))
        ps_mode = int(np.bincount(y_pred_class[mask]).argmax())   # most common prediction
        per_score[score] = {
            'accuracy': ps_acc, 'mae': ps_mae,
            'pred_mode': ps_mode, 'n': int(mask.sum()),
        }
        print(f"  {score}★  acc={ps_acc:.3f}  mae={ps_mae:.3f}  "
              f"pred_mode={ps_mode}★  n={mask.sum()}")

    _plot_per_score(y_test_int, y_pred_class, per_score)

    # Save artifacts
    results = {
        'best_params': best_p,
        'cv_ce':       study.best_trial.value,
        'test_accuracy': acc,
        'test_mae':    mae,
        'test_qwk':    qwk,
        'per_score':   per_score,
    }
    with open(os.path.join(RESULTS_DIR, 'dnn_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'best_model.pt'))
    with open(os.path.join(RESULTS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(RESULTS_DIR, 'ohe.pkl'), 'wb') as f:
        pickle.dump(ohe, f)
    with open(os.path.join(RESULTS_DIR, 'cat_index_map.json'), 'w') as f:
        json.dump(cat_index_map, f, indent=2)
    medians.to_json(os.path.join(RESULTS_DIR, 'feature_medians.json'))
    np.save(os.path.join(RESULTS_DIR, 'feature_names.npy'), np.array(feature_names))
    print(f"\nArtifacts saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
