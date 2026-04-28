# Applied Data Science — Project 4

End-to-end machine learning on the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).  
The pipeline covers data cleaning, EDA, unsupervised feature engineering, and three supervised models.

## Team 22

| Name | UNI |
|---|---|
| Yuhan Guo | `yg2695` |
| Bohong Zheng | `bz2575` |
| Xiang Li | `xl3548` |
| Jiahao Wang | `jw4811` |

---

## Requirements

Python 3.12. Install all dependencies into a virtual environment:

```bash
pip install pandas numpy scikit-learn torch optuna xgboost matplotlib seaborn
```

---

## Data Setup

1. Download the 9 raw Olist CSV files from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
2. Place them directly in `processed/` (alongside any previously generated `clean_df_final.csv`):

```
processed/
  olist_orders_dataset.csv
  olist_order_items_dataset.csv
  olist_order_reviews_dataset.csv
  olist_order_payments_dataset.csv
  olist_products_dataset.csv
  olist_sellers_dataset.csv
  olist_customers_dataset.csv
  olist_geolocation_dataset.csv
  product_category_name_translation.csv
```

---

## Running the Pipeline

### Full end-to-end (recommended)

From the project root:

```bash
python pipeline.py
```

This runs all four steps in sequence and stops with a non-zero exit code if any step fails.

### Individual steps

#### 1 — Data cleaning & EDA

```bash
python Data_clean_EDA_unsupervised.py
```

Reads the 9 raw CSVs from `processed/`, merges them, engineers features
(haversine distance, delivery days, delay, freight ratio, KMeans logistics clusters),
removes outliers with Isolation Forest, and writes **`processed/clean_df_final.csv`**
(~109K rows, 46 columns).

All downstream model scripts require this file to exist before running.

#### 2 — Logistic Regression

```bash
python src/logistic_regression.py
```

Binary classification baseline (`is_bad_review`: review score ≤ 2).  
Reads `processed/clean_df_final.csv`, fits a `OneHotEncoder` + `StandardScaler`
on the training partition, and evaluates with accuracy, F1, and ROC-AUC.  
Artifacts saved to `results/logistic_regression/`.

> **TODO:** tune `C`, `penalty`, and `class_weight` via cross-validated search
> (see `TODO` comments in `src/logistic_regression.py`).

#### 3 — XGBoost

```bash
python src/xgboost_model.py
```

Gradient-boosted tree regression on `review_score` (1–5), enabling direct
RMSE/MAE/R² comparison with the DNN.  
Artifacts saved to `results/xgboost/`.

> **TODO:** tune `n_estimators`, `max_depth`, `learning_rate`, `subsample`, etc.
> (see `TODO` comments in `src/xgboost_model.py`).

#### 4 — Feed-Forward Neural Network (DNN)

```bash
python src/dnn_pipeline.py
```

MLP **ordinal-regularized multi-class classification** of `review_score` (1–5 stars) with Optuna Bayesian
hyperparameter search (20 trials, K=5 stratified cross-validation).

Key design choices:
- `seller_state` / `customer_state` → one-hot encoded (49 cols total)
- `product_category_name` → 8-dim learned embedding (`nn.Embedding`)
- Dense input: 71 features; total input to first hidden layer: 79
- Output: 5 logits → `argmax + 1` → predicted star rating (1–5)
- Loss: class-weighted cross-entropy + 0.5 × ordinal MSE on expected score
- Class weights: inverse-frequency (`2★` receives about `6x` the weight of an average class; `5★` about `0.35x`)
- Final retraining uses a 10% internal val split; test set never touches `train_model`

Artifacts saved to `results/`.

---

## Scoring New Orders

After the DNN pipeline has been run at least once (so `results/` contains all artifacts):

```bash
python src/inference.py path/to/new_orders.csv --out predictions.csv
```

Input must be a CSV in the same format as `processed/clean_df_final.csv`.  
Output is a single-column CSV with `predicted_review_score` (integer 1–5).

---

## Repository Structure

```
├── pipeline.py                        # end-to-end entry point
├── Data_clean_EDA_unsupervised.py     # step 1: cleaning, EDA, feature engineering
├── src/
│   ├── logistic_regression.py         # step 2: LR baseline
│   ├── xgboost_model.py               # step 3: XGBoost
│   ├── dnn_pipeline.py                # step 4: MLP with Optuna search
│   └── inference.py                   # score new orders with saved DNN artifacts
├── processed/                         # raw CSVs + clean_df_final.csv (not in git)
├── results/                           # trained model artifacts (not in git)
│   ├── logistic_regression/
│   ├── xgboost/
│   └── (DNN artifacts at root level)
├── reports/
│   └── final_report.tex
├── docs/                              # planning notes (not in git)
├── figures/
└── notebooks/
```

---

## Branch Structure

| Branch | Content |
|---|---|
| `main` | shared pipeline, report template, all merged model code |
| `eda-unsupervised-learning` | data cleaning, EDA, clustering (merged) |
| `feed-forward-neural-network` | DNN implementation (merged) |
| `logistic-regression` | LR implementation |
| `gradient-boosted-trees` | XGBoost implementation |

---

## Notes

- The project requires at least three distinct supervised models (LR, XGBoost, DNN).
- `processed/` and `results/` are git-ignored; data files must be downloaded locally.
- The final report is in `reports/final_report.tex`.
- Each team member's contribution must be documented in the report.
