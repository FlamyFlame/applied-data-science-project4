# Chosen Dataset and Model Plan

## Final dataset: Olist Brazilian E-Commerce Public Dataset

Link: [Kaggle - Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

Why it fits this assignment well:

- Roughly 100,000 orders across multiple linked tables
- Rich order, payment, freight, customer, seller, product, and review information
- Enough messiness and relational structure to score well on data acquisition and preprocessing
- Strong opportunities for EDA, clustering, dimensionality reduction, and feature engineering
- Multiple reasonable supervised targets can be constructed from the data

Good predictive questions for Olist:

- Predict whether an order will receive a low review score
- Predict whether a delivery will be late
- Predict total order value
- Predict whether a customer will make a repeat purchase after a defined window

Good unsupervised angles for Olist:

- Customer segmentation from RFM-style features
- Seller segmentation by fulfillment and review performance
- PCA or UMAP on engineered behavioral features
- Clustering product categories or regional purchasing patterns

## Recommended modeling trio

Assuming you choose a tabular prediction problem such as low-review prediction or late-delivery prediction, a strong set of three supervised models is:

1. Logistic regression or elastic-net logistic regression
2. Gradient-boosted trees such as XGBoost, LightGBM, or CatBoost
3. Feed-forward neural network for tabular data

Why this trio works:

- It gives you clearly different model families
- It balances interpretability and predictive power
- It creates a meaningful comparison section in the report

## Short answer to the model requirement question

Use three genuinely different model families. Different DNN hyperparameters are useful for tuning, but they should usually count as variants of one model family, not as the required three distinct supervised models.
