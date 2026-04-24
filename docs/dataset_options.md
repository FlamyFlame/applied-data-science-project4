# Open E-Commerce Dataset Options

## Recommended choice: Olist Brazilian E-Commerce Public Dataset

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

## Strong backup choice: UCI Online Retail II

Link: [UCI - Online Retail II](https://archive.ics.uci.edu/dataset/502/online%2Bretail%2Bii)

Why it is a good backup:

- More than 1 million transactions over two years
- UCI explicitly lists classification, regression, and clustering as suitable tasks
- Includes missing values and cancellations, so cleaning is still meaningful
- Excellent for customer segmentation, demand-style targets, and repeat-purchase style problems

Tradeoff versus Olist:

- Simpler schema and fewer business dimensions
- Easier to start quickly
- Slightly less rich for a compelling end-to-end story than Olist

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
