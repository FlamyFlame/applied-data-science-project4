# Applied Data Science Project 4

This repository is a starter scaffold for the Project 4 end-to-end machine learning workflow. The project uses the Olist Brazilian e-commerce dataset and follows a pipeline that includes data acquisition, cleaning, EDA, unsupervised learning, feature engineering, supervised modeling, model comparison, and reporting.

## Suggested branch structure

- `main`: shared scaffolding, report template, and project notes
- `eda-unsupervised-learning`: data cleaning, EDA, clustering, dimensionality reduction
- `logistic-regression`: baseline linear classifier
- `gradient-boosted-trees`: boosted tree model family
- `feed-forward-neural-network`: tabular neural network model family

## Recommended folders

- `data/raw/`: original downloaded data
- `data/processed/`: cleaned and engineered data
- `docs/`: project notes and planning documents
- `figures/`: plots and exported visuals for the report
- `notebooks/`: exploratory notebooks
- `reports/`: LaTeX report and compiled outputs
- `src/`: reusable Python or R code

## Quick start

1. Save the Olist raw files under `data/raw/`.
2. Document the join keys and table relationships before modeling.
3. Do initial cleaning and EDA on `eda-unsupervised-learning`.
4. Keep each supervised model family on its own branch so experiments stay organized.
5. Merge the strongest ideas back into `main` for the final report and presentation.

## Notes

- The project brief requires at least three distinct supervised models.
- The final report must include each team member's contribution.
- The submission must include a GitHub repository with code and a `README.md`.
