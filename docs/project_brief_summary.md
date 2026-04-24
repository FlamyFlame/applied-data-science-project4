# Project 4 Brief Summary

Source: [project_4.pdf](/Users/yuhanguo/Documents/stats_courses/applied_data_science/project4/project_4.pdf)

## What the project is asking for

The assignment is an end-to-end machine learning project built from a real-world dataset that you collect and prepare yourselves. The workflow should cover:

1. Data acquisition and cleaning
2. Exploratory data analysis
3. Unsupervised learning during EDA or preprocessing
4. Feature engineering and preprocessing
5. At least three distinct supervised learning models
6. Model comparison, final selection, and communication

## Deliverables

- Final report describing tasks 1 to 5 and the overall story
- Code in a GitHub repository, with documentation and a `README.md`
- Raw and processed datasets
- Optional enhancement such as a dashboard or web app
- A 10-minute oral presentation

## Important details

- Due date: May 5 at 11:59 PM
- The report must include each member's contribution.
- The report should be understandable to an informed but non-specialist audience.
- The rubric rewards data complexity, thoughtful preprocessing, strong validation, model comparison, and clear communication.

## Interpretation of the "three distinct supervised models" requirement

The assignment says to "build at least three distinct supervised learning models." The strongest and safest interpretation is three different model families, not three versions of the same architecture with different hyperparameters.

What should count well:

- Logistic regression plus gradient-boosted trees plus a feed-forward neural network
- Random forest plus support vector machine plus XGBoost
- Linear regression plus random forest regressor plus neural network regressor

What probably should not be your main claim:

- Three multilayer perceptrons that differ only in depth, width, or dropout
- The same tree model with different hyperparameters

Different hyperparameter settings are still useful, but they are better presented as tuning within one model family rather than as three distinct models.
