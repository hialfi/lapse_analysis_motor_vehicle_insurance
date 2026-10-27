# A Comparative Study of Logistic Regression, Random Forest, and LightGBM for Motor Insurance Lapse Prediction

## Overview

This repository contains the R code and accompanying spreadsheet for the study *A Comparative Study of Logistic Regression, Random Forest, and LightGBM for Motor Insurance Lapse Prediction*.

The study compares three classification models to predict whether a motor insurance policyholder renews or lapses their policy. Model performance is evaluated using multiple classification metrics to account for the class imbalance in the dataset and the different implications of prediction errors.

The analysis also examines feature importance in the LightGBM model to identify variables that contribute to its predictions.

## Research Objectives

The objectives of this study are to:

- Compare the predictive performance of Logistic Regression, Random Forest, and LightGBM for motor insurance policy renewal prediction.
- Evaluate model performance using accuracy, precision, recall, specificity, F1-score, and area under the receiver operating characteristic curve (AUC).
- Examine the predictive importance of policyholder, claims, premium, and vehicle characteristics using LightGBM feature importance.

## Dataset

The study uses the publicly available dataset published by Lledó and Pavía (2024), *Dataset of an Actual Motor Vehicle Insurance Portfolio*.

- **Source:** [Mendeley Data](https://doi.org/10.17632/5cxyb5fp4f.2)
- **Portfolio period:** November 2015–December 2018
- **Number of observations:** 105,555
- **Number of original variables:** 30

The dataset contains motor insurance policy transactions from a Portuguese insurance portfolio.

The response variable, `Buy`, is coded as follows:

- `1`: Policy renewed
- `0`: Policy lapsed

The predictors include policyholder characteristics, insurance tenure, claims history, premium, vehicle characteristics, payment method, and other policy-related variables. Additional variables are derived during preprocessing.

No missing values were present in the dataset used for the analysis; therefore, no observations were removed through missing-value handling.

## Methodology

### Data preprocessing

The preprocessing steps include:

- Defining the binary renewal outcome.
- Deriving policyholder age and driving experience from date-related variables.
- Constructing `Claim_Frequency` and `Premium_per_Value`.
- Preparing categorical predictors for modelling.
- Splitting the data into training and testing sets using a 70:30 ratio.

The same train-test partition is used to compare the three models.

### Prediction models

The study evaluates the following models:

1. **Logistic Regression** — a statistical baseline for binary classification.
2. **Random Forest** — an ensemble of decision trees, fitted using 500 trees.
3. **LightGBM** — a gradient boosting framework
