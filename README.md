# Predicting Airbnb Listing Prices in Melbourne, Australia

A machine learning project that builds a price prediction model for Airbnb listings in Melbourne, using ensemble methods across nine regression models tuned with randomized search cross-validation.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)

---

## Overview

This project develops a reliable model for predicting optimal Airbnb listing prices in Melbourne, providing data-driven price insights to multiple stakeholders:

- **Hosts** — set competitive yet profitable rates
- **Guests** — assess whether a listing is fairly priced
- **Property Investors** — evaluate location and property attributes
- **Financial Institutions** — incorporate price predictions into risk assessments

**Evaluation Metric:** Mean Absolute Error (MAE) — chosen for its interpretability and robustness to outliers in a right-skewed price distribution.

---

## Dataset

The dataset contains Airbnb listing records split into training and test sets, loaded from `train.csv` and `test.csv`. Both sets are combined before preprocessing to ensure consistent transformations.

**Target Variable:** `price` — nightly listing price in AUD (log-transformed as `log_price` during modelling)

### Feature Summary

| Variable Type | Count | Examples |
|---|---|---|
| Numeric | 35 | `LIMIT_BAL`, `accommodates`, `bedrooms`, `review_scores_rating`, `availability_365` |
| Ordinal | 1 | `host_response_time` |
| Nominal | 16 | `room_type`, `property_type`, `neighbourhood_cleansed`, `host_is_superhost` |
| Date | 3 | `host_since`, `first_review`, `last_review` |
| Geo Coordinates | 2 | `latitude`, `longitude` |
| Text | 3 | `description`, `host_about`, `neighborhood_overview` |

---

## Project Structure

```
├── train.csv
├── test.csv
├── Predicting_Airbnb_Listing_Price_in_Melbourne.ipynb
├── prediction_stacking_top_five.csv
└── README.md
```

---

## Methodology

### 1. Exploratory Data Analysis
- Examined feature types and distributions; identified price as right-skewed with a median of $172 and outliers exceeding $100K
- Applied 1st–99th percentile clipping followed by `np.log1p` transformation to stabilise the target variable
- Visualised missing values across train/test sets using stacked bar charts

### 2. Data Cleaning
- Stripped `$` from `price` and `%` from `host_response_rate` / `host_acceptance_rate`, then cast to float
- Parsed the textual `bathrooms` column (e.g. "1.5 baths", "half bath") into numeric values using a custom conversion function

### 3. Feature Engineering

| New Feature | Source Column | Transformation |
|---|---|---|
| `num_verifications` | `host_verifications` | Count of verification methods per host |
| `days_hosting` | `host_since` | Days elapsed since host registration |
| `num_amenities` | `amenities` | Count of amenities listed per property |
| `region` | `neighbourhood_cleansed` | Mapped suburbs to five Melbourne regions: North, South, East, West, Central |

### 4. Data Imputation
- **Mean** — normally distributed numerical variables
- **Median** — skewed numerical variables
- **Mode** — categorical and ordinal variables

### 5. Encoding & Scaling
- One-hot encoding for nominal variables
- Ordinal encoding for `host_response_time`
- `StandardScaler` + PCA (95% variance retained) applied within pipelines for linear models and SVR

### 6. Train/Test Split
- 85/15 split with `random_state=42`

---

## Models

### Base Regressors (9 models)

All models were hyperparameter-tuned using **RandomizedSearchCV** with:
- 10-fold cross-validation
- 40 iterations per model
- Scoring: negative MAE

| Model | Key Notes |
|---|---|
| Linear Regression | Baseline; no tuning required |
| Ridge Regression | L2 regularisation; PCA pipeline |
| Lasso Regression | L1 regularisation; feature selection |
| Elastic Net | Combined L1 + L2 regularisation |
| Decision Tree | Tuned depth and leaf parameters |
| Random Forest | Tuned estimators, depth, and features |
| Gradient Boosting (GBM) | Tuned learning rate, subsample, depth |
| XGBoost | Tuned regularisation, gamma, child weight |
| LightGBM | Tuned leaves, subsample, lambda |
| SVR | Tuned C, epsilon, kernel; PCA pipeline |

### Ensemble Models (4 variants)

| Ensemble | Description |
|---|---|
| Voting (Top 5) | Average predictions from LightGBM, SVR, GBM, Random Forest, XGBoost |
| Stacking (Top 5) | Same top 5 as base models; Linear Regression as meta-learner |
| Voting (All) | Average predictions from all 10 base models |
| Stacking (All) | All 10 base models; Linear Regression as meta-learner |

---

## Results

### Individual Regressors

| Model | MAE Train | MAE Test |
|---|---|---|
| LightGBM | — | ~57–58 |
| GBM | — | ~57–58 |
| Random Forest | — | ~57–58 |
| XGBoost | — | ~58 |
| SVR | 64.39 | 51.97* |
| Lasso / Ridge / LR | — | High (underfitting) |

*SVR's lower test MAE is attributed to underfitting on a favourable test split rather than true generalisation.

### Ensemble Models

Ensemble strategies outperformed all individual regressors. The **Voting Regressor (Top 5)** was selected as the final model due to:
- Best overall generalisation (lowest average rank across train and test MAE)
- Stable predictions through equal-weighted averaging of five diverse, well-tuned models
- Avoidance of noise introduced by weaker linear models in the "all models" variants

### Final Model

> **VotingRegressor** — XGBoost · Gradient Boosting · SVR · LightGBM · Random Forest

Predictions were generated on the log scale and back-transformed using `np.expm1` before submission.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
shap
geopandas
folium
branca
contextily
textblob
tqdm
scipy
```

Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap geopandas folium branca contextily textblob tqdm scipy
```

---
