# Adult Income Prediction — End-to-End Machine Learning System

## Project Overview

This project implements an end-to-end machine learning system
using the UCI Adult Census Income dataset.

The system covers:

1. Data acquisition
2. Data validation
3. Missing-value handling
4. Outlier treatment
5. Numerical feature scaling
6. Categorical encoding
7. Train/test splitting
8. Exploratory visualization
9. Model comparison
10. Model evaluation
11. Feature importance
12. Model serialization
13. REST API
14. Real-time prediction dashboard
15. Docker deployment
16. Automated tests

---

# Dataset

Dataset:

UCI Adult / Census Income

Target:

Income > $50K

Target encoding:

<=50K -> 0

>50K -> 1

Official source:

UCI Machine Learning Repository

---

# Architecture

Raw Dataset
    |
    v
Data Validation
    |
    v
Cleaning
    |
    v
Train/Test Split
    |
    v
Preprocessing Pipeline
    |
    +--> Median Imputation
    |
    +--> Outlier Clipping
    |
    +--> Standard Scaling
    |
    +--> One-Hot Encoding
    |
    v
Model Training
    |
    v
Model Evaluation
    |
    v
Best Model
    |
    +--> REST API
    |
    +--> Streamlit Dashboard

---

# Preprocessing

## Missing Values

Numerical features:

Median imputation.

Categorical features:

Most-frequent imputation.

## Outliers

Numerical values are clipped between
the 1st and 99th percentiles.

The boundaries are learned only
from training data.

## Scaling

Numerical features use StandardScaler.

## Encoding

Categorical variables use OneHotEncoder.

Unknown categories are safely ignored.

---

# Data Leakage Prevention

All preprocessing is contained within
a scikit-learn Pipeline.

The preprocessing pipeline is fitted
only on training data.

The test set is transformed using
parameters learned from training data.

---

# Model Training

The system compares:

- Logistic Regression
- Random Forest
- HistGradientBoosting

The model with the highest ROC-AUC
on the held-out test set is selected.

---

# Evaluation

The project generates:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- ROC curve
- Feature importance

---

# Running the Project

## Install

pip install -r requirements.txt

## Train the complete pipeline

python run.py

## Start API

uvicorn api.main:app --reload

## Start dashboard

streamlit run dashboard/app.py

---

# Docker

docker compose up --build

API:

http://localhost:8000

Dashboard:

http://localhost:8501

Swagger:

http://localhost:8000/docs

---

# Testing

Run:

pytest -q

---

# Generated Artifacts

models/

- best_model.joblib
- preprocessing_pipeline.joblib
- model_metadata.json

outputs/reports/

- model_comparison.csv
- classification_report.csv
- feature_importance.csv

outputs/figures/

- missing-value visualization
- target distribution
- numerical distributions
- outlier analysis
- categorical distributions
- correlation heatmap
- confusion matrix
- ROC curve

---

# Production Considerations

Before production deployment, add:

- authentication
- HTTPS
- rate limiting
- centralized logging
- model monitoring
- data drift detection
- model versioning
- CI/CD
- cloud deployment
- database persistence
- prediction audit logging

---

# Ethical Considerations

Income prediction can encode historical socioeconomic
biases.

The model should therefore not be used as the sole
decision-making mechanism for employment, credit,
insurance, housing, or other high-impact decisions.

Model performance should be evaluated across relevant
subgroups before real-world deployment.