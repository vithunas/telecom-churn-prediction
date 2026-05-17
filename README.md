# Telecom Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue) ![PySpark](https://img.shields.io/badge/PySpark-ML-purple) ![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

Predict which telecom customers are likely to churn using a PySpark ML pipeline, XGBoost classifier, and an interactive Streamlit dashboard.

## Pipeline
`CSV data` → `PySpark ingestion` → `Feature engineering` → `XGBoost training` → `Streamlit dashboard`

## Results
| Metric | Value |
|--------|-------|
| Model accuracy | ~80% |
| Dataset size | 7,043 customers |
| Churn rate | 26.6% |
| Best model | XGBoost |

## Features
- PySpark pipeline for large-scale data ingestion and feature engineering
- StringIndexer, OneHotEncoder, VectorAssembler transformations
- MLlib RandomForest baseline + XGBoost with GridSearchCV tuning
- Streamlit dashboard: EDA charts, prediction form, model evaluation

## Tech stack
Python · PySpark · XGBoost · Scikit-learn · Streamlit · Plotly

## Run locally
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```
