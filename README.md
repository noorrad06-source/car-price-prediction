# Used Car Price Prediction Pipeline

An end-to-end, production-ready machine learning regression pipeline designed to predict used vehicle listing prices based on structural specifications, mechanical attributes, mileage, and condition history.

---

## Executive Summary & Empirical Benchmark Results

The pipeline was benchmarked across three distinct regression algorithms on a held-out test set (793 test listings from 3,961 total cleaned records). Evaluation metrics include **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, **$R^2$ Score (Variance Explained)**, and **Mean Absolute Percentage Error (MAPE)**.

| Model Architecture | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | $R^2$ Score | Mean Absolute Percentage Error (MAPE) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **$7,412.40** | **$12,817.25** | **0.8368** | **24.04%** | 🏆 **Selected Winner** |
| **HistGradientBoosting Regressor** | $8,085.09 | $13,919.20 | 0.8075 | 25.11% | Candidate |
| **Ridge Regression** (Linear Baseline) | $12,054.92 | $19,145.25 | 0.6358 | 48.29% | Baseline |

> **Key Takeaway:** The **Random Forest Regressor** emerged as the top-performing model, capturing **83.68% of price variance** ($R^2 = 0.8368$) with an average absolute prediction error of **$7,412.40** (24.04% MAPE) across diverse vehicle categories.

---s

## Project Architecture

```text
car-price-prediction/
├── config/
│   └── config.yaml             # Feature selection, model parameters, and path configs
├── outputs/
│   ├── benchmark_summary.csv   # Model comparison metrics
│   ├── feature_importance.png  # Top price drivers visualization
│   └── actual_vs_predicted.png # Scatter plot of actual vs. predicted prices
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion, numerical cleaning, and regex feature extraction
│   ├── feature_engineering.py  # ColumnTransformer with StandardScaler & OneHotEncoder
│   ├── models.py             # Pipeline orchestrators for Ridge, RF, and HistGradientBoosting
│   └── evaluate.py           # Metrics calculation & visualization utilities
├── tests/
│   ├── test_data_loader.py     # Unit tests for data cleaning & regex regex parsing
│   └── test_models.py          # Unit tests for training pipeline execution
├── main.py                     # Execution entry point
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation