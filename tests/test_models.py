"""
Unit Tests for Feature Engineering, Regression Pipelines, and Metrics.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure project root is available in module search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import compute_regression_metrics
from src.feature_engineering import build_preprocessor, get_feature_names
from src.models import (
    build_regression_pipelines,
    extract_feature_importances,
    split_data,
)


@pytest.fixture
def sample_car_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provide a minimal realistic dataset of cars for pipeline verification."""
    data = {
        "milage": [20000.0, 50000.0, 85000.0, 12000.0, 45000.0, 95000.0, 15000.0, 70000.0],
        "model_year": [2022, 2019, 2016, 2023, 2020, 2015, 2022, 2018],
        "vehicle_age": [4, 7, 10, 3, 6, 11, 4, 8],
        "hp": [300.0, 250.0, 180.0, 450.0, 280.0, 160.0, 320.0, 210.0],
        "engine_liter": [3.0, 2.5, 2.0, 4.0, 3.5, 1.8, 3.0, 2.4],
        "brand": ["Ford", "Toyota", "Honda", "BMW", "Ford", "Toyota", "Audi", "Honda"],
        "fuel_type": ["Gasoline", "Hybrid", "Gasoline", "Gasoline", "Gasoline", "Gasoline", "Gasoline", "Hybrid"],
        "transmission": ["Automatic", "A/T", "CVT", "Automatic", "A/T", "Manual", "Automatic", "CVT"],
        "accident": ["None reported", "None reported", "At least 1 accident or damage reported", "None reported", "None reported", "None reported", "None reported", "None reported"],
        "clean_title": ["Yes", "Yes", "Yes", "Yes", "Yes", "No", "Yes", "Yes"],
    }
    prices = pd.Series([32000.0, 24000.0, 16000.0, 65000.0, 38000.0, 12000.0, 52000.0, 21000.0])
    return pd.DataFrame(data), prices


class TestRegressionPipelines:
    """Test suite for feature engineering, model training, and metrics."""

    def test_preprocessor_construction(self, sample_car_data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, _ = sample_car_data
        num_cols = ["milage", "model_year", "vehicle_age", "hp", "engine_liter"]
        cat_cols = ["brand", "fuel_type", "transmission", "accident", "clean_title"]

        preprocessor = build_preprocessor(num_cols, cat_cols, sparse_output=False)
        X_trans = preprocessor.fit_transform(X)

        assert X_trans.shape[0] == len(X)
        assert X_trans.shape[1] >= len(num_cols) + len(cat_cols)

        feature_names = get_feature_names(preprocessor, num_cols, cat_cols)
        assert len(feature_names) == X_trans.shape[1]

    def test_split_data(self, sample_car_data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = sample_car_data
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25, random_state=42)

        assert len(X_train) == 6
        assert len(X_test) == 2
        assert len(y_train) == 6
        assert len(y_test) == 2

    def test_candidate_pipelines_training_and_predicting(
        self, sample_car_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = sample_car_data
        num_cols = ["milage", "model_year", "vehicle_age", "hp", "engine_liter"]
        cat_cols = ["brand", "fuel_type", "transmission", "accident", "clean_title"]

        preprocessor = build_preprocessor(num_cols, cat_cols, sparse_output=False)
        pipelines = build_regression_pipelines(preprocessor)

        assert "Ridge Regression" in pipelines
        assert "Random Forest Regressor" in pipelines
        assert "HistGradientBoosting Regressor" in pipelines

        for name, pipeline in pipelines.items():
            pipeline.fit(X, y)
            preds = pipeline.predict(X)

            assert len(preds) == len(y)
            assert np.all(np.isfinite(preds))

    def test_feature_importance_extraction(
        self, sample_car_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = sample_car_data
        num_cols = ["milage", "model_year", "vehicle_age", "hp", "engine_liter"]
        cat_cols = ["brand", "fuel_type", "transmission", "accident", "clean_title"]

        preprocessor = build_preprocessor(num_cols, cat_cols, sparse_output=False)
        rf_pipeline = build_regression_pipelines(preprocessor)["Random Forest Regressor"]

        rf_pipeline.fit(X, y)
        fitted_preprocessor = rf_pipeline.named_steps["preprocessor"]
        feature_names = get_feature_names(fitted_preprocessor, num_cols, cat_cols)

        importances = extract_feature_importances(rf_pipeline, feature_names)
        assert not importances.empty
        assert len(importances) == len(feature_names)
        assert importances.iloc[0] >= importances.iloc[-1]

    def test_compute_regression_metrics(self) -> None:
        y_true = np.array([20000.0, 30000.0, 40000.0])
        y_pred = np.array([21000.0, 29000.0, 41000.0])

        metrics = compute_regression_metrics(y_true, y_pred, model_name="TestModel")

        assert metrics["Model"] == "TestModel"
        assert metrics["MAE ($)"] == 1000.0
        assert metrics["RMSE ($)"] == 1000.0
        assert metrics["R2 Score"] > 0.95
        assert metrics["MAPE (%)"] > 0
