"""
Model Pipeline Architecture and Training Engine.

Constructs self-contained scikit-learn Pipelines combining the ColumnTransformer
preprocessor and candidate regression algorithms: Ridge, Random Forest, and HistGradientBoosting.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

logger = logging.getLogger("models")


def build_regression_pipelines(
    preprocessor: ColumnTransformer,
    hyperparameters: Optional[dict[str, Any]] = None,
) -> dict[str, Pipeline]:
    """
    Construct regression pipelines encapsulating feature preprocessing and models.

    Args:
        preprocessor: Configured ColumnTransformer instance.
        hyperparameters: Dictionary containing model-specific hyperparameters.

    Returns:
        Dictionary mapping model names to un-fitted scikit-learn Pipeline objects.
    """
    hyperparameters = hyperparameters or {}

    # 1. Ridge Regression (Linear Baseline)
    ridge_cfg = hyperparameters.get("ridge", {})
    ridge_model = Ridge(
        alpha=ridge_cfg.get("alpha", 1.0),
        random_state=ridge_cfg.get("random_state", 42),
    )

    # 2. Random Forest Regressor (Ensemble Bagging)
    rf_cfg = hyperparameters.get("random_forest", {})
    rf_model = RandomForestRegressor(
        n_estimators=rf_cfg.get("n_estimators", 150),
        max_depth=rf_cfg.get("max_depth", 16),
        min_samples_split=rf_cfg.get("min_samples_split", 4),
        random_state=rf_cfg.get("random_state", 42),
        n_jobs=rf_cfg.get("n_jobs", -1),
    )

    # 3. HistGradientBoosting Regressor (Ensemble Boosting)
    hgb_cfg = hyperparameters.get("hist_gradient_boosting", {})
    hgb_model = HistGradientBoostingRegressor(
        max_iter=hgb_cfg.get("max_iter", 150),
        learning_rate=hgb_cfg.get("learning_rate", 0.08),
        max_depth=hgb_cfg.get("max_depth", 8),
        random_state=hgb_cfg.get("random_state", 42),
    )

    pipelines = {
        "Ridge Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", ridge_model),
            ]
        ),
        "Random Forest Regressor": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", rf_model),
            ]
        ),
        "HistGradientBoosting Regressor": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", hgb_model),
            ]
        ),
    }

    return pipelines


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into training and testing partitions.

    Args:
        X: Feature matrix.
        y: Target prices.
        test_size: Proportion of test partition (default: 0.20).
        random_state: Reproducibility seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    logger.info(
        f"Partitioning dataset ({len(X)} rows) with test_size={test_size:.0%}, "
        f"random_state={random_state}..."
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    logger.info(
        f"Partitioning complete: {len(X_train)} training records, "
        f"{len(X_test)} testing records."
    )
    return X_train, X_test, y_train, y_test


def extract_feature_importances(
    pipeline: Pipeline,
    feature_names: list[str],
) -> pd.Series:
    """
    Extract or calculate feature importances from a trained regression pipeline.

    Args:
        pipeline: Fitted scikit-learn Pipeline with a 'regressor' step.
        feature_names: List of all transformed feature names.

    Returns:
        pandas Series of importances indexed by feature name, sorted descending.
    """
    regressor = pipeline.named_steps["regressor"]
    importances = None

    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
    elif hasattr(regressor, "coef_"):
        importances = np.abs(regressor.coef_)

    if importances is not None and len(importances) == len(feature_names):
        s = pd.Series(importances, index=feature_names)
        return s.sort_values(ascending=False)

    return pd.Series(dtype=float)
