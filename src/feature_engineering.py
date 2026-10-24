"""
Feature Engineering and Preprocessing Pipeline Module.

Constructs scikit-learn ColumnTransformer architectures combining StandardScaler
for numerical variables and OneHotEncoder for categorical attributes.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger("feature_engineering")


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    sparse_output: bool = False,
) -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer integrating numeric scaling
    and categorical one-hot encoding.

    Args:
        numeric_features: List of continuous feature column names.
        categorical_features: List of categorical feature column names.
        sparse_output: Whether OneHotEncoder returns sparse matrix or dense array.

    Returns:
        Unfitted ColumnTransformer instance.
    """
    logger.info(
        f"Constructing ColumnTransformer: {len(numeric_features)} numeric features, "
        f"{len(categorical_features)} categorical features."
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=sparse_output,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_feature_names(
    preprocessor: ColumnTransformer,
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[str]:
    """
    Extract readable feature names after fitting the ColumnTransformer.

    Args:
        preprocessor: Fitted ColumnTransformer instance.
        numeric_features: Original list of numeric columns.
        categorical_features: Original list of categorical columns.

    Returns:
        List of generated feature names.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback manual reconstruction
        names: list[str] = list(numeric_features)
        try:
            ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cat_names = list(ohe.get_feature_names_out(categorical_features))
            names.extend(cat_names)
        except Exception:
            for cat_col in categorical_features:
                names.append(f"{cat_col}_encoded")
        return names
