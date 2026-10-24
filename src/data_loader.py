"""
Data Ingestion, Cleaning, and Feature Extraction Module.

Handles loading of raw used car listings, regex extraction of engine specifications,
numerical cleaning of price and mileage, outlier removal, and missing value imputation.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure project root is available in module path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("data_loader")

# Precompiled regex patterns for engine specification extraction
HORSEPOWER_REGEX = re.compile(r"(\d+(?:\.\d+)?)\s*(?:hp|horsepower)\b", re.IGNORECASE)
DISPLACEMENT_REGEX = re.compile(r"(\d+(?:\.\d+)?)\s*(?:l|liter|liters)\b", re.IGNORECASE)


class CarDataLoader:
    """
    Ingests and cleans used vehicle datasets, applying feature extraction
    and preprocessing transformations.
    """

    def __init__(
        self,
        reference_year: int = 2026,
        max_price_threshold: float = 250000.0,
        categorical_fill: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Initialize CarDataLoader with domain-specific thresholds.

        Args:
            reference_year: Baseline year for vehicle age calculation (default: 2026).
            max_price_threshold: Upper price cap to filter exotic/luxury outliers.
            categorical_fill: Mapping of column names to default imputation strings.
        """
        self.reference_year = reference_year
        self.max_price_threshold = max_price_threshold
        self.categorical_fill = categorical_fill or {
            "fuel_type": "Unknown",
            "accident": "None reported",
            "clean_title": "No",
        }

    @staticmethod
    def clean_price(price_series: pd.Series) -> pd.Series:
        """
        Sanitize price string values by stripping currency symbols and commas.

        Args:
            price_series: Raw price column.

        Returns:
            Float-converted price Series.
        """
        return (
            price_series.astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
            .pipe(pd.to_numeric, errors="coerce")
        )

    @staticmethod
    def clean_mileage(mileage_series: pd.Series) -> pd.Series:
        """
        Sanitize mileage string values by stripping 'mi.' and commas.

        Args:
            mileage_series: Raw mileage column.

        Returns:
            Float-converted mileage Series.
        """
        return (
            mileage_series.astype(str)
            .str.replace(" mi.", "", regex=False)
            .str.replace("mi.", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
            .pipe(pd.to_numeric, errors="coerce")
        )

    @staticmethod
    def extract_horsepower(engine_series: pd.Series) -> pd.Series:
        """
        Extract numerical horsepower (HP) from descriptive engine text using regex.

        Args:
            engine_series: String column describing vehicle engine.

        Returns:
            Float Series with extracted horsepower values (or NaN if unmatched).
        """
        def _match_hp(val: Any) -> float | None:
            if not isinstance(val, str) or pd.isna(val):
                return None
            match = HORSEPOWER_REGEX.search(val)
            return float(match.group(1)) if match else None

        return engine_series.apply(_match_hp)

    @staticmethod
    def extract_engine_displacement(engine_series: pd.Series) -> pd.Series:
        """
        Extract numerical engine displacement in liters (L) from descriptive engine text.

        Args:
            engine_series: String column describing vehicle engine.

        Returns:
            Float Series with extracted engine liters (or NaN if unmatched).
        """
        def _match_disp(val: Any) -> float | None:
            if not isinstance(val, str) or pd.isna(val):
                return None
            match = DISPLACEMENT_REGEX.search(val)
            return float(match.group(1)) if match else None

        return engine_series.apply(_match_disp)

    def calculate_vehicle_age(self, year_series: pd.Series) -> pd.Series:
        """
        Calculate vehicle age based on reference year: vehicle_age = reference_year - model_year.

        Args:
            year_series: Model year numerical column.

        Returns:
            Vehicle age in years.
        """
        return self.reference_year - pd.to_numeric(year_series, errors="coerce")

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values according to production requirements:
        - Categorical features ('fuel_type', 'accident', 'clean_title') imputed with domain defaults.
        - Extracted numeric features ('hp', 'engine_liter') imputed with median values.

        Args:
            df: DataFrame containing raw/extracted features.

        Returns:
            DataFrame with imputed columns.
        """
        processed_df = df.copy()

        # Categorical imputation
        for col, default_val in self.categorical_fill.items():
            if col in processed_df.columns:
                missing_count = processed_df[col].isna().sum()
                processed_df[col] = processed_df[col].fillna(default_val).astype(str)
                logger.debug(f"Imputed {missing_count} missing values in '{col}' with '{default_val}'.")

        # Numerical median imputation for extracted engine specs
        for num_col in ["hp", "engine_liter", "milage"]:
            if num_col in processed_df.columns:
                missing_count = processed_df[num_col].isna().sum()
                if missing_count > 0:
                    med_val = processed_df[num_col].median()
                    processed_df[num_col] = processed_df[num_col].fillna(med_val)
                    logger.debug(f"Imputed {missing_count} missing values in '{num_col}' with median {med_val:.2f}.")

        return processed_df

    def load_and_preprocess(
        self,
        source_path: str | Path,
        target_column: str = "price",
        numeric_features: Optional[list[str]] = None,
        categorical_features: Optional[list[str]] = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Full data loading and preprocessing lifecycle.

        Args:
            source_path: Path to 'used_cars.csv'.
            target_column: Name of target price column.
            numeric_features: List of numeric features to retain.
            categorical_features: List of categorical features to retain.

        Returns:
            Tuple of (Feature DataFrame X, Target Series y).
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source dataset not found at '{source_path}'.")

        logger.info(f"Loading raw dataset from '{source_path}'...")
        raw_df = pd.read_csv(source_path)
        logger.info(f"Loaded {len(raw_df)} listings across {len(raw_df.columns)} columns.")

        # 1. Clean Target Variable
        df = raw_df.copy()
        df["price_clean"] = self.clean_price(df[target_column])

        # 2. Outlier Filtering: Filter extreme luxury listings price > $250,000 and invalid prices
        initial_count = len(df)
        valid_mask = (df["price_clean"] > 0) & (df["price_clean"] <= self.max_price_threshold)
        df = df[valid_mask].copy()
        filtered_out = initial_count - len(df)
        logger.info(
            f"Filtered {filtered_out} outlier/invalid price listings (> ${self.max_price_threshold:,.0f}). "
            f"Retained {len(df)} listings."
        )

        # 3. Clean Mileage
        df["milage"] = self.clean_mileage(df["milage"])

        # 4. Extract Engine Specifications using Regular Expressions
        logger.info("Extracting horsepower (hp) and engine displacement (engine_liter) via regex...")
        df["hp"] = self.extract_horsepower(df["engine"])
        df["engine_liter"] = self.extract_engine_displacement(df["engine"])

        # 5. Feature Engineering: Vehicle Age
        df["vehicle_age"] = self.calculate_vehicle_age(df["model_year"])

        # 6. Missing Value Imputation
        df = self.impute_missing_values(df)

        # Default feature selections
        numeric_cols = numeric_features or ["milage", "model_year", "vehicle_age", "hp", "engine_liter"]
        cat_cols = categorical_features or ["brand", "fuel_type", "transmission", "accident", "clean_title"]

        feature_cols = numeric_cols + cat_cols
        X = df[feature_cols].copy()
        y = df["price_clean"].copy()

        logger.info(
            f"Preprocessing complete. Matrix shape: X={X.shape}, y={y.shape}. "
            f"Target Price range: ${y.min():,.0f} to ${y.max():,.0f} (Mean: ${y.mean():,.0f}, Median: ${y.median():,.0f})."
        )
        return X, y
