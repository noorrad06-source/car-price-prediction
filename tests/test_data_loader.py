"""
Unit Tests for CarDataLoader Preprocessing and Regex Extraction.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure project root is available in module search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import CarDataLoader


class TestCarDataLoader:
    """Test suite for data loading, sanitization, and regex extractions."""

    @pytest.fixture
    def loader(self) -> CarDataLoader:
        return CarDataLoader(reference_year=2026, max_price_threshold=250000.0)

    def test_clean_price(self, loader: CarDataLoader) -> None:
        raw_prices = pd.Series(["$10,300", "$38,005", " $184,606 ", "$2,954,083", "invalid", np.nan])
        cleaned = loader.clean_price(raw_prices)

        assert cleaned.iloc[0] == 10300.0
        assert cleaned.iloc[1] == 38005.0
        assert cleaned.iloc[2] == 184606.0
        assert cleaned.iloc[3] == 2954083.0
        assert pd.isna(cleaned.iloc[4])
        assert pd.isna(cleaned.iloc[5])

    def test_clean_mileage(self, loader: CarDataLoader) -> None:
        raw_mileage = pd.Series(["51,000 mi.", "34,742 mi.", " 136,397 mi. ", "100 mi.", np.nan])
        cleaned = loader.clean_mileage(raw_mileage)

        assert cleaned.iloc[0] == 51000.0
        assert cleaned.iloc[1] == 34742.0
        assert cleaned.iloc[2] == 136397.0
        assert cleaned.iloc[3] == 100.0
        assert pd.isna(cleaned.iloc[4])

    def test_extract_horsepower(self, loader: CarDataLoader) -> None:
        raw_engines = pd.Series([
            "300.0HP 3.7L V6 Cylinder Engine Flex Fuel Capability",
            "354.0HP 3.5L V6 Cylinder Engine Gas/Electric Hybrid",
            "534.0HP Electric Motor Electric Fuel System",
            "382.0HP 3.0L Straight 6 Cylinder Engine Gasoline Fuel",
            "2.4 Liter",
            "V6",
            np.nan,
        ])
        hp = loader.extract_horsepower(raw_engines)

        assert hp.iloc[0] == 300.0
        assert hp.iloc[1] == 354.0
        assert hp.iloc[2] == 534.0
        assert hp.iloc[3] == 382.0
        assert pd.isna(hp.iloc[4])
        assert pd.isna(hp.iloc[5])
        assert pd.isna(hp.iloc[6])

    def test_extract_engine_displacement(self, loader: CarDataLoader) -> None:
        raw_engines = pd.Series([
            "300.0HP 3.7L V6 Cylinder Engine Flex Fuel Capability",
            "3.8L V6 24V GDI DOHC",
            "3.5 Liter DOHC",
            "2.0L I4 16V GDI DOHC Turbo",
            "2.4 Liter",
            "715.0HP 5.2L 12 Cylinder Engine Gasoline Fuel",
            "Electric Motor",
            np.nan,
        ])
        disp = loader.extract_engine_displacement(raw_engines)

        assert disp.iloc[0] == 3.7
        assert disp.iloc[1] == 3.8
        assert disp.iloc[2] == 3.5
        assert disp.iloc[3] == 2.0
        assert disp.iloc[4] == 2.4
        assert disp.iloc[5] == 5.2
        assert pd.isna(disp.iloc[6])
        assert pd.isna(disp.iloc[7])

    def test_calculate_vehicle_age(self, loader: CarDataLoader) -> None:
        years = pd.Series([2024, 2020, 2017, 2010, 2000])
        age = loader.calculate_vehicle_age(years)

        assert age.iloc[0] == 2   # 2026 - 2024
        assert age.iloc[1] == 6   # 2026 - 2020
        assert age.iloc[2] == 9   # 2026 - 2017
        assert age.iloc[3] == 16  # 2026 - 2010
        assert age.iloc[4] == 26  # 2026 - 2000

    def test_impute_missing_values(self, loader: CarDataLoader) -> None:
        df = pd.DataFrame({
            "fuel_type": ["Gasoline", np.nan, "Hybrid"],
            "accident": ["None reported", np.nan, "At least 1 accident or damage reported"],
            "clean_title": ["Yes", np.nan, "Yes"],
            "hp": [300.0, np.nan, 400.0],
            "engine_liter": [3.0, np.nan, 5.0],
            "milage": [10000.0, 20000.0, np.nan],
        })

        imputed = loader.impute_missing_values(df)

        assert imputed["fuel_type"].iloc[1] == "Unknown"
        assert imputed["accident"].iloc[1] == "None reported"
        assert imputed["clean_title"].iloc[1] == "No"
        assert imputed["hp"].iloc[1] == 350.0          # Median of 300 and 400
        assert imputed["engine_liter"].iloc[1] == 4.0  # Median of 3 and 5
        assert imputed["milage"].iloc[2] == 15000.0    # Median of 10000 and 20000
