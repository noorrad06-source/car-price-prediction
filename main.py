"""
End-to-End Regression Pipeline Orchestrator for Car Price Prediction.

Coordinates data loading, regex engine extraction, numeric and target cleaning,
outlier filtering, ColumnTransformer preprocessing, multi-model pipeline training,
regression benchmarking, and export of visual performance artifacts.
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

import joblib
import pandas as pd
import yaml

# Ensure project root is available in module path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import CarDataLoader
from src.evaluate import (
    compute_regression_metrics,
    display_benchmark_table,
    plot_actual_vs_predicted,
    plot_feature_importance,
)
from src.feature_engineering import build_preprocessor, get_feature_names
from src.models import (
    build_regression_pipelines,
    extract_feature_importances,
    split_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("main")


def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        logger.error(f"Configuration file not found at '{config_path}'.")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration settings from '{config_path}'.")
    return config


def run_pipeline(config_path: str | Path = "config/config.yaml") -> None:
    """
    Execute complete used car price prediction lifecycle.

    Args:
        config_path: File system path to the YAML configuration file.
    """
    start_time = datetime.datetime.now()
    logger.info("=" * 80)
    logger.info("STARTING USED CAR LISTING PRICE PREDICTION PIPELINE")
    logger.info("=" * 80)

    # 1. Load Configurations
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})
    feat_cfg = cfg.get("features", {})
    split_cfg = cfg.get("split", {})
    models_cfg = cfg.get("models", {})
    out_cfg = cfg.get("outputs", {})

    output_dir = Path(out_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_file = output_dir / out_cfg.get("benchmark_summary", "benchmark_summary.csv")
    importance_file = output_dir / out_cfg.get("feature_importance_plot", "feature_importance.png")
    actual_pred_file = output_dir / out_cfg.get("actual_vs_predicted_plot", "actual_vs_predicted.png")

    source_path = Path(data_cfg.get("source_path", "C:\\Users\\Mahmm\\used_cars.csv"))
    reference_year = data_cfg.get("reference_year", 2026)
    max_price = data_cfg.get("max_price_threshold", 250000.0)

    num_cols = feat_cfg.get(
        "numeric_features",
        ["milage", "model_year", "vehicle_age", "hp", "engine_liter"],
    )
    cat_cols = feat_cfg.get(
        "categorical_features",
        ["brand", "fuel_type", "transmission", "accident", "clean_title"],
    )
    cat_impute_dict = feat_cfg.get("imputation", {}).get("categorical_fill", None)

    # 2. Ingestion, Cleaning & Feature Extraction
    loader = CarDataLoader(
        reference_year=reference_year,
        max_price_threshold=max_price,
        categorical_fill=cat_impute_dict,
    )
    X, y = loader.load_and_preprocess(
        source_path=source_path,
        target_column=data_cfg.get("target_column", "price"),
        numeric_features=num_cols,
        categorical_features=cat_cols,
    )

    # 3. Train / Test Split
    test_size = split_cfg.get("test_size", 0.20)
    random_state = split_cfg.get("random_state", 42)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    # 4. Pipeline Architecture Construction
    preprocessor = build_preprocessor(
        numeric_features=num_cols,
        categorical_features=cat_cols,
        sparse_output=False,
    )
    candidate_pipelines = build_regression_pipelines(
        preprocessor=preprocessor,
        hyperparameters=models_cfg,
    )

    # 5. Model Training & Benchmark Evaluation
    logger.info("Fitting and evaluating candidate regression algorithms...")
    metrics_list: list[dict[str, Any]] = []
    fitted_pipelines: dict[str, Any] = {}
    test_predictions: dict[str, Any] = {}

    for model_name, pipeline in candidate_pipelines.items():
        logger.info(f"Training pipeline: '{model_name}' on {len(X_train)} samples...")
        pipeline.fit(X_train, y_train)
        fitted_pipelines[model_name] = pipeline

        y_pred = pipeline.predict(X_test)
        test_predictions[model_name] = y_pred

        metrics = compute_regression_metrics(y_test, y_pred, model_name=model_name)
        metrics_list.append(metrics)
        logger.info(
            f"Finished '{model_name}': R2={metrics['R2 Score']:.4f}, "
            f"MAE=${metrics['MAE ($)']:,.2f}, RMSE=${metrics['RMSE ($)']:,.2f}"
        )

    # 6. Display and Export Benchmark Summary
    summary_df = display_benchmark_table(metrics_list, output_path=benchmark_file)

    winning_model_name = summary_df.iloc[0]["Model"]
    winning_r2 = summary_df.iloc[0]["R2 Score"]
    winning_mae = summary_df.iloc[0]["MAE ($)"]
    logger.info(
        f"Top Performing Model Identified: '{winning_model_name}' "
        f"(R2 Score: {winning_r2:.4f}, MAE: ${winning_mae:,.2f})"
    )

    # 7. Feature Importance Visualization
    logger.info("Generating feature importance driver visualization...")
    # Prefer Random Forest or tree-based ensemble for feature importance attribution
    rf_pipeline = fitted_pipelines.get("Random Forest Regressor") or fitted_pipelines[winning_model_name]
    fitted_prep = rf_pipeline.named_steps["preprocessor"]
    transformed_feature_names = get_feature_names(fitted_prep, num_cols, cat_cols)
    importances = extract_feature_importances(rf_pipeline, transformed_feature_names)

    plot_feature_importance(
        importance_series=importances,
        top_n=15,
        output_path=importance_file,
        model_name="Random Forest Regressor",
    )

    # 8. Actual vs. Predicted Price Visualization
    logger.info(f"Plotting actual vs. predicted prices for winning model '{winning_model_name}'...")
    plot_actual_vs_predicted(
        y_true=y_test,
        y_pred=test_predictions[winning_model_name],
        model_name=winning_model_name,
        r2=float(winning_r2),
        mae=float(winning_mae),
        output_path=actual_pred_file,
    )

    # 9. Persist Winning Model
    model_save_path = output_dir / "best_car_price_model.joblib"
    joblib.dump(
        {
            "model_name": winning_model_name,
            "pipeline": fitted_pipelines[winning_model_name],
            "r2_score": float(winning_r2),
            "mae": float(winning_mae),
            "features": {"numeric": num_cols, "categorical": cat_cols},
            "timestamp": datetime.datetime.now().isoformat(),
        },
        model_save_path,
    )
    logger.info(f"Saved winning model bundle to '{model_save_path}'.")

    elapsed = datetime.datetime.now() - start_time
    logger.info("=" * 80)
    logger.info(f"CAR PRICE REGRESSION PIPELINE COMPLETED IN {elapsed.total_seconds():.2f}s")
    logger.info(f"Model Comparison Table: '{benchmark_file.resolve()}'")
    logger.info(f"Feature Importance:     '{importance_file.resolve()}'")
    logger.info(f"Actual vs Predicted:    '{actual_pred_file.resolve()}'")
    logger.info("=" * 80)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for main orchestrator."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end Used Car Price Prediction machine learning pipeline."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_pipeline(config_path=cli_args.config)
