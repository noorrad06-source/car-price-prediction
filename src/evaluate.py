"""
Regression Evaluation and Visualization Module.

Computes regression performance metrics (MAE, RMSE, R2 Score), prints formatted
terminal benchmark tables, and exports publication-ready visual artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
# Use non-interactive backend for headless and production scripts
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger("evaluate")


def compute_regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str = "Model",
) -> dict[str, Any]:
    """
    Compute standard regression evaluation metrics.

    Args:
        y_true: Actual ground truth car prices.
        y_pred: Predicted car prices.
        model_name: Identifier for evaluated model.

    Returns:
        Dictionary of calculated metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Calculate Mean Absolute Percentage Error (MAPE) safely
    non_zero = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

    return {
        "Model": model_name,
        "MAE ($)": mae,
        "RMSE ($)": rmse,
        "R2 Score": r2,
        "MAPE (%)": mape,
    }


def display_benchmark_table(
    metrics_list: list[dict[str, Any]],
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Format and display an ASCII comparison table of regression metrics,
    and persist results to CSV.

    Args:
        metrics_list: List of metric dictionaries.
        output_path: Destination path for CSV summary.

    Returns:
        Sorted summary DataFrame.
    """
    summary_df = pd.DataFrame(metrics_list)
    summary_df = summary_df.sort_values(by="R2 Score", ascending=False)

    display_cols = ["Model", "MAE ($)", "RMSE ($)", "R2 Score", "MAPE (%)"]
    formatted_df = summary_df.copy()
    formatted_df["MAE ($)"] = formatted_df["MAE ($)"].map(lambda x: f"${x:,.2f}")
    formatted_df["RMSE ($)"] = formatted_df["RMSE ($)"].map(lambda x: f"${x:,.2f}")
    formatted_df["R2 Score"] = formatted_df["R2 Score"].map(lambda x: f"{x:.4f}")
    formatted_df["MAPE (%)"] = formatted_df["MAPE (%)"].map(lambda x: f"{x:.2f}%")

    col_widths = {
        col: max(len(col), max(len(str(val)) for val in formatted_df[col])) + 2
        for col in display_cols
    }
    sep = "+" + "+".join("-" * col_widths[col] for col in display_cols) + "+"
    hdr = "|" + "|".join(f" {col.center(col_widths[col] - 2)} " for col in display_cols) + "|"

    rows = []
    for _, row in formatted_df.iterrows():
        r = "|" + "|".join(f" {str(row[col]).center(col_widths[col] - 2)} " for col in display_cols) + "|"
        rows.append(r)

    ascii_table = "\n" + "\n".join([sep, hdr, sep] + rows + [sep])

    print("\n" + "=" * 80)
    print(" CAR PRICE PREDICTION BENCHMARK RESULTS ".center(80, "="))
    print("=" * 80)
    print(ascii_table)
    print("=" * 80 + "\n")

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_p, index=False)
        logger.info(f"Saved benchmark summary table to '{out_p}'.")

    return summary_df


def plot_feature_importance(
    importance_series: pd.Series,
    top_n: int = 15,
    output_path: str | Path = "outputs/feature_importance.png",
    model_name: str = "Random Forest Regressor",
) -> None:
    """
    Render and save a horizontal bar chart displaying top price-driving features.

    Args:
        importance_series: Series of feature importances sorted descending.
        top_n: Number of top features to display.
        output_path: Destination image path.
        model_name: Model name for plot title.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if importance_series.empty:
        logger.warning("Feature importance series is empty. Skipping plot generation.")
        return

    top_features = importance_series.head(top_n).sort_values(ascending=True)

    plt.figure(figsize=(10, 7))
    palette = sns.color_palette("mako", len(top_features))

    bars = plt.barh(
        y=top_features.index,
        width=top_features.values,
        color=palette,
        edgecolor="black",
        linewidth=0.5,
    )

    plt.title(f"Top {top_n} Price-Driving Features ({model_name})", fontsize=14, pad=15)
    plt.xlabel("Relative Importance Score", fontsize=12)
    plt.ylabel("Vehicle Feature", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    # Value labels on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Feature importance visual saved to '{output_path}'.")


def plot_actual_vs_predicted(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str,
    r2: float,
    mae: float,
    output_path: str | Path = "outputs/actual_vs_predicted.png",
) -> None:
    """
    Render and save an actual vs. predicted price scatter plot with an ideal reference line.

    Args:
        y_true: True car prices.
        y_pred: Predicted car prices.
        model_name: Top-performing model name.
        r2: R-squared score.
        mae: Mean Absolute Error in dollars.
        output_path: Destination image path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        x=y_true_arr,
        y=y_pred_arr,
        alpha=0.45,
        color="#2980b9",
        edgecolor="none",
        s=40,
    )

    # 45-degree reference line
    min_val = min(y_true_arr.min(), y_pred_arr.min())
    max_val = max(y_true_arr.max(), y_pred_arr.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label="Perfect Fit (y = x)",
    )

    plt.title(f"Actual vs. Predicted Used Car Prices: {model_name}", fontsize=14, pad=15)
    plt.xlabel("Actual Listing Price ($ USD)", fontsize=12)
    plt.ylabel("Predicted Price ($ USD)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)

    # Annotate metrics
    annotation_text = f"$R^2 = {r2:.4f}$\n$MAE = \\${mae:,.0f}$"
    plt.text(
        0.05,
        0.90,
        annotation_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#bdc3c7"),
    )

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Actual vs. predicted plot saved to '{output_path}'.")
