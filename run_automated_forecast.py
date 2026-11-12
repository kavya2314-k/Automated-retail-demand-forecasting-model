 
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.data_loader import load_m5_data
from src.preprocessing import prepare_m5_data
from src.forecast_models import ForecastingModels


# ==========================================================
# 1. PROJECT CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SALES_PATH = DATA_DIR / "sales_train_evaluation.csv"
CALENDAR_PATH = DATA_DIR / "calendar.csv"

MODEL_CONFIG_PATH = OUTPUT_DIR / "selected_models.json"

HORIZON = 28
N_STORES = 2


# ==========================================================
# 2. LOAD SELECTED MODELS
# ==========================================================

def load_model_configuration():

    if not MODEL_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Model configuration not found: {MODEL_CONFIG_PATH}"
        )

    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    selected_models = config.get("selected_models")

    if not selected_models:
        raise ValueError(
            "No selected models found in configuration."
        )

    print("\nSelected models loaded:")
    for series_id, model_name in selected_models.items():
        print(f"{series_id}: {model_name}")

    return selected_models


# ==========================================================
# 3. LOAD AND PREPROCESS DATA
# ==========================================================

def load_forecasting_data():

    print("\nLoading M5 dataset...")

    sales, calendar, day_cols = load_m5_data(
        CALENDAR_PATH,
        SALES_PATH
    )

    data = prepare_m5_data(
        sales=sales,
        calendar=calendar,
        day_cols=day_cols,
        n_stores=N_STORES,
        horizon=HORIZON
    )

    print("Data loaded successfully.")
    print("Historical days:", data["T"])
    print("Series:", data["S"])
    print("Forecast horizon:", HORIZON)

    return data


# ==========================================================
# 4. GENERATE AUTOMATED FORECASTS
# ==========================================================

def generate_forecasts(data, selected_models):

    series_ids = data["series_ids"]
    forecast_origin = data["T"] - 1

    # Confirm that every prepared series has a model.
    missing_series = set(series_ids) - set(selected_models)
    extra_series = set(selected_models) - set(series_ids)

    if missing_series or extra_series:
        raise ValueError(
            f"Model configuration mismatch.\n"
            f"Missing series: {sorted(missing_series)}\n"
            f"Unknown series: {sorted(extra_series)}"
        )

    engine = ForecastingModels(
        data=data,
        horizon=HORIZON,
        stat_history=730,
        gbm_origin_step=7,
        first_gbm_origin=60
    )

    # The latest historical day is the forecast origin.
    print(
        "\nForecast origin:",
        data["hist_dates"][forecast_origin].date()
    )

    # Run each distinct model only once.
    unique_models = sorted(set(selected_models.values()))

    forecasts_by_model = {}

    for model_name in unique_models:

        print(f"\nRunning model: {model_name}")

        predictions = engine.predict_model(
            model_name=model_name,
            forecast_origin=forecast_origin
        )

        forecasts_by_model[model_name] = predictions

        print(
            f"{model_name} completed. "
            f"Output shape: {predictions.shape}"
        )

    # Select the appropriate model forecast for each series.
    final_forecasts = np.zeros(
        (len(series_ids), HORIZON),
        dtype=float
    )

    for series_position, series_id in enumerate(series_ids):

        selected_model = selected_models[series_id]

        model_predictions = forecasts_by_model[selected_model]

        final_forecasts[series_position] = (
            model_predictions[series_position]
        )

    if not np.isfinite(final_forecasts).all():
        raise ValueError("Final forecast contains NaN or infinity.")

    if (final_forecasts < 0).any():
        raise ValueError("Final forecast contains negative values.")

    return final_forecasts, forecast_origin


# ==========================================================
# 5. FORMAT FORECAST OUTPUT
# ==========================================================

def create_forecast_dataframe(
    data,
    forecasts,
    forecast_origin
):

    future_dates = data["cal_dates"][
        forecast_origin + 1:
        forecast_origin + 1 + HORIZON
    ]

    if len(future_dates) != HORIZON:
        raise ValueError(
            "Incorrect number of future forecast dates."
        )

    records = []

    for series_position, series_id in enumerate(data["series_ids"]):

        for day in range(HORIZON):

            records.append({
                "series_id": series_id,
                "forecast_day": day + 1,
                "forecast_date": future_dates[day],
                "forecast_demand": forecasts[
                    series_position, day
                ],
                "selected_model": selected_models[series_id]
            })

    return pd.DataFrame(records)


# ==========================================================
# 6. CREATE SERIES-LEVEL SUMMARY
# ==========================================================

def create_forecast_summary(forecast_df):

    summary = (
        forecast_df
        .groupby(
            ["series_id", "selected_model"],
            as_index=False
        )
        .agg(
            total_forecast_demand=(
                "forecast_demand", "sum"
            ),
            average_daily_demand=(
                "forecast_demand", "mean"
            ),
            minimum_daily_demand=(
                "forecast_demand", "min"
            ),
            maximum_daily_demand=(
                "forecast_demand", "max"
            )
        )
    )

    return summary


# ==========================================================
# 7. SAVE FORECAST RESULTS
# ==========================================================

def save_forecast_results(forecast_df, summary_df):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    forecast_path = (
        OUTPUT_DIR / "automated_28_day_forecast.csv"
    )

    summary_path = (
        OUTPUT_DIR / "automated_forecast_summary.csv"
    )

    forecast_df.to_csv(
        forecast_path,
        index=False
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\nForecast files saved:")
    print(forecast_path)
    print(summary_path)


# ==========================================================
# 8. MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("M5 AUTOMATED DEMAND FORECASTING SYSTEM")
    print("=" * 60)

    selected_models = load_model_configuration()

    data = load_forecasting_data()

    # Make the configuration available to output formatting.
    globals()["selected_models"] = selected_models

    forecasts, forecast_origin = generate_forecasts(
        data,
        selected_models
    )

    forecast_df = create_forecast_dataframe(
        data,
        forecasts,
        forecast_origin
    )

    summary_df = create_forecast_summary(forecast_df)

    save_forecast_results(
        forecast_df,
        summary_df
    )

    print("\nFORECAST SUMMARY")
    print(summary_df.to_string(index=False))

    print("\nTotal forecast records:", len(forecast_df))

    print("\nAUTOMATED FORECASTING COMPLETED SUCCESSFULLY.")
