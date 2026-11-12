
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. PROJECT IMPORTS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_m5_data
from src.preprocessing import prepare_m5_data
from src.forecast_models import ForecastingModels


# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

CALENDAR_PATH = DATA_DIR / "calendar.csv"
SALES_PATH = DATA_DIR / "sales_train_evaluation.csv"

N_STORES = 2
HORIZON = 28

MODEL_NAMES = [
    "SeasonalNaive",
    "HoltWinters",
    "ARIMA",
    "SARIMA",
    "Prophet",
    "RandomForest",
    "LightGBM"
]


# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

print("\n" + "=" * 60)
print("STEP 1: LOADING M5 DATA")
print("=" * 60)

sales, calendar, day_cols = load_m5_data(
    str(CALENDAR_PATH),
    str(SALES_PATH)
)

print("Sales data loaded:", sales.shape)
print("Calendar data loaded:", calendar.shape)
print("Historical day columns:", len(day_cols))


# --------------------------------------------------
# 4. PREPROCESS DATA
# --------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2: PREPROCESSING DATA")
print("=" * 60)

data = prepare_m5_data(
    sales=sales,
    calendar=calendar,
    day_cols=day_cols,
    n_stores=N_STORES,
    horizon=HORIZON
)

print("Number of series:", data["S"])
print("Historical observations:", data["T"])
print("Forecast horizon:", data["HORIZON"])
print("Series IDs:", data["series_ids"])


# --------------------------------------------------
# 5. INITIALIZE FORECASTING ENGINE
# --------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3: INITIALIZING FORECASTING ENGINE")
print("=" * 60)

engine = ForecastingModels(
    data=data,
    horizon=HORIZON,
    stat_history=730,
    gbm_origin_step=7,
    first_gbm_origin=60
)

print("Forecasting engine initialized successfully.")


# --------------------------------------------------
# 6. SELECT CHRONOLOGICAL HOLDOUT ORIGIN
# --------------------------------------------------

T = data["T"]

# Forecast the final 28 historical days.
# The model must only use observations available
# up to the selected forecast origin.

forecast_origin = T - HORIZON - 1

print("\nForecast origin index:", forecast_origin)
print("Forecast horizon:", HORIZON)

actual = data["Yv"][
    forecast_origin + 1:
    forecast_origin + 1 + HORIZON
].T

expected_shape = (data["S"], HORIZON)

print("Expected forecast shape:", expected_shape)
print("Actual holdout shape:", actual.shape)

assert actual.shape == expected_shape, (
    f"Actual data shape mismatch: {actual.shape}"
)


# --------------------------------------------------
# 7. EXECUTE ALL FORECASTING MODELS
# --------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4: EXECUTING FORECASTING MODELS")
print("=" * 60)

results = []
forecasts = {}

for model_name in MODEL_NAMES:

    print("\nRunning model:", model_name)

    start_time = time.time()

    try:

        predictions = engine.predict_model(
            model_name,
            forecast_origin
        )

        predictions = np.asarray(
            predictions,
            dtype=float
        )

        # Check dimensions
        assert predictions.shape == expected_shape, (
            f"Expected {expected_shape}, "
            f"received {predictions.shape}"
        )

        # Check NaN and infinite values
        assert np.isfinite(predictions).all(), (
            "Forecast contains NaN or infinite values."
        )

        # Check negative values
        negative_count = int(
            np.sum(predictions < 0)
        )

        # Runtime
        runtime = time.time() - start_time

        # Calculate basic validation metrics
        mae = np.mean(
            np.abs(actual - predictions)
        )

        rmse = np.sqrt(
            np.mean((actual - predictions) ** 2)
        )

        total_actual = np.sum(actual)

        wape = (
            np.sum(np.abs(actual - predictions))
            / total_actual
            if total_actual != 0
            else np.nan
        )

        # Store forecasts
        forecasts[model_name] = predictions

        results.append({

            "Model": model_name,
            "Status": "PASSED",
            "Shape": str(predictions.shape),
            "MAE": mae,
            "RMSE": rmse,
            "WAPE": wape,
            "Negative Predictions": negative_count,
            "Runtime (seconds)": runtime

        })

        print("Status: PASSED")
        print("Forecast shape:", predictions.shape)
        print("MAE:", round(mae, 4))
        print("RMSE:", round(rmse, 4))
        print("WAPE:", round(wape, 4))
        print("Negative predictions:", negative_count)
        print("Runtime:", round(runtime, 2), "seconds")

    except Exception as error:

        runtime = time.time() - start_time

        results.append({

            "Model": model_name,
            "Status": "FAILED",
            "Shape": "N/A",
            "MAE": np.nan,
            "RMSE": np.nan,
            "WAPE": np.nan,
            "Negative Predictions": np.nan,
            "Runtime (seconds)": runtime

        })

        print("Status: FAILED")
        print("Error:", str(error))

        traceback.print_exc()


# --------------------------------------------------
# 8. DISPLAY FINAL TEST REPORT
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL FORECAST EXECUTION REPORT")
print("=" * 60)

results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False
    )
)

passed = (
    results_df["Status"] == "PASSED"
).sum()

failed = (
    results_df["Status"] == "FAILED"
).sum()

print("\nTotal models:", len(MODEL_NAMES))
print("Passed:", passed)
print("Failed:", failed)


# --------------------------------------------------
# 9. SAVE TEST RESULTS
# --------------------------------------------------

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_DIR / "forecast_execution_report.csv",
    index=False
)

print(
    "\nTest report saved to:",
    OUTPUT_DIR / "forecast_execution_report.csv"
)

if failed == 0:

    print("\nALL FORECAST EXECUTION TESTS PASSED.")

else:

    print("\nSOME FORECASTING MODELS FAILED.")
    print("Review the error messages above.")

