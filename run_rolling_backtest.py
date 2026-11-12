 
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 1. PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_m5_data
from src.preprocessing import prepare_m5_data
from src.forecast_models import ForecastingModels


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CALENDAR_PATH = DATA_DIR / "calendar.csv"
SALES_PATH = DATA_DIR / "sales_train_evaluation.csv"

N_STORES = 2
HORIZON = 28
N_FOLDS = 3

MODEL_NAMES = [
    "SeasonalNaive",
    "HoltWinters",
    "ARIMA",
    "SARIMA",
    "Prophet",
    "RandomForest",
    "LightGBM"
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. METRICS FUNCTION
# ============================================================

def calculate_metrics(actual, forecast):

    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    error = forecast - actual

    denominator = np.sum(np.abs(actual))

    wape = (
        np.sum(np.abs(error)) / denominator
        if denominator != 0
        else np.nan
    )

    bias = (
        np.sum(error) / denominator
        if denominator != 0
        else np.nan
    )

    return {
        "MAE": np.mean(np.abs(error)),
        "RMSE": np.sqrt(np.mean(error ** 2)),
        "WAPE": wape,
        "Bias": bias
    }


# ============================================================
# 4. LOAD M5 DATA
# ============================================================

print("\n" + "=" * 65)
print("STEP 1: LOADING M5 DATA")
print("=" * 65)

sales, calendar, day_cols = load_m5_data(
    str(CALENDAR_PATH),
    str(SALES_PATH)
)

data = prepare_m5_data(
    sales=sales,
    calendar=calendar,
    day_cols=day_cols,
    n_stores=N_STORES,
    horizon=HORIZON
)

print("Data loaded successfully.")
print("Number of series:", data["S"])
print("Historical observations:", data["T"])
print("Forecast horizon:", HORIZON)


# ============================================================
# 5. INITIALIZE FORECASTING ENGINE
# ============================================================

print("\n" + "=" * 65)
print("STEP 2: INITIALIZING FORECASTING ENGINE")
print("=" * 65)

engine = ForecastingModels(
    data=data,
    horizon=HORIZON,
    stat_history=730,
    gbm_origin_step=7,
    first_gbm_origin=60
)

print("Forecasting engine initialized.")


# ============================================================
# 6. GENERATE CHRONOLOGICAL FOLD ORIGINS
# ============================================================

T = data["T"]

last_origin = T - 1 - HORIZON

origins = [
    last_origin - HORIZON * j
    for j in range(N_FOLDS)
]

origins.reverse()

print("\nBacktesting forecast origins:")

for fold_number, origin in enumerate(origins, start=1):
    print(
        f"Fold {fold_number}: "
        f"origin index = {origin}"
    )

assert min(origins) >= 0
assert max(origins) + HORIZON < T


# ============================================================
# 7. RUN ROLLING-ORIGIN BACKTESTING
# ============================================================

print("\n" + "=" * 65)
print("STEP 3: RUNNING ROLLING-ORIGIN BACKTESTING")
print("=" * 65)

records = []
runtime = {model: 0.0 for model in MODEL_NAMES}

for fold_number, forecast_origin in enumerate(origins, start=1):

    print("\n" + "-" * 65)
    print(f"RUNNING FOLD {fold_number} OF {N_FOLDS}")
    print("-" * 65)

    # Actual sales for the next 28 days
    actuals = data["Yv"][
        forecast_origin + 1:
        forecast_origin + 1 + HORIZON
    ].T

    expected_shape = (data["S"], HORIZON)

    if actuals.shape != expected_shape:
        raise ValueError(
            f"Fold {fold_number}: "
            f"Expected actual shape {expected_shape}, "
            f"received {actuals.shape}"
        )

    for model_name in MODEL_NAMES:

        print(f"\nRunning model: {model_name}")

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

            # Validate forecast shape
            if predictions.shape != expected_shape:
                raise ValueError(
                    f"Expected shape {expected_shape}, "
                    f"received {predictions.shape}"
                )

            # Validate forecast values
            if not np.isfinite(predictions).all():
                raise ValueError(
                    "Forecast contains NaN or infinite values."
                )

            model_runtime = time.time() - start_time
            runtime[model_name] += model_runtime

            # Save individual series predictions
            for series_position, series_id in enumerate(
                data["series_ids"]
            ):

                for h in range(HORIZON):

                    records.append({

                        "fold": fold_number,

                        "forecast_origin": forecast_origin,

                        "series_id": series_id,

                        "horizon": h + 1,

                        "model": model_name,

                        "actual": actuals[
                            series_position, h
                        ],

                        "forecast": predictions[
                            series_position, h
                        ]

                    })

            print(
                f"{model_name} completed successfully. "
                f"Runtime: {model_runtime:.2f} seconds"
            )

        except Exception as error:

            print(f"\nERROR in {model_name}, Fold {fold_number}")
            traceback.print_exc()

            raise RuntimeError(
                f"Backtesting stopped at {model_name}, "
                f"Fold {fold_number}: {error}"
            ) from error


# ============================================================
# 8. CREATE BACKTESTING DATAFRAME
# ============================================================

print("\n" + "=" * 65)
print("STEP 4: CREATING BACKTESTING RESULTS")
print("=" * 65)

backtest_df = pd.DataFrame(records)

expected_records = (
    N_FOLDS
    * len(MODEL_NAMES)
    * data["S"]
    * HORIZON
)

if len(backtest_df) != expected_records:
    raise ValueError(
        f"Expected {expected_records} records, "
        f"received {len(backtest_df)}"
    )

print("Backtesting completed successfully.")
print("Backtesting dataframe shape:", backtest_df.shape)

backtest_df.to_csv(
    OUTPUT_DIR / "rolling_backtest_predictions.csv",
    index=False
)


# ============================================================
# 9. MODEL PERFORMANCE ACROSS ALL FOLDS
# ============================================================

print("\n" + "=" * 65)
print("STEP 5: CALCULATING MODEL PERFORMANCE")
print("=" * 65)

summary_records = []

for model_name in MODEL_NAMES:

    model_data = backtest_df[
        backtest_df["model"] == model_name
    ]

    metrics = calculate_metrics(
        model_data["actual"],
        model_data["forecast"]
    )

    summary_records.append({

        "Model": model_name,

        "MAE": metrics["MAE"],

        "RMSE": metrics["RMSE"],

        "WAPE": metrics["WAPE"],

        "Bias": metrics["Bias"],

        "Runtime (seconds)": runtime[model_name]

    })


summary_df = pd.DataFrame(summary_records)

summary_df = summary_df.sort_values(
    by="WAPE"
).reset_index(drop=True)

summary_df.to_csv(
    OUTPUT_DIR / "rolling_backtest_model_comparison.csv",
    index=False
)

print("\nOVERALL MODEL PERFORMANCE:")

print(
    summary_df.to_string(index=False)
)


# ============================================================
# 10. PERFORMANCE BY FOLD
# ============================================================

print("\n" + "=" * 65)
print("STEP 6: FOLD-WISE MODEL PERFORMANCE")
print("=" * 65)

fold_records = []

for (fold, model), group in backtest_df.groupby(
    ["fold", "model"]
):

    metrics = calculate_metrics(
        group["actual"],
        group["forecast"]
    )

    fold_records.append({

        "Fold": fold,

        "Model": model,

        "MAE": metrics["MAE"],

        "RMSE": metrics["RMSE"],

        "WAPE": metrics["WAPE"],

        "Bias": metrics["Bias"]

    })


fold_summary_df = pd.DataFrame(fold_records)

fold_summary_df.to_csv(
    OUTPUT_DIR / "rolling_backtest_fold_comparison.csv",
    index=False
)

print(
    fold_summary_df.to_string(index=False)
)


# ============================================================
# 11. FINAL VALIDATION REPORT
# ============================================================

print("\n" + "=" * 65)
print("ROLLING-ORIGIN BACKTESTING COMPLETED")
print("=" * 65)

print("Total folds:", N_FOLDS)
print("Total models:", len(MODEL_NAMES))
print("Total series:", data["S"])
print("Forecast horizon:", HORIZON)
print("Total forecast records:", len(backtest_df))

print("\nCurrent best model based on pooled WAPE:")

print(summary_df.iloc[0]["Model"])

print("\nFiles saved successfully:")

print(
    OUTPUT_DIR / "rolling_backtest_predictions.csv"
)

print(
    OUTPUT_DIR / "rolling_backtest_model_comparison.csv"
)

print(
    OUTPUT_DIR / "rolling_backtest_fold_comparison.csv"
)

print("\nALL ROLLING-ORIGIN BACKTESTS COMPLETED.")
