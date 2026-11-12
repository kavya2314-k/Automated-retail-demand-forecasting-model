 
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "rolling_backtest_predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. LOAD BACKTESTING PREDICTIONS
# ============================================================

print("\n" + "=" * 65)
print("STEP 1: LOADING BACKTESTING PREDICTIONS")
print("=" * 65)

df = pd.read_csv(INPUT_PATH)

required_columns = [
    "fold",
    "forecast_origin",
    "series_id",
    "horizon",
    "model",
    "actual",
    "forecast"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

if df[["actual", "forecast"]].isna().any().any():
    raise ValueError(
        "Actual or forecast columns contain missing values."
    )

if not np.isfinite(
    df[["actual", "forecast"]].to_numpy(dtype=float)
).all():
    raise ValueError(
        "Actual or forecast contains infinite values."
    )

print("Data loaded successfully.")
print("Rows:", len(df))
print("Models:", df["model"].nunique())
print("Series:", df["series_id"].nunique())
print("Folds:", df["fold"].nunique())


# ============================================================
# 3. METRICS FUNCTION
# ============================================================

def calculate_metrics(group):

    actual = group["actual"].to_numpy(dtype=float)
    forecast = group["forecast"].to_numpy(dtype=float)

    error = forecast - actual

    denominator = np.sum(np.abs(actual))

    mae = np.mean(np.abs(error))

    rmse = np.sqrt(np.mean(error ** 2))

    wape = (
        np.sum(np.abs(error)) / denominator
        if denominator != 0
        else np.nan
    )

    bias = (
        np.sum(error) / denominator * 100
        if denominator != 0
        else np.nan
    )

    return pd.Series({

        "MAE": mae,
        "RMSE": rmse,
        "WAPE (%)": wape * 100,
        "Bias (%)": bias,
        "Actual Demand": denominator,
        "Forecast Demand": np.sum(forecast)

    })


# ============================================================
# 4. MODEL PERFORMANCE FOR EACH SERIES
# ============================================================

print("\n" + "=" * 65)
print("STEP 2: SERIES-WISE MODEL PERFORMANCE")
print("=" * 65)

series_performance = (
    df.groupby(["series_id", "model"])
    .apply(calculate_metrics, include_groups=False)
    .reset_index()
)

series_performance = series_performance.sort_values(
    ["series_id", "WAPE (%)"]
)

print("\nSERIES-WISE PERFORMANCE:")

print(
    series_performance.to_string(index=False)
)

series_performance.to_csv(
    OUTPUT_DIR / "series_model_performance.csv",
    index=False
)


# ============================================================
# 5. IDENTIFY BEST MODEL FOR EACH SERIES
# ============================================================

print("\n" + "=" * 65)
print("STEP 3: BEST MODEL FOR EACH SERIES")
print("=" * 65)

valid_results = series_performance.dropna(
    subset=["WAPE (%)"]
)

best_models = (
    valid_results
    .loc[
        valid_results.groupby("series_id")["WAPE (%)"].idxmin()
    ]
    .sort_values("series_id")
    .reset_index(drop=True)
)

print("\nBEST MODEL PER SERIES:")

print(
    best_models[
        [
            "series_id",
            "model",
            "MAE",
            "RMSE",
            "WAPE (%)",
            "Bias (%)"
        ]
    ].to_string(index=False)
)

best_models.to_csv(
    OUTPUT_DIR / "best_model_per_series.csv",
    index=False
)


# ============================================================
# 6. FIND MOST DIFFICULT SERIES TO FORECAST
# ============================================================

print("\n" + "=" * 65)
print("STEP 4: SERIES FORECASTING DIFFICULTY")
print("=" * 65)

series_difficulty = (
    df.groupby("series_id")
    .apply(calculate_metrics, include_groups=False)
    .reset_index()
    .sort_values("WAPE (%)", ascending=False)
)

print("\nSERIES RANKED BY FORECASTING ERROR:")

print(
    series_difficulty.to_string(index=False)
)

series_difficulty.to_csv(
    OUTPUT_DIR / "series_forecasting_difficulty.csv",
    index=False
)


# ============================================================
# 7. FOLD-WISE PERFORMANCE FOR EACH SERIES
# ============================================================

print("\n" + "=" * 65)
print("STEP 5: FOLD-WISE SERIES PERFORMANCE")
print("=" * 65)

fold_series_performance = (
    df.groupby(["fold", "series_id", "model"])
    .apply(calculate_metrics, include_groups=False)
    .reset_index()
)

fold_series_performance.to_csv(
    OUTPUT_DIR / "fold_series_model_performance.csv",
    index=False
)

print("Fold-wise series performance saved.")


# ============================================================
# 8. LIGHTGBM VS HOLT-WINTERS
# ============================================================

print("\n" + "=" * 65)
print("STEP 6: LIGHTGBM VS HOLT-WINTERS")
print("=" * 65)

comparison = series_performance[
    series_performance["model"].isin(
        ["LightGBM", "HoltWinters"]
    )
].pivot(
    index="series_id",
    columns="model",
    values="WAPE (%)"
)

comparison["Better Model"] = np.where(
    comparison["LightGBM"] < comparison["HoltWinters"],
    "LightGBM",
    "HoltWinters"
)

comparison["WAPE Difference (percentage points)"] = (
    comparison["HoltWinters"] - comparison["LightGBM"]
)

print("\nMODEL COMPARISON:")

print(comparison.to_string())

comparison.to_csv(
    OUTPUT_DIR / "lightgbm_vs_holtwinters.csv"
)


# ============================================================
# 9. FINAL DIAGNOSTIC SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("STEP 7: FINAL DIAGNOSTIC SUMMARY")
print("=" * 65)

print("\nBest model for each series:")

for _, row in best_models.iterrows():

    print(
        f"{row['series_id']}: "
        f"{row['model']} "
        f"(WAPE: {row['WAPE (%)']:.2f}%)"
    )

print("\nMost difficult series:")

hardest = series_difficulty.iloc[0]

print(
    hardest["series_id"],
    "- WAPE:",
    round(hardest["WAPE (%)"], 2),
    "%"
)

print("\nNumber of series won by each model:")

print(
    best_models["model"].value_counts().to_string()
)

print("\nDiagnostic files saved:")

print("1. series_model_performance.csv")
print("2. best_model_per_series.csv")
print("3. series_forecasting_difficulty.csv")
print("4. fold_series_model_performance.csv")
print("5. lightgbm_vs_holtwinters.csv")

print("\nALL SERIES DIAGNOSTICS COMPLETED.")
