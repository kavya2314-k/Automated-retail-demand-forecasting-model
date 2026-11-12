 
from pathlib import Path

import pandas as pd

from src.data_loader import load_m5_data
from src.preprocessing import prepare_m5_data

from src.visualization import (
    plot_actual_vs_forecast,
    plot_forecast_comparison,
    plot_model_performance,
    plot_forecast_kpis,
    save_visualizations
)


# ==========================================================
# 1. PATH CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "visualizations"

SALES_PATH = DATA_DIR / "sales_train_evaluation.csv"
CALENDAR_PATH = DATA_DIR / "calendar.csv"

FORECAST_PATH = OUTPUT_DIR / "automated_28_day_forecast.csv"
SUMMARY_PATH = OUTPUT_DIR / "automated_forecast_summary.csv"
MODEL_COMPARISON_PATH = (
    OUTPUT_DIR / "rolling_backtest_model_comparison.csv"
)


# ==========================================================
# 2. LOAD HISTORICAL DATA
# ==========================================================

print("\nLoading historical data...")

sales, calendar, day_cols = load_m5_data(
    CALENDAR_PATH,
    SALES_PATH
)

data = prepare_m5_data(
    sales=sales,
    calendar=calendar,
    day_cols=day_cols,
    n_stores=2,
    horizon=28
)

historical_data = data["Y"]


# ==========================================================
# 3. LOAD FORECAST AND PERFORMANCE DATA
# ==========================================================

print("\nLoading forecast results...")

forecast_df = pd.read_csv(FORECAST_PATH)
summary_df = pd.read_csv(SUMMARY_PATH)
model_comparison_df = pd.read_csv(
    MODEL_COMPARISON_PATH
)

forecast_df["forecast_date"] = pd.to_datetime(
    forecast_df["forecast_date"]
)


# ==========================================================
# 4. CREATE VISUALIZATIONS
# ==========================================================

print("\nCreating visualizations...")

figures = {}

# Chart 1: Historical vs forecast for each series.
for series_id in data["series_ids"]:

    figures[f"forecast_{series_id}"] = (
        plot_actual_vs_forecast(
            historical_data=historical_data,
            forecast_df=forecast_df,
            series_id=series_id,
            history_days=180
        )
    )

# Chart 2: Forecast comparison.
figures["forecast_comparison"] = (
    plot_forecast_comparison(summary_df)
)

# Chart 3: Model performance.
figures["model_performance"] = (
    plot_model_performance(model_comparison_df)
)

# Chart 4: Forecast KPI analysis.
figures["forecast_kpis"] = (
    plot_forecast_kpis(summary_df)
)


# ==========================================================
# 5. SAVE VISUALIZATIONS
# ==========================================================

print("\nSaving interactive HTML charts...")

saved_files = save_visualizations(
    figures=figures,
    output_dir=CHART_DIR
)

print("\n" + "=" * 55)
print("VISUALIZATION GENERATION COMPLETED")
print("=" * 55)

print("Total charts generated:", len(saved_files))
print("Charts directory:", CHART_DIR)
