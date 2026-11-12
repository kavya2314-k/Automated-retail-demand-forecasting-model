 
from pathlib import Path
import subprocess
import sys

import pandas as pd
import streamlit as st

from src.data_loader import load_m5_data
from src.preprocessing import prepare_m5_data

from src.visualization import (
    plot_actual_vs_forecast,
    plot_forecast_comparison,
    plot_model_performance,
    plot_forecast_kpis
)


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SALES_PATH = DATA_DIR / "sales_train_evaluation.csv"
CALENDAR_PATH = DATA_DIR / "calendar.csv"

FORECAST_PATH = OUTPUT_DIR / "automated_28_day_forecast.csv"
SUMMARY_PATH = OUTPUT_DIR / "automated_forecast_summary.csv"

MODEL_PATH = OUTPUT_DIR / "rolling_backtest_model_comparison.csv"

HORIZON = 28
N_STORES = 2


# ==========================================================
# 2. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="M5 Retail Demand Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 3. LOAD DATA
# ==========================================================

@st.cache_data
def load_historical_data():

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

    return data["Y"]


@st.cache_data
def load_forecast_results():

    forecast = pd.read_csv(FORECAST_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    comparison = pd.read_csv(MODEL_PATH)

    forecast["forecast_date"] = pd.to_datetime(
        forecast["forecast_date"]
    )

    return forecast, summary, comparison


# ==========================================================
# 4. HEADER
# ==========================================================

st.title("M5 Retail Demand Forecasting System")

st.markdown(
    """
    An automated retail demand forecasting platform integrating
    statistical and machine-learning models, rolling-origin
    validation and automated model selection.
    """
)

st.divider()


# ==========================================================
# 5. CHECK OUTPUT FILES
# ==========================================================

required_files = [
    FORECAST_PATH,
    SUMMARY_PATH,
    MODEL_PATH
]

if not all(path.exists() for path in required_files):

    st.warning(
        "Forecast output files are missing. "
        "Run the forecasting pipeline first."
    )

    st.stop()


# ==========================================================
# 6. SIDEBAR
# ==========================================================

st.sidebar.title("Forecast Controls")

st.sidebar.markdown(
    "Configure and inspect the forecasting system."
)

if st.sidebar.button(
    "Run New Forecast",
    use_container_width=True
):

    with st.spinner("Executing automated forecasting pipeline..."):

        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "run_automated_forecast.py")
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )

    if result.returncode == 0:

        st.sidebar.success("Forecast generated successfully.")

        st.cache_data.clear()
        st.rerun()

    else:

        st.sidebar.error("Forecast execution failed.")

        st.error(result.stderr[-5000:])

st.sidebar.divider()

st.sidebar.caption(
    "Forecast horizon: 28 days"
)

st.sidebar.caption(
    "Selected stores: CA_1 and CA_2"
)


# ==========================================================
# 7. LOAD RESULTS
# ==========================================================

try:

    historical_data = load_historical_data()

    forecast_df, summary_df, model_comparison_df = (
        load_forecast_results()
    )

except Exception as exc:

    st.error(f"Unable to load project data: {exc}")
    st.stop()


if forecast_df.empty or summary_df.empty:

    st.error("Forecast results are empty.")
    st.stop()


# ==========================================================
# 8. OVERVIEW KPIs
# ==========================================================

total_demand = summary_df[
    "total_forecast_demand"
].sum()

average_daily = summary_df[
    "average_daily_demand"
].sum()

number_of_series = summary_df[
    "series_id"
].nunique()

best_model = (
    model_comparison_df
    .sort_values("WAPE")
    .iloc[0]["Model"]
)

st.subheader("Forecast Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total 28-Day Demand",
    f"{total_demand:,.0f}"
)

col2.metric(
    "Average Daily Demand",
    f"{average_daily:,.0f}"
)

col3.metric(
    "Forecasting Series",
    number_of_series
)

col4.metric(
    "Best Overall Model",
    best_model
)

st.caption(
    "The best overall model is determined using pooled "
    "rolling-validation WAPE."
)

st.divider()


# ==========================================================
# 9. DEMAND FORECASTING
# ==========================================================

st.subheader("Demand Forecasting")

selected_series = st.selectbox(
    "Select Store-Category Series",
    options=summary_df["series_id"].tolist()
)

forecast_chart = plot_actual_vs_forecast(
    historical_data=historical_data,
    forecast_df=forecast_df,
    series_id=selected_series,
    history_days=180
)

st.plotly_chart(
    forecast_chart,
    use_container_width=True
)

selected_summary = summary_df[
    summary_df["series_id"] == selected_series
]

st.markdown("**Selected Series Forecast Summary**")

st.dataframe(
    selected_summary,
    use_container_width=True,
    hide_index=True
)

st.divider()


# ==========================================================
# 10. FORECAST COMPARISON
# ==========================================================

st.subheader("Store and Category Comparison")

comparison_chart = plot_forecast_comparison(
    summary_df
)

st.plotly_chart(
    comparison_chart,
    use_container_width=True
)

st.divider()


# ==========================================================
# 11. MODEL PERFORMANCE
# ==========================================================

st.subheader("Forecasting Model Performance")

st.markdown(
    "Models are compared using rolling-origin validation. "
    "Lower WAPE indicates better pooled forecast accuracy."
)

performance_chart = plot_model_performance(
    model_comparison_df
)

st.plotly_chart(
    performance_chart,
    use_container_width=True
)

with st.expander("View detailed model performance"):

    st.dataframe(
        model_comparison_df.sort_values("WAPE"),
        use_container_width=True,
        hide_index=True
    )

st.divider()


# ==========================================================
# 12. FORECAST KPI ANALYSIS
# ==========================================================

st.subheader("Forecast KPI Analysis")

kpi_chart = plot_forecast_kpis(
    summary_df
)

st.plotly_chart(
    kpi_chart,
    use_container_width=True
)

st.divider()


# ==========================================================
# 13. FORECAST DATA TABLE
# ==========================================================

st.subheader("Detailed Forecast Results")

selected_store = st.selectbox(
    "Filter by Series",
    options=["All Series"] + summary_df["series_id"].tolist()
)

if selected_store == "All Series":

    displayed_forecast = forecast_df.copy()

else:

    displayed_forecast = forecast_df[
        forecast_df["series_id"] == selected_store
    ].copy()

st.dataframe(
    displayed_forecast,
    use_container_width=True,
    hide_index=True
)


# ==========================================================
# 14. DOWNLOAD FORECAST
# ==========================================================

forecast_csv = displayed_forecast.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Forecast CSV",
    data=forecast_csv,
    file_name="m5_demand_forecast.csv",
    mime="text/csv"
)

summary_csv = summary_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Forecast Summary",
    data=summary_csv,
    file_name="m5_forecast_summary.csv",
    mime="text/csv"
)


# ==========================================================
# 15. FOOTER
# ==========================================================

st.divider()

st.caption(
    "M5 Automated Retail Demand Forecasting System | "
    "Statistical Forecasting | Machine Learning | "
    "Rolling-Origin Validation"
)
