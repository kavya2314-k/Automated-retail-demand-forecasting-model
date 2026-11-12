 
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ==========================================================
# 1. HISTORICAL DEMAND VS FUTURE FORECAST
# ==========================================================

def plot_actual_vs_forecast(
    historical_data,
    forecast_df,
    series_id,
    history_days=180
):
    """
    Plot historical demand and future forecast for one series.
    """

    if series_id not in historical_data.columns:
        raise ValueError(f"Unknown series: {series_id}")

    series_forecast = forecast_df[
        forecast_df["series_id"] == series_id
    ].copy()

    if series_forecast.empty:
        raise ValueError(
            f"No forecast records found for {series_id}"
        )

    series_forecast["forecast_date"] = pd.to_datetime(
        series_forecast["forecast_date"]
    )

    historical = historical_data[series_id].tail(history_days)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=historical.index,
            y=historical.values,
            mode="lines",
            name="Historical Demand",
            line=dict(width=2)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=series_forecast["forecast_date"],
            y=series_forecast["forecast_demand"],
            mode="lines+markers",
            name="Forecast Demand",
            line=dict(width=3, dash="dash")
        )
    )

    # Connect the final historical point to the first forecast point.
    fig.add_trace(
        go.Scatter(
            x=[
                historical.index[-1],
                series_forecast["forecast_date"].iloc[0]
            ],
            y=[
                historical.iloc[-1],
                series_forecast["forecast_demand"].iloc[0]
            ],
            mode="lines",
            showlegend=False,
            line=dict(width=2, dash="dot")
        )
    )

    fig.update_layout(
        title=f"Historical vs Forecast Demand — {series_id}",
        xaxis_title="Date",
        yaxis_title="Sales Units",
        template="plotly_white",
        hovermode="x unified",
        legend_title="Demand Type"
    )

    return fig


# ==========================================================
# 2. FORECAST DEMAND COMPARISON
# ==========================================================

def plot_forecast_comparison(summary_df):
    """
    Compare total forecast demand across all series.
    """

    required = {"series_id", "total_forecast_demand"}

    if not required.issubset(summary_df.columns):
        raise ValueError(
            f"Summary must contain columns: {sorted(required)}"
        )

    plot_df = summary_df.sort_values(
        "total_forecast_demand",
        ascending=False
    )

    fig = px.bar(
        plot_df,
        x="series_id",
        y="total_forecast_demand",
        color="selected_model",
        text_auto=".2s",
        title="28-Day Forecast Demand by Store and Category",
        labels={
            "series_id": "Store-Category Series",
            "total_forecast_demand": "Forecast Demand (Units)",
            "selected_model": "Selected Model"
        },
        template="plotly_white"
    )

    fig.update_layout(
        xaxis_title="Store-Category Series",
        yaxis_title="Forecast Demand (Units)",
        legend_title="Forecasting Model"
    )

    return fig


# ==========================================================
# 3. MODEL PERFORMANCE COMPARISON
# ==========================================================


def plot_model_performance(model_comparison_df):
    """
    Compare forecasting models using pooled WAPE.
    Lower WAPE indicates better forecasting performance.
    """

    plot_df = model_comparison_df.copy()

    # Standardise column names.
    plot_df.columns = plot_df.columns.str.strip().str.lower()

    required = {"model", "wape"}

    if not required.issubset(plot_df.columns):
        raise ValueError(
            f"Model comparison must contain: {sorted(required)}. "
            f"Found: {plot_df.columns.tolist()}"
        )

    # Backtest WAPE is stored as a fraction.
    plot_df["WAPE_Percent"] = plot_df["wape"] * 100

    plot_df = plot_df.sort_values(
        "WAPE_Percent",
        ascending=True
    )

    fig = px.bar(
        plot_df,
        x="model",
        y="WAPE_Percent",
        color="WAPE_Percent",
        text="WAPE_Percent",
        title="Forecasting Model Comparison — WAPE",
        labels={
            "model": "Forecasting Model",
            "WAPE_Percent": "WAPE (%)"
        },
        template="plotly_white",
        color_continuous_scale="Blues"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="Forecasting Model",
        yaxis_title="WAPE (%)",
        coloraxis_showscale=False
    )

    return fig


# ==========================================================
# 4. FORECAST KPI SUMMARY

from plotly.subplots import make_subplots


def plot_forecast_kpis(summary_df):
    """
    Display total forecast demand separately from daily demand KPIs.
    """

    required = {
        "series_id",
        "total_forecast_demand",
        "average_daily_demand",
        "minimum_daily_demand",
        "maximum_daily_demand"
    }

    if not required.issubset(summary_df.columns):
        raise ValueError(
            f"Summary is missing columns: "
            f"{sorted(required - set(summary_df.columns))}"
        )

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Total Demand — 28-Day Forecast",
            "Daily Demand — Average, Minimum and Maximum"
        ),
        vertical_spacing=0.18
    )

    fig.add_trace(
        go.Bar(
            x=summary_df["series_id"],
            y=summary_df["total_forecast_demand"],
            name="Total Forecast Demand"
        ),
        row=1,
        col=1
    )

    daily_metrics = [
        ("average_daily_demand", "Average Daily Demand"),
        ("minimum_daily_demand", "Minimum Daily Demand"),
        ("maximum_daily_demand", "Maximum Daily Demand")
    ]

    for column, label in daily_metrics:
        fig.add_trace(
            go.Bar(
                x=summary_df["series_id"],
                y=summary_df[column],
                name=label
            ),
            row=2,
            col=1
        )

    fig.update_layout(
        title="Retail Demand Forecast KPI Analysis",
        template="plotly_white",
        height=750,
        barmode="group",
        hovermode="x unified"
    )

    fig.update_xaxes(
        title_text="Store-Category Series",
        row=1,
        col=1
    )

    fig.update_xaxes(
        title_text="Store-Category Series",
        row=2,
        col=1
    )

    fig.update_yaxes(
        title_text="Total Units",
        row=1,
        col=1
    )

    fig.update_yaxes(
        title_text="Daily Units",
        row=2,
        col=1
    )

    return fig


# ==========================================================
# 5. SAVE PLOTLY FIGURES
# ==========================================================

def save_visualizations(figures, output_dir):
    """
    Save each Plotly figure as an interactive HTML file.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for filename, figure in figures.items():

        output_path = output_dir / f"{filename}.html"

        figure.write_html(
            str(output_path),
            include_plotlyjs=True
        )

        saved_files.append(output_path)

        print("Saved:", output_path)

    return saved_files
