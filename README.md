# Automated Retail Demand Forecasting System

**An end-to-end machine learning solution for automated retail demand forecasting, model selection and interactive business intelligence.**

## 1. Project Overview

Retail demand forecasting is essential for inventory planning, replenishment, supply-chain optimization and business decision-making.

This project develops an automated demand forecasting system that predicts future product demand using statistical and machine learning approaches. It combines time-series preprocessing, feature engineering, model evaluation, automated model selection and interactive visualization.

The system generates **28-day demand forecasts** and evaluates forecasting models using rolling-origin validation to identify suitable models for individual demand series.

## 2. Key Features

- Automated 28-day retail demand forecasting.
- Statistical and machine learning forecasting models.
- Time-series feature engineering using lag, rolling-window and calendar features.
- Automated model selection across demand series.
- Three-fold rolling-origin validation.
- Forecast accuracy evaluation using MAPE and other performance metrics.
- Series-level forecasting diagnostics.
- Interactive Streamlit dashboard.
- Forecast comparison and model performance visualizations.
- Exportable forecasting results and reports.

## 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, LightGBM |
| Statistical Forecasting | Statsmodels, Prophet |
| Visualization | Matplotlib, Plotly |
| Dashboard | Streamlit |
| Development | Jupyter Notebook, VS Code |

## 4. Dataset

The project uses the **M5 Forecasting dataset**, which contains historical Walmart retail sales data.

The major data files are:

- `sales_train_evaluation.csv` – Historical daily product-level sales.
- `sell_prices.csv` – Product selling prices across stores and weeks.
- `calendar.csv` – Calendar dates, events and relevant temporal information.

The two large CSV files are excluded from this GitHub repository because of their size. They must be downloaded separately from the original M5 dataset source.

**Dataset source:** [M5 Forecasting – Accuracy Competition (Kaggle)](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)

Place the downloaded files inside the `data/` directory before running the forecasting pipeline.

## 5. Forecasting Methodology

The system follows an automated forecasting workflow:

1. **Data Loading:** Load historical sales, pricing and calendar information.
2. **Preprocessing:** Prepare the time-series data and handle relevant data transformations.
3. **Feature Engineering:** Generate lag-based, rolling-window and calendar features.
4. **Model Training:** Train and evaluate the implemented statistical and machine learning forecasting models.
5. **Model Validation:** Apply three-fold rolling-origin validation to evaluate forecasting performance over successive time periods.
6. **Automated Model Selection:** Compare model performance and select suitable models for individual demand series.
7. **Forecast Generation:** Generate 28-day future demand predictions.
8. **Visualization:** Present forecasts, model comparisons and performance indicators through interactive visualizations.

## 6. Model Evaluation

The forecasting system evaluates models using time-series validation rather than relying on a random train-test split.

The evaluation includes:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Rolling-origin fold comparison
- Series-level model performance

**Reported result:** LightGBM achieved a MAPE of **7.93%** in the project's reported evaluation.

The model comparison and selection results are available in the `outputs/` directory.

## 7. Project Structure

```text
M5_Automated_Forecasting/
│
├── data/
│   └── calendar.csv
│
├── notebooks/
│   └── Walnut sales forecasting.ipynb
│
├── outputs/
│   ├── automated_28_day_forecast.csv
│   ├── automated_forecast_summary.csv
│   ├── best_model_per_series.csv
│   ├── model_selection_overall.csv
│   ├── rolling_backtest_model_comparison.csv
│   └── visualizations/
│       ├── forecast_comparison.html
│       ├── forecast_kpis.html
│       └── model_performance.html
│
├── src/
│   ├── data_loader.py
│   ├── forecast_models.py
│   ├── model_selector.py
│   ├── preprocessing.py
│   └── visualization.py
│
├── app.py
├── run_automated_forecast.py
├── run_rolling_backtest.py
├── run_series_diagnostics.py
├── run_visualizations.py
│
├── test_data_pipeline.py
├── test_forecast_execution.py
├── test_forecast_models.py
│
├── requirements.txt
├── .gitignore
└── README.md
```


## 8. Results and Visualizations

The repository includes selected interactive HTML visualizations:

- Forecast comparison
- Forecast KPI analysis
- Model performance comparison

It also contains generated forecasting results, rolling-backtest comparisons and model-selection reports in the `outputs/` directory.

## 9. Business Applications

This forecasting solution can support:

- Retail inventory planning.
- Demand and replenishment forecasting.
- Product-level sales analysis.
- Forecast accuracy monitoring.
- Data-driven supply-chain decisions.
- Identification of products with challenging demand patterns.

## 10. Future Enhancements

- Integration of external economic and market indicators.
- Advanced ensemble forecasting strategies.
- Forecast explainability using SHAP.
- Automated model monitoring and retraining.
- Cloud deployment and scheduled forecasting.
- Integration with Agentic AI for natural-language business analytics.

