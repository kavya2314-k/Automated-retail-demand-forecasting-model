

from pathlib import Path

from src.data_loader import load_m5_data
from src.preprocessing import prepare_m5_data
from src.forecast_models import ForecastingModels


# Configuration
DATA_DIR = Path("data")

sales_path = DATA_DIR / "sales_train_evaluation.csv"
calendar_path = DATA_DIR / "calendar.csv"

N_STORES = 2
HORIZON = 28


# Load data
print("Loading datasets...")

sales, calendar, day_cols = load_m5_data(
    calendar_path,
    sales_path
)


# Preprocess data
print("Preparing time-series data...")

data = prepare_m5_data(
    sales,
    calendar,
    day_cols,
    n_stores=N_STORES,
    horizon=HORIZON
)


# Initialize forecasting engine
print("Initializing forecasting models...")

forecaster = ForecastingModels(
    data=data,
    horizon=HORIZON
)


# Validate initialization
assert forecaster.S == 6
assert forecaster.HORIZON == 28

print("\nForecasting engine initialized successfully!")

print("Number of series:", forecaster.S)
print("Forecast horizon:", forecaster.HORIZON)
print("Series:", forecaster.series_ids)

print("\nAll initialization tests passed!")
