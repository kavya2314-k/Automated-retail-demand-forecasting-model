

from pathlib import Path

from src.data_loader import load_m5_data
from src.preprocessing import prepare_m5_data


# Configuration
DATA_DIR = Path("data")

CALENDAR_PATH = DATA_DIR / "calendar.csv"
SALES_PATH = DATA_DIR / "sales_train_evaluation.csv"

N_STORES = 2
HORIZON = 28


# 1. Load datasets
print("\nLoading M5 datasets...")

sales, calendar, day_cols = load_m5_data(
    CALENDAR_PATH,
    SALES_PATH
)


# 2. Prepare time-series data
print("\nPreparing time-series data...")

data = prepare_m5_data(
    sales,
    calendar,
    day_cols,
    n_stores=N_STORES,
    horizon=HORIZON
)


# 3. Extract prepared variables
Y = data["Y"]
Yv = data["Yv"]

T = data["T"]
S = data["S"]

series_ids = data["series_ids"]

DOW = data["DOW"]
MONTH = data["MONTH"]
DOM = data["DOM"]
EV = data["EV"]

SNAP = data["SNAP"]
SNAP_df = data["SNAP_df"]

cal_dates = data["cal_dates"]


# 4. Validate dimensions
assert Y.shape == (T, S)
assert Yv.shape == (T, S)

assert len(series_ids) == S

assert len(cal_dates) >= T + HORIZON

assert SNAP.shape == (len(cal_dates), S)
assert SNAP_df.shape == (len(cal_dates), S)


# 5. Validate missing values
assert Y.isnull().sum().sum() == 0
assert SNAP_df.isnull().sum().sum() == 0


# 6. Display results
print("\n========== DATA PIPELINE RESULTS ==========")

print("Sales dataset shape:", sales.shape)
print("Calendar dataset shape:", calendar.shape)

print("Historical days:", T)
print("Number of series:", S)
print("Selected stores:", data["selected_stores"])

print("\nSeries IDs:")
print(series_ids)

print("\nTime-series shape:", Y.shape)
print("SNAP feature shape:", SNAP.shape)

print("\nHistorical date range:")
print(Y.index.min(), "to", Y.index.max())

print("\nFuture forecast dates:")
print(cal_dates[T:T + HORIZON][0], "to",
      cal_dates[T:T + HORIZON][-1])

print("\nTime-series preview:")
print(Y.head())

print("\nAll data pipeline tests passed!")
