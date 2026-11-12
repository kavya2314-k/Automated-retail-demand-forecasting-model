

from pathlib import Path
import re
import pandas as pd


ID_COLUMNS = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id"
]


def validate_file_paths(calendar_path, sales_path):

    calendar_path = Path(calendar_path)
    sales_path = Path(sales_path)

    missing_files = []

    if not calendar_path.exists():
        missing_files.append(str(calendar_path))

    if not sales_path.exists():
        missing_files.append(str(sales_path))

    if missing_files:
        raise FileNotFoundError(
            "Required files not found: " + ", ".join(missing_files)
        )

    return calendar_path, sales_path


def load_m5_data(calendar_path, sales_path):

    calendar_path, sales_path = validate_file_paths(
        calendar_path,
        sales_path
    )

    header = pd.read_csv(sales_path, nrows=0)

    missing_columns = [
        col for col in ID_COLUMNS
        if col not in header.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Sales dataset is missing columns: {missing_columns}"
        )

    day_cols = [
        col for col in header.columns
        if re.fullmatch(r"d_\d+", col)
    ]

    day_cols = sorted(
        day_cols,
        key=lambda col: int(col.split("_")[1])
    )

    if not day_cols:
        raise ValueError(
            "No daily sales columns found."
        )

    sales_dtype = {
        col: "int32"
        for col in day_cols
    }

    sales = pd.read_csv(
        sales_path,
        dtype=sales_dtype
    )

    calendar = pd.read_csv(
        calendar_path,
        parse_dates=["date"]
    )

    required_calendar_columns = [
        "date",
        "d",
        "event_name_1",
        "event_name_2",
        "snap_CA",
        "snap_TX",
        "snap_WI"
    ]

    missing_calendar_columns = [
        col for col in required_calendar_columns
        if col not in calendar.columns
    ]

    if missing_calendar_columns:
        raise ValueError(
            "Calendar dataset is missing columns: "
            f"{missing_calendar_columns}"
        )

    calendar["dnum"] = (
        calendar["d"]
        .str.extract(r"(\d+)", expand=False)
        .astype(int)
    )

    calendar = (
        calendar
        .sort_values("dnum")
        .reset_index(drop=True)
    )

    if sales["id"].duplicated().any():
        raise ValueError(
            "Duplicate item identifiers found."
        )

    if sales[day_cols].isna().any().any():
        raise ValueError(
            "Missing values found in daily sales."
        )

    if (sales[day_cols] < 0).any().any():
        raise ValueError(
            "Negative sales values found."
        )

    if calendar["date"].isna().any():
        raise ValueError(
            "Missing dates found in calendar."
        )

    if calendar["date"].duplicated().any():
        raise ValueError(
            "Duplicate calendar dates found."
        )

    if calendar["dnum"].duplicated().any():
        raise ValueError(
            "Duplicate calendar day identifiers found."
        )

    expected_days = list(range(1, len(day_cols) + 1))

    if calendar["dnum"].iloc[:len(day_cols)].tolist() != expected_days:
        raise ValueError(
            "Sales days and calendar days are not aligned."
        )

    print("M5 datasets loaded successfully.")
    print("Sales shape:", sales.shape)
    print("Calendar shape:", calendar.shape)
    print("Historical days:", len(day_cols))

    return sales, calendar, day_cols
