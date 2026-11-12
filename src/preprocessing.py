

import numpy as np
import pandas as pd


def prepare_m5_data(sales, calendar, day_cols, n_stores=2, horizon=28):

    # 1. Select stores
    available_stores = sorted(sales["store_id"].unique())

    if n_stores > len(available_stores):
        raise ValueError(
            f"Requested {n_stores} stores, but only "
            f"{len(available_stores)} are available."
        )

    selected_stores = available_stores[:n_stores]

    selected_sales = sales[
        sales["store_id"].isin(selected_stores)
    ].copy()

    # 2. Map each store to its state
    store_state = selected_sales[
        ["store_id", "state_id"]
    ].drop_duplicates()

    state_by_store = (
        store_state.set_index("store_id")["state_id"].to_dict()
    )

    # 3. Aggregate item-level sales into store-category series
    wide = selected_sales.groupby(
        ["store_id", "cat_id"]
    )[day_cols].sum()

    series_ids = [
        f"{store}__{category}"
        for store, category in wide.index
    ]

    Yv = wide.values.T.astype(float)

    # 4. Create historical time-series DataFrame
    T = len(day_cols)
    S = len(series_ids)

    hist_dates = pd.DatetimeIndex(
        calendar["date"].iloc[:T]
    )

    Y = pd.DataFrame(
        Yv,
        index=hist_dates,
        columns=series_ids
    )

    # 5. Validate forecast calendar
    cal_dates = pd.DatetimeIndex(calendar["date"])
    NCAL = len(calendar)

    if NCAL < T + horizon:
        raise ValueError(
            f"Calendar has {NCAL} days, but at least "
            f"{T + horizon} are required."
        )

    if not hist_dates.equals(cal_dates[:T]):
        raise ValueError(
            "Historical dates in sales and calendar do not match."
        )

    if not hist_dates.is_monotonic_increasing:
        raise ValueError("Historical dates are not sorted.")

    # 6. Calendar features
    DOW = cal_dates.dayofweek.values
    MONTH = cal_dates.month.values
    DOM = cal_dates.day.values

    EV = (
        calendar[["event_name_1", "event_name_2"]]
        .notna()
        .any(axis=1)
        .astype(int)
        .values
    )

    # 7. SNAP features for each store-category series
    snap_columns = {
        "CA": "snap_CA",
        "TX": "snap_TX",
        "WI": "snap_WI"
    }

    snap_values = []

    for series_id in series_ids:

        store_id, category = series_id.split("__", 1)
        state = state_by_store[store_id]

        if state not in snap_columns:
            raise ValueError(
                f"Unsupported state for SNAP mapping: {state}"
            )

        snap_column = snap_columns[state]

        snap_values.append(
            calendar[snap_column].to_numpy()
        )

    SNAP = np.column_stack(snap_values).astype(int)

    SNAP_df = pd.DataFrame(
        SNAP,
        index=cal_dates,
        columns=series_ids
    )

    # 8. Return all prepared data
    return {
        "Y": Y,
        "Yv": Yv,
        "T": T,
        "S": S,
        "NCAL": NCAL,
        "series_ids": series_ids,
        "selected_stores": selected_stores,
        "state_by_store": state_by_store,
        "hist_dates": hist_dates,
        "cal_dates": cal_dates,
        "DOW": DOW,
        "MONTH": MONTH,
        "DOM": DOM,
        "EV": EV,
        "SNAP": SNAP,
        "SNAP_df": SNAP_df,
        "HORIZON": horizon
    }
