

import math
import warnings

import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")


class ForecastingModels:

    def __init__(
        self,
        data,
        horizon=28,
        stat_history=730,
        gbm_origin_step=7,
        first_gbm_origin=60
    ):

        # Prepared data from preprocessing.py
        self.Y = data["Y"]
        self.Yv = data["Yv"]

        self.T = data["T"]
        self.S = data["S"]

        self.series_ids = data["series_ids"]

        self.cal_dates = data["cal_dates"]

        self.DOW = data["DOW"]
        self.MONTH = data["MONTH"]
        self.DOM = data["DOM"]
        self.EV = data["EV"]

        self.SNAP = data["SNAP"]

        self.HORIZON = horizon
        self.STAT_HISTORY = stat_history

        self.GBM_ORIGIN_STEP = gbm_origin_step
        self.FIRST_GBM_ORIGIN = first_gbm_origin

        if self.T < 7:
            raise ValueError("At least 7 historical days are required.")

        if len(self.cal_dates) < self.T + self.HORIZON:
            raise ValueError(
                "Calendar does not contain enough future dates."
            )

        if self.Y.shape != (self.T, self.S):
            raise ValueError("Unexpected historical data dimensions.")

        if self.SNAP.shape != (len(self.cal_dates), self.S):
            raise ValueError("Unexpected SNAP feature dimensions.")

        # Prepare ML training data
        self.FEATS_BASE = [
            "h", "dow", "month", "dom", "event", "snap",
            "sw1", "sw2", "sw3", "sw4",
            "lag0", "roll7", "roll14", "l364"
        ]

        self.FEATS_LGB = self.FEATS_BASE + ["series_code"]

        self.ROWS = None

    # --------------------------------------------------
    # 1. GET HISTORICAL SALES
    # --------------------------------------------------

    def get_hist(self, series_id, forecast_origin):

        if series_id not in self.series_ids:
            raise ValueError(f"Unknown series: {series_id}")

        if forecast_origin < 0 or forecast_origin >= self.T:
            raise ValueError("Invalid forecast origin.")

        end_position = forecast_origin + 1

        start_position = max(
            0,
            end_position - self.STAT_HISTORY
        )

        return self.Y[series_id].iloc[
            start_position:end_position
        ]

    # --------------------------------------------------
    # 2. SEASONAL NAIVE
    # --------------------------------------------------

    def fc_snaive(self, series_id, forecast_origin):

        historical_sales = self.get_hist(
            series_id,
            forecast_origin
        )

        sales_values = historical_sales.values

        last_week_sales = sales_values[-7:]

        forecast = np.array([
            last_week_sales[day % 7]
            for day in range(self.HORIZON)
        ])

        return np.maximum(forecast, 0)

    # --------------------------------------------------
    # 3. HOLT-WINTERS
    # --------------------------------------------------

    def fc_holtwinters(self, series_id, forecast_origin):

        historical_sales = self.get_hist(
            series_id,
            forecast_origin
        )

        model = ExponentialSmoothing(
            historical_sales,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=7
        )

        fitted_model = model.fit()

        forecast = fitted_model.forecast(self.HORIZON)

        return np.maximum(
            np.asarray(forecast, dtype=float),
            0
        )

    # --------------------------------------------------
    # 4. ARIMA
    # --------------------------------------------------

    def fc_arima(
        self,
        series_id,
        forecast_origin,
        order=(2, 1, 2),
        **kwargs
    ):

        historical_sales = self.get_hist(
            series_id,
            forecast_origin
        )

        model = ARIMA(
            historical_sales,
            order=order
        )

        fitted_model = model.fit()

        predictions = fitted_model.forecast(
            steps=self.HORIZON
        )

        return np.maximum(
            np.asarray(predictions, dtype=float),
            0
        )

    # --------------------------------------------------
    # 5. SARIMA
    # --------------------------------------------------

    def fc_sarima(
        self,
        series_id,
        forecast_origin,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        **kwargs
    ):

        historical_sales = self.get_hist(
            series_id,
            forecast_origin
        )

        model = SARIMAX(
            historical_sales,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted_model = model.fit(
            disp=False,
            maxiter=100
        )

        predictions = fitted_model.forecast(
            steps=self.HORIZON
        )

        return np.maximum(
            np.asarray(predictions, dtype=float),
            0
        )

    # --------------------------------------------------
    # 6. PROPHET
    # --------------------------------------------------

    def fc_prophet(
        self,
        series_id,
        forecast_origin,
        cps=0.05
    ):

        try:
            from prophet import Prophet
        except ImportError as exc:
            raise ImportError(
                "Prophet is not installed. Install it in tf_env."
            ) from exc

        historical_sales = self.get_hist(
            series_id,
            forecast_origin
        )

        series_position = self.series_ids.index(series_id)

        start_position = (
            forecast_origin + 1 - len(historical_sales)
        )

        end_position = forecast_origin + 1

        training_data = pd.DataFrame({

            "ds": historical_sales.index,

            "y": historical_sales.values,

            "event": self.EV[
                start_position:end_position
            ],

            "snap": self.SNAP[
                start_position:end_position,
                series_position
            ]

        })

        future_dates = self.cal_dates[
            forecast_origin + 1:
            forecast_origin + 1 + self.HORIZON
        ]

        future_data = pd.DataFrame({

            "ds": future_dates,

            "event": self.EV[
                forecast_origin + 1:
                forecast_origin + 1 + self.HORIZON
            ],

            "snap": self.SNAP[
                forecast_origin + 1:
                forecast_origin + 1 + self.HORIZON,
                series_position
            ]

        })

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=cps
        )

        model.add_regressor("event")
        model.add_regressor("snap")

        model.fit(training_data)

        predictions = model.predict(future_data)

        return np.maximum(
            predictions["yhat"].values,
            0
        )

    # --------------------------------------------------
    # 7. MACHINE LEARNING FEATURE ENGINEERING
    # --------------------------------------------------

    def build_rows(self, origins):

        all_rows = []

        for t in origins:

            if t < self.FIRST_GBM_ORIGIN:
                raise ValueError(
                    "ML forecast origin is earlier than "
                    "FIRST_GBM_ORIGIN."
                )

            average_28 = self.Yv[
                t - 27:t + 1
            ].mean(axis=0)

            average_7 = self.Yv[
                t - 6:t + 1
            ].mean(axis=0)

            average_14 = self.Yv[
                t - 13:t + 1
            ].mean(axis=0)

            sales_scale = np.maximum(
                average_28,
                1.0
            )

            for h in range(1, self.HORIZON + 1):

                target_date = t + h

                previous_week_date = (
                    target_date - 7 * math.ceil(h / 7)
                )

                previous_year_date = target_date - 364

                if previous_year_date >= 0:
                    previous_year_sales = self.Yv[
                        previous_year_date
                    ]
                else:
                    previous_year_sales = average_28

                if target_date < self.T:
                    actual_sales = (
                        self.Yv[target_date] / sales_scale
                    )
                else:
                    actual_sales = np.full(
                        self.S,
                        np.nan
                    )

                new_rows = pd.DataFrame({

                    "origin": t,

                    "tgt_idx": target_date,

                    "series_idx": np.arange(self.S),

                    "h": h,

                    "dow": self.DOW[target_date],

                    "month": self.MONTH[target_date],

                    "dom": self.DOM[target_date],

                    "event": self.EV[target_date],

                    "snap": self.SNAP[target_date],

                    "sw1": (
                        self.Yv[previous_week_date] /
                        sales_scale
                    ),

                    "sw2": (
                        self.Yv[previous_week_date - 7] /
                        sales_scale
                    ),

                    "sw3": (
                        self.Yv[previous_week_date - 14] /
                        sales_scale
                    ),

                    "sw4": (
                        self.Yv[previous_week_date - 21] /
                        sales_scale
                    ),

                    "lag0": self.Yv[t] / sales_scale,

                    "roll7": average_7 / sales_scale,

                    "roll14": average_14 / sales_scale,

                    "l364": (
                        previous_year_sales /
                        sales_scale
                    ),

                    "series_code": np.arange(self.S),

                    "scale": sales_scale,

                    "target": actual_sales

                })

                all_rows.append(new_rows)

        return pd.concat(
            all_rows,
            ignore_index=True
        )

    # --------------------------------------------------
    # 8. PREPARE MACHINE LEARNING TRAINING DATA
    # --------------------------------------------------

    def prepare_ml_data(self):

        forecast_origins = range(
            self.FIRST_GBM_ORIGIN,
            self.T - 1,
            self.GBM_ORIGIN_STEP
        )

        if len(forecast_origins) == 0:
            raise ValueError(
                "Insufficient historical data for ML training."
            )

        self.ROWS = self.build_rows(
            forecast_origins
        )

        self.ROWS["series_code"] = (
            self.ROWS["series_code"].astype("category")
        )

        return self.ROWS

    # --------------------------------------------------
    # 9. SELECT TRAINING ROWS
    # --------------------------------------------------

    def _train_mask(self, forecast_origin):

        if self.ROWS is None:
            raise RuntimeError(
                "Call prepare_ml_data() before ML forecasting."
            )

        historical_rows = (
            self.ROWS["tgt_idx"] <= forecast_origin
        )

        available_targets = self.ROWS["target"].notna()

        return historical_rows & available_targets

    # --------------------------------------------------
    # 10. CREATE PREDICTION ROWS
    # --------------------------------------------------

    def _pred_rows(self, forecast_origin):

        prediction_data = self.build_rows(
            [forecast_origin]
        )

        if self.ROWS is None:
            raise RuntimeError(
                "Call prepare_ml_data() before ML forecasting."
            )

        prediction_data["series_code"] = pd.Categorical(
            prediction_data["series_code"],
            categories=self.ROWS["series_code"].cat.categories
        )

        return prediction_data

    # --------------------------------------------------
    # 11. LIGHTGBM
    # --------------------------------------------------

    def fc_lgbm(self, forecast_origin, params=None):

        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "LightGBM is not installed in the active environment."
            ) from exc

        if self.ROWS is None:
            self.prepare_ml_data()

        model_parameters = {

            "objective": "regression",

            "learning_rate": 0.05,

            "n_estimators": 300,

            "num_leaves": 15,

            "min_child_samples": 20,

            "subsample": 0.8,

            "subsample_freq": 1,

            "colsample_bytree": 0.8,

            "random_state": 42,

            "n_jobs": -1,

            "verbose": -1

        }

        if params is not None:
            model_parameters.update(params)

        training_data = self.ROWS[
            self._train_mask(forecast_origin)
        ]

        if training_data.empty:
            raise ValueError(
                "No valid training rows for LightGBM."
            )

        model = lgb.LGBMRegressor(
            **model_parameters
        )

        model.fit(
            training_data[self.FEATS_LGB],
            training_data["target"]
        )

        prediction_data = self._pred_rows(
            forecast_origin
        )

        normalized_predictions = model.predict(
            prediction_data[self.FEATS_LGB]
        )

        normalized_predictions = np.maximum(
            normalized_predictions,
            0
        )

        actual_predictions = (
            normalized_predictions *
            prediction_data["scale"].values
        )

        final_predictions = actual_predictions.reshape(
            self.HORIZON,
            self.S
        ).T

        return final_predictions

    # --------------------------------------------------
    # 12. RANDOM FOREST
    # --------------------------------------------------

    def fc_rf(self, forecast_origin, params=None):

        if self.ROWS is None:
            self.prepare_ml_data()

        model_parameters = {

            "n_estimators": 200,

            "min_samples_leaf": 5,

            "max_features": 0.6,

            "n_jobs": -1,

            "random_state": 42

        }

        if params is not None:
            model_parameters.update(params)

        training_data = self.ROWS[
            self._train_mask(forecast_origin)
        ]

        if training_data.empty:
            raise ValueError(
                "No valid training rows for Random Forest."
            )

        model = RandomForestRegressor(
            **model_parameters
        )

        model.fit(
            training_data[self.FEATS_BASE],
            training_data["target"]
        )

        prediction_data = self._pred_rows(
            forecast_origin
        )

        normalized_predictions = model.predict(
            prediction_data[self.FEATS_BASE]
        )

        normalized_predictions = np.maximum(
            normalized_predictions,
            0
        )

        actual_predictions = (
            normalized_predictions *
            prediction_data["scale"].values
        )

        final_predictions = actual_predictions.reshape(
            self.HORIZON,
            self.S
        ).T

        return final_predictions

    # --------------------------------------------------
    # 13. UNIFIED MODEL PREDICTION
    # --------------------------------------------------

    def predict_model(
        self,
        model_name,
        forecast_origin,
        parameters=None
    ):

        parameters = parameters or {}

        if model_name == "SeasonalNaive":

            predictions = np.vstack([
                self.fc_snaive(sid, forecast_origin)
                for sid in self.series_ids
            ])

        elif model_name == "HoltWinters":

            predictions = np.vstack([
                self.fc_holtwinters(sid, forecast_origin)
                for sid in self.series_ids
            ])

        elif model_name == "ARIMA":

            predictions = np.vstack([
                self.fc_arima(
                    sid,
                    forecast_origin,
                    order=parameters.get(
                        "arima_order",
                        (2, 1, 2)
                    )
                )
                for sid in self.series_ids
            ])

        elif model_name == "SARIMA":

            predictions = np.vstack([
                self.fc_sarima(
                    sid,
                    forecast_origin,
                    order=parameters.get(
                        "sarima_order",
                        (1, 1, 1)
                    ),
                    seasonal_order=parameters.get(
                        "seasonal_order",
                        (1, 1, 1, 7)
                    )
                )
                for sid in self.series_ids
            ])

        elif model_name == "Prophet":

            predictions = np.vstack([
                self.fc_prophet(
                    sid,
                    forecast_origin,
                    cps=parameters.get(
                        "prophet_cps",
                        0.05
                    )
                )
                for sid in self.series_ids
            ])

        elif model_name == "RandomForest":

            predictions = self.fc_rf(
                forecast_origin,
                params=parameters.get("rf")
            )

        elif model_name == "LightGBM":

            predictions = self.fc_lgbm(
                forecast_origin,
                params=parameters.get("lgbm")
            )

        else:
            raise ValueError(
                f"Unknown forecasting model: {model_name}"
            )

        predictions = np.asarray(
            predictions,
            dtype=float
        )

        if predictions.shape != (
            self.S,
            self.HORIZON
        ):
            raise ValueError(
                f"Expected {(self.S, self.HORIZON)}, "
                f"received {predictions.shape}."
            )

        if not np.isfinite(predictions).all():
            raise ValueError(
                "Forecast contains NaN or infinite values."
            )

        return np.maximum(predictions, 0)
