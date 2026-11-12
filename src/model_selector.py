 
from pathlib import Path
import json

import numpy as np
import pandas as pd


class ModelSelector:
    """
    Select the best forecasting model overall and for each series
    using rolling-origin backtesting predictions.
    """

    REQUIRED_COLUMNS = {
        "fold",
        "series_id",
        "model",
        "actual",
        "forecast"
    }

    def __init__(self, predictions_path, output_dir):
        self.predictions_path = Path(predictions_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.predictions = None
        self.series_comparison = None
        self.overall_comparison = None
        self.selected_models = None

    def load_predictions(self):
        if not self.predictions_path.exists():
            raise FileNotFoundError(
                f"Backtest predictions not found: "
                f"{self.predictions_path}"
            )

        df = pd.read_csv(self.predictions_path)

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        if df.empty:
            raise ValueError("Backtest predictions file is empty.")

        numeric_columns = ["actual", "forecast", "fold"]
        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

        if df[numeric_columns].isna().any().any():
            raise ValueError(
                "Backtest file contains missing or non-numeric "
                "actual, forecast, or fold values."
            )

        if not np.isfinite(
            df[["actual", "forecast"]].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Actual or forecast values contain infinity."
            )

        if (df["actual"] < 0).any():
            raise ValueError("Actual demand contains negative values.")

        if (df["forecast"] < 0).any():
            raise ValueError("Forecast contains negative values.")

        df["series_id"] = df["series_id"].astype(str)
        df["model"] = df["model"].astype(str)

        self.predictions = df

        print("Backtest predictions loaded successfully.")
        print("Prediction records:", len(df))
        print("Models:", sorted(df["model"].unique()))
        print("Series:", sorted(df["series_id"].unique()))
        print("Validation folds:", sorted(df["fold"].unique()))

        return df

    @staticmethod
    def calculate_metrics(group):
        actual = group["actual"].to_numpy(dtype=float)
        forecast = group["forecast"].to_numpy(dtype=float)

        errors = forecast - actual
        denominator = np.abs(actual).sum()

        wape = (
            np.abs(errors).sum() / denominator
            if denominator > 0
            else np.nan
        )

        bias = (
            errors.sum() / denominator
            if denominator > 0
            else np.nan
        )

        return pd.Series({
            "MAE": np.abs(errors).mean(),
            "RMSE": np.sqrt(np.mean(errors ** 2)),
            "WAPE": wape,
            "Bias": bias,
            "Folds": group["fold"].nunique(),
            "Records": len(group)
        })

    def compare_models(self):
        if self.predictions is None:
            self.load_predictions()

        # Performance of every model on every series.
        series_results = (
            self.predictions
            .groupby(["series_id", "model"], sort=True)
            .apply(self.calculate_metrics)
            .reset_index()
        )

        series_results = series_results.sort_values(
            ["series_id", "WAPE", "MAE", "RMSE"]
        ).reset_index(drop=True)

        # Overall performance pooled across all series and folds.
        overall_results = (
            self.predictions
            .groupby("model", sort=True)
            .apply(self.calculate_metrics)
            .reset_index()
        )

        overall_results = overall_results.sort_values(
            ["WAPE", "MAE", "RMSE"]
        ).reset_index(drop=True)

        self.series_comparison = series_results
        self.overall_comparison = overall_results

        return series_results, overall_results

    def select_best_models(self, minimum_folds=3):
        if self.series_comparison is None:
            self.compare_models()

        comparison = self.series_comparison.copy()

        insufficient = comparison[
            comparison["Folds"] < minimum_folds
        ]

        if not insufficient.empty:
            raise ValueError(
                "Some model-series combinations have fewer than "
                f"{minimum_folds} validation folds."
            )

        # First row for each series is the lowest-WAPE model.
        selected = (
            comparison
            .sort_values(["series_id", "WAPE", "MAE", "RMSE"])
            .groupby("series_id", as_index=False)
            .first()
        )

        self.selected_models = {
            row["series_id"]: row["model"]
            for _, row in selected.iterrows()
        }

        return selected

    def save_results(self):
        if self.selected_models is None:
            self.select_best_models()

        series_path = (
            self.output_dir / "model_selection_by_series.csv"
        )
        overall_path = (
            self.output_dir / "model_selection_overall.csv"
        )
        config_path = (
            self.output_dir / "selected_models.json"
        )

        self.series_comparison.to_csv(
            series_path, index=False
        )

        self.overall_comparison.to_csv(
            overall_path, index=False
        )

        configuration = {
            "selection_method": "lowest pooled WAPE across rolling folds",
            "validation_folds": int(
                self.predictions["fold"].nunique()
            ),
            "selected_models": self.selected_models,
            "overall_best_model": str(
                self.overall_comparison.iloc[0]["model"]
            )
        }

        with open(config_path, "w", encoding="utf-8") as file:
            json.dump(configuration, file, indent=4)

        print("\nModel selection files saved:")
        print(series_path)
        print(overall_path)
        print(config_path)

        return configuration


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    selector = ModelSelector(
        predictions_path=(
            project_root
            / "outputs"
            / "rolling_backtest_predictions.csv"
        ),
        output_dir=project_root / "outputs"
    )

    selector.load_predictions()

    series_comparison, overall_comparison = (
        selector.compare_models()
    )

    selected = selector.select_best_models(
        minimum_folds=3
    )

    configuration = selector.save_results()

    print("\nOVERALL MODEL RANKING")
    print(
        overall_comparison[
            ["model", "MAE", "RMSE", "WAPE", "Bias"]
        ].to_string(index=False)
    )

    print("\nBEST MODEL FOR EACH SERIES")
    print(
        selected[
            ["series_id", "model", "WAPE", "MAE", "RMSE"]
        ].to_string(index=False)
    )

    print("\nSELECTED MODEL CONFIGURATION")
    print(json.dumps(configuration, indent=4))
