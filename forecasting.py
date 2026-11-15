"""Gaussian-process forecasting for the 1-11 November 2026 study window."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, ExpSineSquared, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


SEED = 42
MIN_TRAIN_DAYS = 7


def _day_index(dates, origin):
    dates = pd.DatetimeIndex(dates)
    return ((dates - pd.Timestamp(origin)).days.to_numpy(dtype=float)).reshape(-1, 1)


def _kernel():
    """Weekly periodic signal plus observation noise.

    The seven-day period is fixed because eleven observations are not sufficient to
    estimate the period reliably. Amplitude, smoothness and noise are learned.
    """
    periodic = ExpSineSquared(
        length_scale=1.0,
        periodicity=7.0,
        length_scale_bounds=(0.2, 10.0),
        periodicity_bounds="fixed",
    )
    return (
        ConstantKernel(1.0, (0.1, 10.0)) * periodic
        + WhiteKernel(0.03, (1e-4, 0.5))
    )


@dataclass
class GPRFit:
    model: GaussianProcessRegressor
    origin: pd.Timestamp


def fit_gpr(series: pd.Series) -> GPRFit:
    series = series.dropna().sort_index().astype(float)
    if len(series) < MIN_TRAIN_DAYS:
        raise ValueError("GPR needs at least seven consecutive daily observations.")
    origin = pd.Timestamp(series.index.min())
    model = GaussianProcessRegressor(
        kernel=_kernel(),
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=SEED,
    )
    model.fit(_day_index(series.index, origin), series.to_numpy())
    return GPRFit(model=model, origin=origin)


def predict_gpr(fit: GPRFit, dates) -> pd.DataFrame:
    dates = pd.DatetimeIndex(dates)
    mean, std = fit.model.predict(_day_index(dates, fit.origin), return_std=True)
    lower = np.clip(mean - 1.96 * std, 0, None)
    upper = mean + 1.96 * std
    return pd.DataFrame(
        {
            "Date": dates,
            "Predicted": mean,
            "Std. dev.": std,
            "Lower 95%": lower,
            "Upper 95%": upper,
        }
    )


def seasonal_naive(history: pd.Series, date) -> float:
    prior = pd.Timestamp(date) - pd.Timedelta(days=7)
    if prior not in history.index:
        raise ValueError(f"No observation is available for {prior:%Y-%m-%d}.")
    return float(history.loc[prior])


def summarize_predictions(comparison: pd.DataFrame):
    """Calculate common regression metrics for a comparison frame."""
    rows = []
    for name in ("GPR", "Seasonal Naive"):
        observed = comparison["Observed"].to_numpy()
        predicted = comparison[name].to_numpy()
        rows.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(observed, predicted),
                "RMSE": np.sqrt(mean_squared_error(observed, predicted)),
                "MAPE (%)": 100 * mean_absolute_percentage_error(observed, predicted),
            }
        )
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def rolling_origin_evaluation(series: pd.Series, min_train_days=MIN_TRAIN_DAYS):
    """One-step forecasts for days 8 onward, with refitting after each day."""
    series = series.dropna().sort_index().astype(float)
    if len(series) <= min_train_days:
        raise ValueError("Evaluation needs at least eight daily observations.")
    records = []
    for end in range(min_train_days, len(series)):
        train = series.iloc[:end]
        date = series.index[end]
        predicted = float(predict_gpr(fit_gpr(train), [date])["Predicted"].iloc[0])
        records.append(
            {
                "Date": date,
                "Observed": float(series.iloc[end]),
                "GPR": predicted,
                "Seasonal Naive": seasonal_naive(series, date),
            }
        )
    comparison = pd.DataFrame(records).set_index("Date")
    return summarize_predictions(comparison), comparison


def forecast_series(series: pd.Series, horizon=3):
    series = series.dropna().sort_index().astype(float)
    future_dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
    return predict_gpr(fit_gpr(series), future_dates)

