"""Reproducible one-day-ahead passenger-flow model comparison.

All estimators fit on observations before the final chronological test window.
Test predictions are rolling one-step-ahead forecasts; future dates are recursively forecast.
"""
from dataclasses import dataclass
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

LOOKBACK = 28
SEED = 42


def calendar_features(dates):
    d = pd.DatetimeIndex(dates)
    return pd.DataFrame({
        'dow': d.dayofweek,
        'month': d.month,
        'day': d.day,
        'week': d.isocalendar().week.astype(int).to_numpy(),
        'is_weekend': (d.dayofweek >= 5).astype(int),
        'time_index': (d - pd.Timestamp('2021-01-01')).days.to_numpy(),
    }, index=d)


def same_weekday_forecast(history: pd.Series, dates):
    """One-step seasonal-naive forecasts, using ONLY previously observed values."""
    result = []
    for dt in dates:
        prior = pd.Timestamp(dt) - pd.Timedelta(days=7)
        if prior in history.index:
            result.append(float(history.loc[prior]))
        else:
            result.append(float(history.loc[history.index < dt].tail(7).median()))
    return np.asarray(result, dtype=float)


def rf_forecast(train, dates):
    model = RandomForestRegressor(n_estimators=180, min_samples_leaf=3,
                                  max_features=0.8, random_state=SEED, n_jobs=-1)
    model.fit(calendar_features(train.index), train.to_numpy(dtype=float))
    return np.clip(model.predict(calendar_features(dates)), 0, None)


def make_windows(vals, lookback=LOOKBACK):
    x, y = [], []
    for end in range(lookback, len(vals)):
        x.append(vals[end - lookback:end])
        y.append(vals[end])
    return np.asarray(x, dtype=np.float32)[..., None], np.asarray(y, dtype=np.float32)[..., None]


@dataclass
class LSTMFit:
    model: object
    scaler: StandardScaler
    lookback: int
    epochs_trained: int


def fit_lstm(train: pd.Series, lookback=LOOKBACK, max_epochs=35):
    """Use last 15% of pre-test history as validation; no future test samples used."""
    import torch
    from torch import nn

    if len(train) < max(90, lookback + 35):
        raise ValueError('LSTM needs at least 90 consecutive historical days.')
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(2, torch.get_num_threads()))

    cutoff = max(lookback + 20, int(len(train) * .85))
    fit_values = train.iloc[:cutoff].to_numpy(dtype=float)
    scaler = StandardScaler().fit(fit_values.reshape(-1, 1))
    standardized = scaler.transform(train.to_numpy(dtype=float).reshape(-1, 1)).ravel()
    X, Y = make_windows(standardized, lookback)
    end_indices = np.arange(lookback, len(train))
    train_mask = end_indices < cutoff
    val_mask = ~train_mask
    xtr = torch.from_numpy(X[train_mask])
    ytr = torch.from_numpy(Y[train_mask])
    xval = torch.from_numpy(X[val_mask])
    yval = torch.from_numpy(Y[val_mask])

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(input_size=1, hidden_size=24, num_layers=1, batch_first=True)
            self.out = nn.Linear(24, 1)
        def forward(self, x):
            seq, _ = self.rnn(x)
            return self.out(seq[:, -1, :])

    net = _Net()
    optimizer = torch.optim.Adam(net.parameters(), lr=.008)
    loss_fn = nn.MSELoss()
    best_loss, best_state, patience, epoch_count = float('inf'), None, 0, 0
    for epoch in range(max_epochs):
        net.train()
        order = torch.randperm(len(xtr))
        for inds in order.split(128):
            optimizer.zero_grad()
            loss = loss_fn(net(xtr[inds]), ytr[inds])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
        net.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(net(xval), yval).item())
        epoch_count = epoch + 1
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 7:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return LSTMFit(net, scaler, lookback, epoch_count)


def lstm_predict_one(fit: LSTMFit, past: pd.Series):
    import torch
    tail = past.iloc[-fit.lookback:].to_numpy(dtype=float)
    if len(tail) < fit.lookback:
        raise ValueError('Insufficient LSTM input history')
    normalized = fit.scaler.transform(tail.reshape(-1, 1)).astype(np.float32)
    x = torch.from_numpy(normalized.reshape(1, fit.lookback, 1))
    with torch.no_grad():
        scaled_pred = float(fit.model(x).item())
    return max(0.0, float(fit.scaler.inverse_transform([[scaled_pred]])[0, 0]))


def evaluate_models(series: pd.Series, test_days=30):
    """Evaluate on the same final dates, with no fitting on the held-out window."""
    if len(series) < 120:
        raise ValueError('At least 120 consecutive daily observations are needed for three-model comparison.')
    train = series.iloc[:-test_days]
    test = series.iloc[-test_days:]
    rf_preds = rf_forecast(train, test.index)
    seasonal = same_weekday_forecast(series, test.index)
    fit = fit_lstm(train)
    preds_lstm = []
    for dt in test.index:
        preds_lstm.append(lstm_predict_one(fit, series.loc[series.index < dt]))
    pred_map = {
        'Seasonal Naive': seasonal,
        'Random Forest': rf_preds,
        'LSTM': np.asarray(preds_lstm),
    }
    summary = pd.DataFrame([
        {'Model': name, 'MAE': mean_absolute_error(test.values, values),
         'RMSE': np.sqrt(mean_squared_error(test.values, values))}
        for name, values in pred_map.items()
    ]).sort_values('MAE').reset_index(drop=True)
    comparison = pd.DataFrame({'Observed': test.to_numpy()}, index=test.index)
    for name, values in pred_map.items():
        comparison[name] = values
    return summary, comparison, fit


def predict_future(series, horizon, chosen, lstm_fit=None):
    days = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
    if chosen == 'Seasonal Naive':
        extended = series.copy()
        results = []
        for dt in days:
            value = float(same_weekday_forecast(extended, [dt])[0])
            results.append(value)
            extended.loc[dt] = value
    elif chosen == 'Random Forest':
        results = rf_forecast(series, days).tolist()
    else:
        fit = lstm_fit if lstm_fit is not None else fit_lstm(series)
        extended = series.copy()
        results = []
        for dt in days:
            value = lstm_predict_one(fit, extended)
            results.append(value)
            extended.loc[dt] = value
    return pd.DataFrame({'Date': days, 'Predicted trips': np.round(results).astype(int)})
