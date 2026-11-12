# Hong Kong Passenger Flow Forecast — v2 Research Demo

A reproducible Streamlit proof of concept for a Hong Kong University MSc research discussion. It predicts **daily immigration-clearance passenger trips**, **not** MTR station occupancy or congestion.

## Features

- Official Hong Kong Immigration Department daily CSV ingest and optional offline snapshot fallback.
- Filter by border control point and arrival/departure.
- Forecast 1–14 days with **Seasonal Naive**, **Random Forest**, or **PyTorch LSTM**.
- Compare all three on a **common chronological 30-day holdout**, reporting MAE and RMSE, a grouped comparison chart, and actual vs predicted test plot.
- Automatically generated research conclusion based on the measured test metrics and an explicit MTR research-extension roadmap.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

First execution trains the LSTM (CPU). Streamlit caches results for an hour per selected data/filter combination. The LSTM uses 28 lagged days, a fit-only scaler, pre-test validation for early stopping, and no target values from the test interval during fitting. Holdout predictions are **rolling one-day ahead**, with the true past known through the previous day. Future multi-day forecasts are **recursive**; the 30-day holdout score is **not** a validated 14-day recursive forecast score. Model errors are not guaranteed to be stable across time.

## Deploy via Streamlit Community Cloud

1. Push **`app.py`, `forecasting.py`, `requirements.txt`, `README.md`** to the root of the `main` branch of `exefyx/hk-passenger-flow-prediction`.
2. Open https://share.streamlit.io/ → the existing deployed app → **Manage app** → **Reboot** after new commits (usually auto-updates).
3. Confirm the `torch` CPU installation and first-run training succeed in the deployment logs. Streamlit Community Cloud memory/CPU restrictions can make first LSTM run slower.
4. Confirm the metrics and model dropdown on https://hk-passenger-flow-prediction.streamlit.app/.

## Data source

- [Official dataset page](https://data.gov.hk/en-data/dataset/hk-immd-set5-statistics-daily-passenger-traffic)
- [Official CSV](https://www.immd.gov.hk/opendata/eng/transport/immigration_clearance/statistics_on_daily_passenger_traffic.csv)
- Fields: Date (DD-MM-YYYY), Control Point, Arrival / Departure, Total. The data reflect **clearance trips**, not unique persons or railway entries.
- If official CSV downloading is restricted by the host, optionally add an authorized snapshot at `data/passenger_traffic.csv` and identify it as a snapshot. A snapshot is not live data.

## Research limitations

This is a technical demonstration, not a production-grade MTR operation or dispatch system. It has no real-time station input, train-headway information, simulation, exogenous variables, or congestion interventions. Before production use: rolling-origin and multi-horizon validation, checking coverage bias and data leakage, and cross-station evaluation would be required. The Streamlit app learns on each filter setting and only displays one 30-day test period.

## License

Original demo code MIT. The government dataset remains subject to its own terms and attribution requirements.
