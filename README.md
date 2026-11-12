# Hong Kong Passenger Flow Forecast — Research Demo

A lightweight proof of concept showing how to fetch public daily passenger traffic, evaluate a predictive baseline, and forecast the next three days. Designed as a technical demonstration for an MSc research discussion, **not** an operational MTR congestion system.

## Data
- Official source: [Hong Kong Immigration Department — Statistics on Daily Passenger Traffic](https://data.gov.hk/en-data/dataset/hk-immd-set5-statistics-daily-passenger-traffic)
- CSV: https://www.immd.gov.hk/opendata/eng/transport/immigration_clearance/statistics_on_daily_passenger_traffic.csv
- Fields: Date (DD-MM-YYYY), Control Point, Arrival / Departure, Hong Kong Residents, Mainland Visitors, Other Visitors, Total.
- Observations represent immigration clearance **trips**, not unique passengers or MTR entries.

## What the demo does
1. Loads the official CSV at runtime (cached 1 hour).
2. Aggregates total daily passenger trips; filter by control point and direction.
3. Builds two forecasting benchmarks: same-weekday seasonal naive and calendar-feature Random Forest.
4. Performs a chronological held-out backtest and displays MAE / RMSE.
5. Forecasts the next 1–14 days (default 3), plus interactive plots.

## Run locally
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

If the official URL cannot be downloaded from your network, save the English CSV as `data/passenger_traffic.csv`. This file is intentionally not bundled; do not treat an old snapshot as live data.

## Publish on your GitHub and deploy Streamlit
1. Create a public GitHub repository, e.g. `hk-passenger-flow-demo`.
2. Upload `app.py`, `requirements.txt`, and `README.md` to its default branch.
3. Go to https://share.streamlit.io/ and choose **Create app** → **From existing repo**. Select your repository, branch, and entrypoint `app.py`.
4. Deploy. Share both your GitHub repository URL and the deployed Streamlit app URL with your supervisor.
5. If the remote CSV cannot be fetched in cloud hosting, put an authorized local snapshot at `data/passenger_traffic.csv` and describe its date clearly. Do not silently show synthetic data as official observations.

## Research limitations
- These are historical official clearance volumes, not real-time railway station flows or train headways.
- Calendar-only models are simple demonstrations. They should be compared with stronger baselines, seasonality and event controls using a longer historical period.
- A temporal holdout is shown, but a robust study should use rolling-origin backtesting, station-level data, and external operational covariates.
- This repository is original demo scaffolding; it does not copy external project's code.

## License
Project source under MIT License (see LICENSE). Official government data remains subject to its own terms and attribution requirements.
