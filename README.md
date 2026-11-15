# Hong Kong Passenger Flow Forecast

A reproducible MSc research demonstration that uses official Hong Kong Immigration Department data from **1-11 November 2026** to forecast arrivals, departures and total clearance trips for the following three days.

The current stage uses **Gaussian Process Regression (GPR)** with a fixed seven-day periodic kernel. It does not claim to predict MTR station crowding: immigration-clearance trips are a public proxy dataset, while the dissertation ultimately requires station-level passenger flow, train-headway data and a congestion-control simulation.

## Current research question

Can a small probabilistic machine-learning model extract a weekly pattern from eleven official daily observations and produce transparent three-day forecasts with uncertainty?

## Why the technology stack was changed

The first version compared Seasonal Naive, Random Forest and a PyTorch LSTM using a 28-day lookback and a 30-day test set. That design contradicted the stated eleven-day study window. An LSTM cannot form a 28-day input sequence from eleven observations, and adding PyTorch substantially increases deployment time without solving the data shortage. Random Forest also provides no native predictive distribution and does not encode the known seven-day cycle directly.

The revised stack is deliberately smaller:

| Component | Technology | Purpose |
|---|---|---|
| Data retrieval | Requests | Download the official CSV with an optional authorized local fallback |
| Data processing | Pandas and NumPy | Parse dates, aggregate all control points, and separate arrival and departure flows |
| Machine learning | scikit-learn GPR | Learn a weekly periodic relationship and return forecast mean and uncertainty |
| Interface | Streamlit | Present the fixed study data, forecasts and validation results |
| Visualization | Plotly | Plot observations, forecasts and approximate 95% intervals |
| Testing | unittest | Check the fixed period, rolling validation and interval ordering |

PyTorch, Random Forest and LSTM have been removed from the current prototype. They may become relevant only after authorized high-frequency station data are available.

## Method

Two GPR models are fitted independently: one for arrivals and one for departures. Their mean forecasts are summed to produce total daily flow.

The covariance kernel is:

`ConstantKernel × ExpSineSquared(periodicity=7 days) + WhiteKernel`

- `ExpSineSquared` represents weekly repetition.
- The period is fixed at seven days because eleven samples are insufficient to estimate it reliably.
- `ConstantKernel` learns the signal amplitude.
- `WhiteKernel` represents unexplained observation noise.
- `normalize_y=True` stabilizes optimization when passenger counts are large.
- The model returns a predictive standard deviation, which is converted to a 95% interval.

Validation uses an expanding time window. Days 1-7 predict day 8, days 1-8 predict day 9, and so on through day 11. This produces only four validation forecasts, so the metrics are exploratory.

When the official source contains observations for 12-14 November, the application shows a separate post-forecast audit. Those values are never supplied to model fitting.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Run tests:

```bash
python -m unittest discover -s tests
```

## Data source

- [Official dataset page](https://data.gov.hk/en-data/dataset/hk-immd-set5-statistics-daily-passenger-traffic)
- [Official CSV](https://www.immd.gov.hk/opendata/eng/transport/immigration_clearance/statistics_on_daily_passenger_traffic.csv)
- Required fields: `Date`, `Arrival / Departure`, and `Total`

The records are clearance trips, not unique persons, railway entries, train loads or platform occupancy.

## Dissertation extension

The full MSc project requires three later modules that are outside the public-data prototype:

1. Station-level passenger forecasting using authorized 5-15 minute entry, exit, transfer and platform-count data.
2. Train-headway forecasting using train event or AVL data along the line.
3. A simulation environment that tests holding, dispatching, passenger-flow control or skip-stop policies against waiting time and platform crowding.

See [REPORT.md](REPORT.md) for the literature review, report structure and implementation roadmap.

## License

Original project code is MIT licensed. The government dataset and cited papers remain subject to their own terms and attribution requirements.

