"""Eleven-day Hong Kong passenger-flow GPR research demonstration."""

from pathlib import Path
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from forecasting import forecast_series, rolling_origin_evaluation, summarize_predictions


SOURCE = "https://www.immd.gov.hk/opendata/eng/transport/immigration_clearance/statistics_on_daily_passenger_traffic.csv"
DATASET = "https://data.gov.hk/en-data/dataset/hk-immd-set5-statistics-daily-passenger-traffic"
LOCAL = Path(__file__).parent / "data" / "passenger_traffic.csv"
STUDY_START = pd.Timestamp("2026-11-01")
STUDY_END = pd.Timestamp("2026-11-11")
FORECAST_DAYS = 3

st.set_page_config(page_title="HK Passenger Flow GPR", page_icon="🚉", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    try:
        response = requests.get(SOURCE, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        raw = response.content
        origin = "Official online CSV"
    except requests.RequestException as exc:
        if not LOCAL.exists():
            raise RuntimeError(
                f"Cannot retrieve the official CSV: {exc}. "
                "Add an authorized snapshot at data/passenger_traffic.csv."
            ) from exc
        raw = LOCAL.read_bytes()
        origin = "Authorized local snapshot"

    data = None
    for encoding in ("utf-8-sig", "utf-8", "big5"):
        try:
            data = pd.read_csv(io.BytesIO(raw), encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if data is None:
        raise ValueError("CSV encoding was not recognized.")

    data.columns = data.columns.str.replace("\ufeff", "", regex=False).str.strip()
    required = {"Date", "Arrival / Departure", "Total"}
    if not required.issubset(data.columns):
        raise ValueError(f"Unexpected schema: {data.columns.tolist()}")
    data["date"] = pd.to_datetime(data["Date"], format="%d-%m-%Y", errors="coerce")
    data["total"] = pd.to_numeric(data["Total"], errors="coerce")
    data["direction"] = data["Arrival / Departure"].astype(str).str.strip().str.casefold()
    return data.dropna(subset=["date", "total"]), origin


def study_series(data, direction):
    selected = data.loc[
        data["date"].between(STUDY_START, STUDY_END)
        & data["direction"].eq(direction.casefold())
    ]
    series = selected.groupby("date")["total"].sum().sort_index().astype(float)
    expected = pd.date_range(STUDY_START, STUDY_END)
    if not series.index.equals(expected):
        missing = expected.difference(series.index)
        raise ValueError(f"Study window is incomplete. Missing: {list(missing.strftime('%Y-%m-%d'))}")
    return series


@st.cache_data(show_spinner="Fitting periodic Gaussian-process models...")
def run_analysis(arrival, departure):
    arrival_metrics, arrival_backtest = rolling_origin_evaluation(arrival)
    departure_metrics, departure_backtest = rolling_origin_evaluation(departure)
    arrival_future = forecast_series(arrival, FORECAST_DAYS)
    departure_future = forecast_series(departure, FORECAST_DAYS)
    return (
        arrival_metrics,
        arrival_backtest,
        departure_metrics,
        departure_backtest,
        arrival_future,
        departure_future,
    )


st.title("Hong Kong Daily Passenger Flow Forecast")
st.caption(
    "Gaussian Process Regression using only 1-11 November 2026 official immigration-clearance data"
)

try:
    raw, origin = load_data()
    arrival = study_series(raw, "Arrival")
    departure = study_series(raw, "Departure")
    results = run_analysis(arrival, departure)
except (RuntimeError, ValueError, pd.errors.ParserError) as error:
    st.error(str(error))
    st.markdown(f"[Open the official dataset]({DATASET})")
    st.stop()

arrival_metrics, arrival_bt, departure_metrics, departure_bt, arrival_fc, departure_fc = results
observed = pd.DataFrame({"Arrival": arrival, "Departure": departure})
observed["Total"] = observed.sum(axis=1)

forecast = pd.DataFrame({"Date": arrival_fc["Date"]})
forecast["Arrival"] = arrival_fc["Predicted"]
forecast["Departure"] = departure_fc["Predicted"]
forecast["Total"] = forecast["Arrival"] + forecast["Departure"]
total_std = np.sqrt(arrival_fc["Std. dev."] ** 2 + departure_fc["Std. dev."] ** 2)
forecast["Total lower 95%"] = np.clip(
    forecast["Total"] - 1.96 * total_std,
    0,
    None,
)
forecast["Total upper 95%"] = forecast["Total"] + 1.96 * total_std

total_backtest = pd.DataFrame(
    {
        "Observed": arrival_bt["Observed"] + departure_bt["Observed"],
        "GPR": arrival_bt["GPR"] + departure_bt["GPR"],
        "Seasonal Naive": arrival_bt["Seasonal Naive"] + departure_bt["Seasonal Naive"],
    }
)
total_metrics = summarize_predictions(total_backtest)

st.caption(
    f"Source: {origin} · Fixed training window: {STUDY_START:%Y-%m-%d} to {STUDY_END:%Y-%m-%d}"
)

a, b, c = st.columns(3)
a.metric("Training observations", "11 days")
b.metric("Forecast horizon", "3 days")
c.metric("Model", "Periodic GPR")

st.subheader("01 Official study data")
st.dataframe(observed.round(0).astype(int).style.format("{:,}"), use_container_width=True)

history = go.Figure()
for column in observed.columns:
    history.add_trace(go.Scatter(x=observed.index, y=observed[column], name=column))
history.update_layout(xaxis_title="Date", yaxis_title="Passenger trips", hovermode="x unified")
st.plotly_chart(history, use_container_width=True)

st.subheader("02 Three-day GPR forecast")
display = forecast.copy()
display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
for column in display.columns[1:]:
    display[column] = display[column].round(0).astype(int)
st.dataframe(display.style.format({c: "{:,}" for c in display.columns[1:]}), hide_index=True, use_container_width=True)

chart = go.Figure()
chart.add_trace(go.Scatter(x=observed.index, y=observed["Total"], name="Observed total"))
chart.add_trace(go.Scatter(x=forecast["Date"], y=forecast["Total"], name="GPR forecast", mode="lines+markers", line={"dash": "dash"}))
chart.add_trace(go.Scatter(x=forecast["Date"], y=forecast["Total upper 95%"], line={"width": 0}, showlegend=False))
chart.add_trace(go.Scatter(x=forecast["Date"], y=forecast["Total lower 95%"], fill="tonexty", line={"width": 0}, name="Approximate 95% interval"))
chart.update_layout(xaxis_title="Date", yaxis_title="Passenger trips", hovermode="x unified")
st.plotly_chart(chart, use_container_width=True)

audit_dates = pd.DatetimeIndex(forecast["Date"])
audit_raw = raw.loc[raw["date"].isin(audit_dates)]
audit_pivot = audit_raw.groupby(["date", "direction"])["total"].sum().unstack()
if {"arrival", "departure"}.issubset(audit_pivot.columns) and audit_pivot.index.equals(audit_dates):
    audit = pd.DataFrame(
        {
            "Date": audit_dates,
            "Forecast total": forecast["Total"].to_numpy(),
            "Observed total": (audit_pivot["arrival"] + audit_pivot["departure"]).to_numpy(),
        }
    )
    audit["Absolute error"] = (audit["Observed total"] - audit["Forecast total"]).abs()
    audit["Absolute percentage error (%)"] = 100 * audit["Absolute error"] / audit["Observed total"]
    audit["Inside 95% interval"] = (
        (audit["Observed total"].to_numpy() >= forecast["Total lower 95%"].to_numpy())
        & (audit["Observed total"].to_numpy() <= forecast["Total upper 95%"].to_numpy())
    )
    st.subheader("03 Post-forecast audit")
    st.caption(
        "These observations were released after the 1-11 November training window and are used only for evaluation."
    )
    audit_display = audit.copy()
    audit_display["Date"] = audit_display["Date"].dt.strftime("%Y-%m-%d")
    for column in ("Forecast total", "Observed total", "Absolute error"):
        audit_display[column] = audit_display[column].round(0).astype(int)
    audit_display["Absolute percentage error (%)"] = audit_display["Absolute percentage error (%)"].round(2)
    st.dataframe(audit_display, hide_index=True, use_container_width=True)
    x, y = st.columns(2)
    x.metric("Three-day total MAE", f"{audit['Absolute error'].mean():,.0f}")
    y.metric("Three-day total MAPE", f"{audit['Absolute percentage error (%)'].mean():.2f}%")

st.subheader("04 Time-ordered validation")
st.caption(
    "Days 8-11 are forecast one at a time. Each model is refitted using only earlier observations. "
    "Four validation points are enough for a demonstration, not for a production accuracy claim."
)
left, middle, right = st.columns(3)
with left:
    st.markdown("**Arrival**")
    st.dataframe(arrival_metrics.round(2), hide_index=True, use_container_width=True)
with middle:
    st.markdown("**Departure**")
    st.dataframe(departure_metrics.round(2), hide_index=True, use_container_width=True)
with right:
    st.markdown("**Total**")
    st.dataframe(total_metrics.round(2), hide_index=True, use_container_width=True)

st.subheader("05 Interpretation and project boundary")
st.markdown(
    """
The periodic GPR is a supervised probabilistic model. It learns the similarity between observed days and returns both a mean forecast and uncertainty. The weekly period is fixed at seven days because eleven observations cannot identify seasonality reliably.

This dataset measures border-control clearance trips. It does not measure MTR station entries, platform occupancy, train loads or headway. The next project phase therefore requires authorized station-level 5-15 minute passenger counts and train event data. A headway model and congestion-control simulation must be evaluated separately before the system can support operational decisions.
"""
)
st.info("Research demonstration only. Do not interpret these estimates as MTR congestion predictions.")
st.markdown(f"[Official dataset]({DATASET}) · [Project repository](https://github.com/exefyx/hk-passenger-flow-prediction)")

