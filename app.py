"""Hong Kong immigration passenger-flow forecasting: public-data MSc PoC."""
from pathlib import Path
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st
from forecasting import evaluate_models, predict_future

SOURCE = 'https://www.immd.gov.hk/opendata/eng/transport/immigration_clearance/statistics_on_daily_passenger_traffic.csv'
DATASET = 'https://data.gov.hk/en-data/dataset/hk-immd-set5-statistics-daily-passenger-traffic'
LOCAL = Path(__file__).parent / 'data' / 'passenger_traffic.csv'
st.set_page_config(page_title='HK Passenger Flow | Research Demo', page_icon='🚉', layout='wide')

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    try:
        response = requests.get(SOURCE, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        raw, origin = response.content, 'Official online CSV (cached up to one hour)'
    except requests.RequestException as exc:
        if not LOCAL.exists():
            raise RuntimeError(f'Cannot retrieve official CSV: {exc}. Put a permitted snapshot at data/passenger_traffic.csv.') from exc
        raw, origin = LOCAL.read_bytes(), 'Local offline CSV snapshot (not live)'
    data = None
    for enc in ('utf-8-sig', 'utf-8', 'big5'):
        try:
            data = pd.read_csv(io.BytesIO(raw), encoding=enc)
            break
        except UnicodeDecodeError:
            pass
    if data is None:
        raise ValueError('CSV encoding not recognized')
    data.columns = data.columns.str.replace('\ufeff', '', regex=False).str.strip()
    required = {'Date', 'Control Point', 'Arrival / Departure', 'Total'}
    if not required.issubset(data.columns):
        raise ValueError(f'Unexpected schema. Found columns: {data.columns.tolist()}')
    data['date'] = pd.to_datetime(data['Date'].astype(str).str.strip(), format='%d-%m-%Y', errors='coerce')
    data['total'] = pd.to_numeric(data['Total'].astype(str).str.replace(',', '', regex=False), errors='coerce')
    data = data.dropna(subset=['date', 'total']).copy()
    data['Control Point'] = data['Control Point'].astype(str).str.strip()
    data['Arrival / Departure'] = data['Arrival / Departure'].astype(str).str.strip()
    data = data.drop_duplicates(subset=['date', 'Control Point', 'Arrival / Departure'])
    return data, origin

@st.cache_data(ttl=3600, show_spinner='Training and backtesting three models on historical observations…')
def cached_evaluation(series):
    summary, comparison, _ = evaluate_models(series)
    return summary, comparison

@st.cache_data(ttl=3600, show_spinner='Training LSTM and forecasting…')
def cached_lstm_future(series, horizon):
    return predict_future(series, horizon, 'LSTM')

st.title('Hong Kong Passenger Flow Forecast')
st.caption('An MSc research proof of concept · Hong Kong immigration-clearance trips, not MTR train station entries')

try:
    raw, origin = load_data()
except (RuntimeError, ValueError, pd.errors.ParserError) as error:
    st.error(str(error))
    st.markdown(f'[Open the official data source]({DATASET})')
    st.stop()

with st.sidebar:
    st.header('Analysis settings')
    point_names = sorted(raw['Control Point'].unique())
    selected = st.multiselect('Control points', point_names, default=point_names)
    direction = st.selectbox('Journey direction', ['Both', 'Arrival', 'Departure'])
    horizon = st.slider('Forecast horizon (days)', 1, 14, 3)
    chosen = st.selectbox('Forecasting model', ['Seasonal Naive', 'Random Forest', 'LSTM'], index=2)
    st.caption(f'Source: {origin}')
    st.markdown(f'[Official data portal]({DATASET})')

if not selected:
    st.warning('Select at least one control point.')
    st.stop()
subset = raw.loc[raw['Control Point'].isin(selected)].copy()
if direction != 'Both':
    subset = subset.loc[subset['Arrival / Departure'].str.casefold() == direction.casefold()]
series = subset.groupby('date')['total'].sum().sort_index().astype(float)
if series.empty:
    st.error('No data matches these filters.')
    st.stop()
expected = pd.date_range(series.index.min(), series.index.max(), freq='D')
missing = expected.difference(series.index)
if len(missing):
    st.warning(f'{len(missing)} missing daily dates detected. Analysis uses the contiguous section after the latest gap.')
    series = series.loc[series.index > missing.max()]
if len(series) < 120:
    st.error('This three-model demonstration needs 120 consecutive daily observations. Select more control points or a longer data period.')
    st.stop()

st.caption(f'Coverage: {series.index.min():%Y-%m-%d} – {series.index.max():%Y-%m-%d} · {len(series):,} daily observations · Source updated through {series.index.max():%Y-%m-%d}')
a, b, c = st.columns(3)
a.metric('Latest daily trips', f'{series.iloc[-1]:,.0f}')
b.metric('Last 30 days mean', f'{series.tail(30).mean():,.0f}')
c.metric('Total daily observations', f'{len(series):,}')

st.subheader('01 · Historical pattern')
history_chart = px.line(x=series.tail(120).index, y=series.tail(120).values,
                        labels={'x': 'Date', 'y': 'Passenger trips'}, title='Recent 120 days')
history_chart.update_traces(line_color='#30A7E8')
st.plotly_chart(history_chart, use_container_width=True)

try:
    summary, backtest = cached_evaluation(series)
    if chosen == 'LSTM':
        future = cached_lstm_future(series, horizon)
    else:
        future = predict_future(series, horizon, chosen)
except Exception as error:
    st.error(f'Model evaluation or training failed: {type(error).__name__}: {error}')
    st.stop()

st.subheader('02 · Forecast')
st.caption(f'Current model: **{chosen}** · Forecast begins on the day after the last available observation. LSTM recursively predicts multiple future days.')
forecast_chart = go.Figure()
forecast_chart.add_trace(go.Scatter(x=series.tail(90).index, y=series.tail(90).values, name='Observed', mode='lines'))
forecast_chart.add_trace(go.Scatter(x=future['Date'], y=future['Predicted trips'], name=f'{chosen} forecast', mode='lines+markers', line=dict(dash='dash')))
forecast_chart.update_layout(xaxis_title='Date', yaxis_title='Daily passenger trips', hovermode='x unified')
st.plotly_chart(forecast_chart, use_container_width=True)
st.dataframe(future.assign(Date=future['Date'].dt.strftime('%Y-%m-%d')), hide_index=True, use_container_width=True)

st.subheader('03 · Model comparison — chronological holdout')
st.caption('All three models evaluated on the same final 30 observed days. Model training stops before the test interval. Each test-day LSTM prediction uses observed values only up to the previous day (rolling one-step ahead); future forecasts use recursive predictions. MAE and RMSE are in passenger trips.')
model_bar = px.bar(summary.melt(id_vars='Model', value_vars=['MAE','RMSE'], var_name='Metric', value_name='Error'), x='Model', y='Error', color='Metric', barmode='group', text_auto=',.0f')
model_bar.update_layout(yaxis_title='Absolute error (passenger trips)', xaxis_title='')
st.plotly_chart(model_bar, use_container_width=True)
model_table = summary.copy()
model_table['MAE'] = model_table['MAE'].round(0).astype(int)
model_table['RMSE'] = model_table['RMSE'].round(0).astype(int)
st.dataframe(model_table, use_container_width=True, hide_index=True)

st.subheader('04 · Actual vs predicted — test data')
comparison_chart = go.Figure()
for col in backtest.columns:
    comparison_chart.add_trace(go.Scatter(x=backtest.index, y=backtest[col], mode='lines', name=col))
comparison_chart.update_layout(xaxis_title='Date', yaxis_title='Daily trips', hovermode='x unified')
st.plotly_chart(comparison_chart, use_container_width=True)

st.subheader('05 · Research interpretation and next steps')
leader = summary.iloc[0]
seasonal_mae = float(summary.loc[summary.Model == 'Seasonal Naive', 'MAE'].iloc[0])
relative = 100 * (seasonal_mae - float(leader.MAE)) / seasonal_mae if seasonal_mae else 0
st.markdown(f'''**Key finding.** On the common held-out 30-day window, **{leader.Model}** has the lowest MAE (**{leader.MAE:,.0f} trips/day**), a **{relative:.1f}%** change relative to the same-weekday baseline. This result describes **this selected dataset and test period only**; it does not establish superiority in other seasons or stations.

**What has been demonstrated.** We can ingest public operational-style data, aggregate daily passenger volumes, develop classical and deep learning forecasting methods, and evaluate them without fitting on the test period.

**Limitations.** Immigration-clearance trips are not metro station arrivals, occupancy or crowding. This system is daily rather than real time; weather, holidays, exceptional events and train headways are omitted. The LSTM uses one-dimensional history only. A 30-day test window is insufficient for production reliability claims.

**Future extension to the MTR dissertation.** Subject to obtaining authorized MTR datasets: (1) replace daily clearance volumes with station-level 5–15 minute entries, exits and platform counts; (2) add train headway, service disruptions, event and weather features; (3) compare rolling-origin forecasting accuracy and peak-hour robustness; (4) simulate capacity / dispatch alternatives; (5) evaluate crowding reduction and passenger wait times against a no-intervention baseline.''')
st.info('Academic demonstration only. Forecasts are estimated clearance trips and should not be interpreted as MTR congestion predictions.')
st.markdown(f'[Dataset and methodology]({DATASET}) · [GitHub source](https://github.com/exefyx/hk-passenger-flow-prediction)')
