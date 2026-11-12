"""Public-data proof-of-concept; this is not an MTR station occupancy model."""
from datetime import date, timedelta
from pathlib import Path
import io
import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

SOURCE = 'https://www.immd.gov.hk/opendata/eng/transport/immigration_clearance/statistics_on_daily_passenger_traffic.csv'
LOCAL = Path(__file__).parent / 'data' / 'passenger_traffic.csv'
st.set_page_config(page_title='Hong Kong Passenger Flow | Forecast Demo', page_icon='🚉', layout='wide')
st.title('Hong Kong Passenger Flow Forecast')
st.caption('Research proof of concept · Hong Kong Immigration Department public data · Cross-border passenger trips, NOT MTR station entries')

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    try:
        resp = requests.get(SOURCE, timeout=25, headers={'User-Agent': 'Mozilla/5.0 PassengerFlowResearchDemo/1.0'})
        resp.raise_for_status()
        raw = resp.content
        origin = 'Official live CSV'
    except requests.RequestException as error:
        if not LOCAL.exists():
            raise RuntimeError(f'Official CSV unavailable ({error}). Upload the CSV to data/passenger_traffic.csv to enable the offline fallback.') from error
        raw = LOCAL.read_bytes()
        origin = 'Local CSV fallback (check its freshness)'
    data = None
    for encoding in ('utf-8-sig', 'utf-8', 'big5'):
        try:
            data = pd.read_csv(io.BytesIO(raw), encoding=encoding)
            break
        except UnicodeDecodeError:
            pass
    if data is None:
        raise ValueError('Unable to decode CSV')
    data.columns = data.columns.str.replace('\ufeff','',regex=False).str.strip()
    required = {'Date', 'Control Point', 'Arrival / Departure', 'Total'}
    if not required.issubset(data.columns):
        raise ValueError(f'Unexpected CSV schema: {list(data.columns)}; expected {sorted(required)}')
    data['date'] = pd.to_datetime(data['Date'].astype(str).str.strip(), format='%d-%m-%Y', errors='coerce')
    data['total'] = pd.to_numeric(data['Total'].astype(str).str.replace(',', '', regex=False), errors='coerce')
    data = data.dropna(subset=['date', 'total'])
    data['Control Point'] = data['Control Point'].astype(str).str.strip()
    data['Arrival / Departure'] = data['Arrival / Departure'].astype(str).str.strip()
    # Guard against duplicate date / control-point / direction rows
    data = data.drop_duplicates(subset=['date', 'Control Point', 'Arrival / Departure'])
    return data, origin

def features(dates):
    d = pd.DatetimeIndex(dates)
    return pd.DataFrame({'dow': d.dayofweek, 'month': d.month, 'day': d.day, 'week': d.isocalendar().week.astype(int).to_numpy(),
                         'is_weekend': (d.dayofweek >= 5).astype(int), 'time_index': ((d-pd.Timestamp('2021-01-01')).days).to_numpy()}, index=d)

def seasonal_forecast(series, dates):
    result = []
    for d in dates:
        prior = d - pd.Timedelta(days=7)
        if prior in series.index:
            result.append(float(series.loc[prior]))
        else:
            prev = series[series.index < d]
            result.append(float(prev.tail(7).median()))
    return np.array(result)

def random_forest_forecast(series, dates):
    model = RandomForestRegressor(n_estimators=250, min_samples_leaf=3, max_features=0.8, random_state=42, n_jobs=-1)
    model.fit(features(series.index), series.values)
    return np.maximum(0, model.predict(features(dates)))

try:
    source_data, origin = load_data()
except (RuntimeError, ValueError, pd.errors.ParserError) as error:
    st.error(str(error))
    st.info('Download the official English CSV via the link below and place it at data/passenger_traffic.csv. Then restart.')
    st.link_button('Official dataset', 'https://data.gov.hk/en-data/dataset/hk-immd-set5-statistics-daily-passenger-traffic')
    st.stop()

points = sorted(source_data['Control Point'].unique())
with st.sidebar:
    st.header('Filters')
    selected_points = st.multiselect('Control points', points, default=points)
    direction = st.selectbox('Journey direction', ['Both', 'Arrival', 'Departure'])
    forecast_days = st.slider('Days to forecast', 1, 14, 3)
    model_choice = st.selectbox('Forecast model', ['Seasonal naive (same weekday)', 'Random forest (calendar features)'])
    st.markdown('[Official data source](https://data.gov.hk/en-data/dataset/hk-immd-set5-statistics-daily-passenger-traffic)')
    st.caption(f'Data origin: {origin}')

if not selected_points:
    st.warning('Select at least one control point.')
    st.stop()
filtered = source_data[source_data['Control Point'].isin(selected_points)].copy()
if direction != 'Both':
    filtered = filtered[filtered['Arrival / Departure'].str.casefold() == direction.casefold()]
series = filtered.groupby('date')['total'].sum().sort_index().astype(float)
if series.empty:
    st.error('No matching rows for your filters.')
    st.stop()
# Use only observed dates, and warn about missing days; do not interpret missing rows as zero passengers.
full_days = pd.date_range(series.index.min(), series.index.max(), freq='D')
missing = full_days.difference(series.index)
if len(missing):
    st.warning(f'{len(missing)} missing dates found. Using only consecutive days after the latest gap for evaluation.')
    series = series.loc[missing.max() + pd.Timedelta(days=1):]
if len(series) < 21:
    st.error('At least 21 daily observations are required for this reproducible demo.')
    st.stop()

st.caption(f'Coverage: {series.index.min():%Y-%m-%d} to {series.index.max():%Y-%m-%d} · {len(series):,} days · updated: {series.index.max():%Y-%m-%d}')
col1, col2, col3 = st.columns(3)
col1.metric('Latest daily trips', f'{series.iloc[-1]:,.0f}')
col2.metric('Average daily trips (last 30)', f'{series.tail(30).mean():,.0f}')
col3.metric('Historical observations', f'{len(series):,}')

# Held-out chronological backtest. Never train on the test period.
test_days = min(30, max(7, len(series)//5))
train = series.iloc[:-test_days]
test = series.iloc[-test_days:]
if model_choice.startswith('Random'):
    test_pred = random_forest_forecast(train, test.index)
else:
    test_pred = seasonal_forecast(train, test.index)
mae = mean_absolute_error(test, test_pred)
rmse = float(np.sqrt(mean_squared_error(test, test_pred)))
base_pred = seasonal_forecast(train, test.index)
base_mae = mean_absolute_error(test, base_pred)

future_dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
if model_choice.startswith('Random'):
    forecast = random_forest_forecast(series, future_dates)
else:
    forecast = seasonal_forecast(series, future_dates)

st.subheader('Historical traffic and future forecast')
fig = go.Figure()
fig.add_trace(go.Scatter(x=series.tail(90).index, y=series.tail(90).values, mode='lines', name='Observed'))
fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines+markers', name='Forecast', line=dict(dash='dash')))
fig.update_layout(xaxis_title='Date', yaxis_title='Daily passenger trips', hovermode='x unified', legend_title='')
st.plotly_chart(fig, use_container_width=True)

st.subheader('Next days forecast')
st.dataframe(pd.DataFrame({'Date': future_dates.date.astype(str), 'Predicted passenger trips': np.round(forecast).astype(int)}), hide_index=True, use_container_width=True)

st.subheader('Out-of-sample evaluation')
a,b,c = st.columns(3)
a.metric('Test MAE', f'{mae:,.0f}')
b.metric('Test RMSE', f'{rmse:,.0f}')
c.metric('Seasonal baseline MAE', f'{base_mae:,.0f}')
st.caption(f'Last {test_days} observed days are reserved for testing. Random Forest uses only calendar features (weekday/month/day/week/time index), not future passenger data. Scores are not causal claims or guarantees of future performance.')
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=test.index, y=test.values, mode='lines+markers', name='Observed'))
fig2.add_trace(go.Scatter(x=test.index, y=test_pred, mode='lines+markers', name='Model prediction'))
fig2.update_layout(xaxis_title='Held-out date', yaxis_title='Daily passenger trips', hovermode='x unified')
st.plotly_chart(fig2, use_container_width=True)

st.info('Scope limitation: This dashboard predicts cross-border clearance trips, not platform occupancy, MTR train headway, or crowding. The Random Forest is a simple calendar-feature model; advanced sequence models require further data/model validation.')
