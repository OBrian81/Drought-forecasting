#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import date, timedelta

# Set page config
st.set_page_config(
    page_title="Drought Forecasting Dashboard",
    page_icon="🌵",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    .stAlert {padding: 20px;}
    .st-b7 {color: #ff4b4b;}
    .css-18e3th9 {padding: 2rem 5rem;}
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("🌵 Machakos County Drought Forecasting")
st.markdown("""
This dashboard predicts drought conditions using Prophet time series forecasting
combined with LSTM neural networks. The model analyzes precipitation, temperature,
humidity, and wind speed data to forecast the Standardized Precipitation Index (SPI).
""")

# Sidebar for user inputs
with st.sidebar:
    st.header("Model Configuration")
    st.image("https://i.imgur.com/JZxQt1m.png", width=200)

    # Model selection
    model_type = st.radio(
        "Select Model Type",
        ["Prophet Only", "Hybrid Prophet-LSTM"],
        help="Choose between Prophet or the hybrid model"
    )

    # Forecast horizon
    forecast_days = st.slider(
        "Forecast Horizon (days)",
        min_value=30,
        max_value=365,
        value=90,
        help="How many days ahead to forecast"
    )

    # Threshold adjustment
    drought_threshold = st.slider(
        "Drought Threshold (SPI)",
        min_value=-2.0,
        max_value=0.0,
        value=-1.0,
        step=0.1,
        help="SPI value below which drought is declared"
    )

    # Data upload option
    uploaded_file = st.file_uploader(
        "Upload your climate data (CSV)",
        type=["csv"],
        help="Should contain columns: date, precip, temp, humidity, windspeed"
    )

# Load sample data if no file uploaded
@st.cache_data
def load_sample_data():
    # This would be your actual sample data path
    return pd.read_csv("sample_drought_data.csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_sample_data()
    st.info("Using sample data. Upload your own CSV file to customize.")

# Data preprocessing function
def preprocess_data(df):
    # Convert date column and set as index
    df['ds'] = pd.to_datetime(df['date'])
    df.set_index('ds', inplace=True)

    # Rename columns for Prophet
    df.rename(columns={'precip': 'y'}, inplace=True)

    # Handle missing values
    for col in ['y', 'temp', 'humidity', 'windspeed']:
        df[col].interpolate(method='time', inplace=True)

    # Calculate SPI
    df['spi_3'] = calculate_spi(df['y'], scale=3)

    return df[['ds', 'spi_3', 'temp', 'humidity', 'windspeed']].dropna()

# SPI calculation function

def calculate_spi(precip_series, scale=3):
    from scipy import stats
    if precip_series.dropna().shape[0] < scale:
        return pd.Series([np.nan] * len(precip_series), index=precip_series.index)

    rolling_sum = precip_series.rolling(window=scale, min_periods=scale).sum()
    spi_values = []

    for i in range(len(rolling_sum)):
        if i < scale - 1:
            spi_values.append(np.nan)
        else:
            hist_data = rolling_sum[:i+1].dropna()
            hist_data = hist_data[hist_data > 0]  # Filter out zeros/negatives

            if len(hist_data) < 5:
                spi_values.append(np.nan)
                continue

            try:
                params = stats.gamma.fit(hist_data)
                cdf = stats.gamma.cdf(rolling_sum.iloc[i], *params)
                spi = stats.norm.ppf(cdf)
                spi_values.append(spi)
            except Exception:
                spi_values.append(np.nan)

    return pd.Series(spi_values, index=precip_series.index)


