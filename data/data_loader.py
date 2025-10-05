# data/data_loader.py

import os
import pickle
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
from datetime import datetime

def setup_and_load_data(config):
    """
    Handles data downloading based on specific start/end dates from the config,
    preprocessing, and caching.
    """
    # Create a unique cache filename based on dates and ticker
    start_date_str = config['TRAIN_START_DATE']
    end_date_str = config['TRAIN_END_DATE']
    cache_path = os.path.join(config['CACHE_DIR'], f"data_cache_{config['STOCK_TICKER']}_{start_date_str}_to_{end_date_str}.pkl")
    
    if os.path.exists(cache_path):
        print(f"--- Loading cached data from {cache_path} ---")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print("--- Stage 1: Setting up and loading new data based on paper's date range ---")
    
    # Download all available data up to today to ensure we have recent data for testing
    print(f"Downloading all available data for {config['STOCK_TICKER']} from {start_date_str}...")
    full_data_raw = yf.download(config['STOCK_TICKER'], start=start_date_str, progress=False)
    
    if full_data_raw.empty:
        print("\nERROR: No data was downloaded.")
        print("Please check your network connection and the ticker symbol.")
        sys.exit(1)

    full_data_raw['log_return'] = np.log(full_data_raw['Close'] / full_data_raw['Close'].shift(1))
    full_data_raw.dropna(inplace=True)

    # Split data based on the paper's training end date
    training_end_date = pd.to_datetime(end_date_str)
    
    # All data up to the training end date is considered historical
    historical_data = full_data_raw[full_data_raw.index <= training_end_date].copy()
    
    # The most recent data is used for the walk-forward test set
    # We take the last (lookback + horizon) days of the entire dataset for this
    if len(full_data_raw) < config['LOOKBACK_WINDOW'] + config['FORECAST_HORIZON']:
        print("\nERROR: Not enough total data available to create a test set.")
        sys.exit(1)

    # The current window and true future are always the most recent available data
    true_future_data = full_data_raw.iloc[-config['FORECAST_HORIZON']:].copy()
    current_window_data = full_data_raw.iloc[-config['LOOKBACK_WINDOW']-config['FORECAST_HORIZON']:-config['FORECAST_HORIZON']].copy()

    # Fit the scaler ONLY on the historical training data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    historical_data['scaled_return'] = scaler.fit_transform(historical_data[['log_return']])
    # Transform the current window using the same fitted scaler
    current_window_data['scaled_return'] = scaler.transform(current_window_data[['log_return']])

    print(f"\nData is ready. Training data from {historical_data.index.min().date()} to {historical_data.index.max().date()}.")
    data_bundle = {
        "full_data": full_data_raw,
        "historical_data": historical_data,
        "current_window": current_window_data,
        "true_future": true_future_data,
        "scaler": scaler
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(data_bundle, f)
    print(f"Data loading complete and results cached to {cache_path}.")
    
    return data_bundle

