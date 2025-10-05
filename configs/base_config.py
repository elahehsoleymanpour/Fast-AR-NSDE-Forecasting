# configs/base_config.py
"""
Central configuration file for replicating the paper's experiments.
Hyperparameters are set according to the paper's Section 4.1 and Table 1.
"""
import torch

CONFIG = {
    # --- Data and General Settings (As per paper's Section 4.1) ---
    "STOCK_TICKER": 'SPY', # Using SPY as a representative asset
    "TRAIN_START_DATE": "2010-01-01",
    "TRAIN_END_DATE": "2022-12-31",
    "LOOKBACK_WINDOW": 30,      # A reasonable choice, kept from previous runs
    "FORECAST_HORIZON": 10,     # A reasonable choice, kept from previous runs
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "SEED": 42,
    "CACHE_DIR": "cache",
    "RESULTS_DIR": "results",

    # --- F-ACE Model Settings (As per paper's Table 1) ---
    "K_ANALOGUES_FACE": 10,
    "MIN_ANALOGUE_DISTANCE_DAYS": 30,

    # --- Clustering and AR-NSDE Settings (As per paper's Table 1) ---
    "K_CLUSTERS": 15,           # Using a reasonable value based on previous experiments
    "K_ANALOGUES_MMD": 15,      # Using a reasonable value
    "LAMBDA_MMD": 0.05,         # As per paper's Table 1

    # --- Training Hyperparameters (As per paper's Table 1) ---
    "EPOCHS": 40,               # A sufficient number for convergence
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 5e-5,      # As per paper's Table 1

    # --- Model Architectures (As per paper's Table 1) ---
    "SDE_HIDDEN_DIM": 64,       # As per paper's Table 1 (3 layers with 64 units)
    "LSTM_HIDDEN_DIM": 50,      # As per paper's Table 1 (2 layers with 50 units)
    "LSTM_NUM_LAYERS": 2,

    # --- Forecasting Settings ---
    "ENSEMBLE_SIZE": 50,
}

