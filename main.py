# main.py

import os
import torch
import torch.nn as nn
import numpy as np
import warnings

# --- Import project modules ---
from configs.base_config import CONFIG
from data.data_loader import setup_and_load_data
from analysis.f_ace import run_face_benchmark
from analysis.clustering import get_market_regime_codebook
from models.base_models import NeuralSDE, LSTMBenchmark, create_pytorch_windows
from models.ar_nsde import get_or_train_fast_ar_nsde
from utils.plotting import generate_final_results_and_plot

def get_or_train_model(config, model_class, model_args, model_name, training_df, is_nsde=False):
    """
    Manages the training and caching of standard models (LSTM, Standard NSDE).
    """
    model_path = os.path.join(config['CACHE_DIR'], f"{model_name.lower().replace(' ', '_')}_model_{config['STOCK_TICKER']}_{config['DATA_START_YEARS_AGO']}y.pth")
    model = model_class(**model_args)
    
    if os.path.exists(model_path):
        print(f"\n--- Loading cached {model_name} model ---")
        model.load_state_dict(torch.load(model_path, map_location=config['DEVICE']))
        model.to(config['DEVICE'])
        return model

    print(f"\n--- Stage 3: Training {model_name} model ---")
    device = config['DEVICE']; model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE']); criterion = nn.MSELoss()
    
    X_train, y_train = create_pytorch_windows(training_df['scaled_return'].values, config['LOOKBACK_WINDOW'], config['FORECAST_HORIZON'])
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).unsqueeze(-1), torch.from_numpy(y_train))
    loader = torch.utils.data.DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
    
    model.train()
    for epoch in range(config['EPOCHS']):
        # Using tqdm from the main script
        from tqdm import tqdm
        pbar = tqdm(loader, desc=f"{model_name} Epoch {epoch+1}/{config['EPOCHS']}", leave=False)
        for x_batch, y_batch in pbar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device); optimizer.zero_grad()
            if is_nsde:
                pred = torch.median(torch.stack([model(x_batch) for _ in range(10)], dim=0), dim=0).values.squeeze(-1)
            else:
                pred = model(x_batch)
            loss = criterion(pred, y_batch); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
    torch.save(model.state_dict(), model_path)
    print(f"{model_name} training complete and model cached.")
    return model

# =============================================================================
# Main Execution Block
# =============================================================================
if __name__ == '__main__':
    # Initial setup
    warnings.filterwarnings('ignore')
    np.random.seed(CONFIG['SEED'])
    torch.manual_seed(CONFIG['SEED'])
    if not os.path.exists(CONFIG['CACHE_DIR']):
        os.makedirs(CONFIG['CACHE_DIR'])

    # --- STAGE 1: Data Loading ---
    data = setup_and_load_data(CONFIG)

    # --- STAGE 2: F-ACE Benchmark ---
    face_returns = run_face_benchmark(CONFIG, data['historical_data'], data['current_window'])

    # --- STAGE 3: Train Standard Models ---
    nsde_args = {'lookback': CONFIG['LOOKBACK_WINDOW'], 'horizon': CONFIG['FORECAST_HORIZON'], 'sde_hidden': CONFIG['SDE_HIDDEN_DIM'], 'device': CONFIG['DEVICE']}
    std_nsde_model = get_or_train_model(CONFIG, NeuralSDE, nsde_args, "Standard NSDE", data['historical_data'], is_nsde=True)
    
    lstm_args = {'lookback': CONFIG['LOOKBACK_WINDOW'], 'horizon': CONFIG['FORECAST_HORIZON'], 'hidden_dim': CONFIG['LSTM_HIDDEN_DIM'], 'num_layers': CONFIG['LSTM_NUM_LAYERS']}
    lstm_model = get_or_train_model(CONFIG, LSTMBenchmark, lstm_args, "LSTM", data['historical_data'])
    
    # --- STAGE 4: Get Market Regimes ---
    codebook = get_market_regime_codebook(CONFIG, data['historical_data'])
    
    # --- STAGE 5: Train Fast AR-NSDE ---
    fast_ar_nsde_model = get_or_train_fast_ar_nsde(CONFIG, data['historical_data'], codebook)

    # --- STAGE 6: Final Results and Visualization ---
    models_to_evaluate = {
        "Standard NSDE": std_nsde_model,
        "Fast AR-NSDE": fast_ar_nsde_model,
        "LSTM": lstm_model
    }
    generate_final_results_and_plot(CONFIG, models_to_evaluate, face_returns, data)

    print("\nProject execution finished successfully.")

