# models/ar_nsde.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.neighbors import KernelDensity

from .base_models import NeuralSDE, create_pytorch_windows

def get_or_train_fast_ar_nsde(config, training_df, codebook):
    """
    Manages the training and caching of the Fast AR-NSDE model.
    """
    # Create cache path using the start and end dates from config
    start_date = config['TRAIN_START_DATE'].replace('-', '')
    end_date = config['TRAIN_END_DATE'].replace('-', '')
    model_path = os.path.join(config['CACHE_DIR'], f"fast_ar_nsde_model_{config['STOCK_TICKER']}_{start_date}_to_{end_date}.pth")
    model = NeuralSDE(config['LOOKBACK_WINDOW'], config['FORECAST_HORIZON'], config['SDE_HIDDEN_DIM'], config['DEVICE'])
    
    if os.path.exists(model_path):
        print(f"\n--- Loading cached Fast AR-NSDE model ---")
        model.load_state_dict(torch.load(model_path, map_location=config['DEVICE']))
        model.to(config['DEVICE'])
        return model

    print("\n--- Stage 5: Training Fast AR-NSDE model ---")
    device = config['DEVICE']; model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE']); criterion = nn.MSELoss()
    
    X_train, y_train = create_pytorch_windows(training_df['scaled_return'].values, config['LOOKBACK_WINDOW'], config['FORECAST_HORIZON'])
    train_indices = torch.arange(len(X_train)); dataset = TensorDataset(train_indices, torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
    
    grid = np.linspace(np.min(X_train), np.max(X_train), 500).reshape(-1, 1)
    bandwidth = 1.06 * np.std(X_train) * (len(X_train))**(-1/5)
    
    def fit_and_score_pdf(win):
        pdf = np.exp(KernelDensity(bandwidth=bandwidth).fit(win.reshape(-1, 1)).score_samples(grid))
        return pdf / pdf.sum()

    train_pdfs = np.array(Parallel(n_jobs=-1)(delayed(fit_and_score_pdf)(win) for win in tqdm(X_train, desc="Pre-calculating PDFs")))
    
    def compute_mmd_loss(s, t, sigma=0.1):
        s, t = s.contiguous().view(s.size(0), -1), t.contiguous().view(t.size(0), -1)
        rbf = lambda x1, x2: torch.exp(-torch.cdist(x1, x2, p=2)**2 / (2 * sigma**2))
        return (rbf(s,s).mean() if s.size(0)>1 else 1.0) + (rbf(t,t).mean() if t.size(0)>1 else 1.0) - 2*rbf(s,t).mean()
    
    model.train()
    for epoch in range(config['EPOCHS']):
        pbar = tqdm(loader, desc=f"AR-NSDE Epoch {epoch+1}/{config['EPOCHS']}", leave=False)
        for x_indices, y_true in pbar:
            y_true = y_true.to(device); optimizer.zero_grad()
            labels = np.argmin(np.sum((train_pdfs[x_indices.numpy()][:, np.newaxis, :] - codebook['centroids'])**2, axis=2), axis=1)
            initial_sequences = torch.from_numpy(X_train[x_indices.numpy()]).unsqueeze(-1).to(device)
            ensemble = torch.stack([model(initial_sequences) for _ in range(config['K_ANALOGUES_MMD'])], dim=1)
            median = torch.median(ensemble, dim=1).values.squeeze(-1)
            nll = criterion(median, y_true); mmd = 0.0
            for i, label in enumerate(labels):
                emp_futures = torch.from_numpy(codebook['futures'][label]).to(device)
                if len(emp_futures) > 0:
                    s_idx = torch.randint(0, emp_futures.shape[0], (config['K_ANALOGUES_MMD'],))
                    mmd += compute_mmd_loss(ensemble[i].squeeze(-1), emp_futures[s_idx])
            loss = nll + config['LAMBDA_MMD'] * (mmd / len(labels)); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
    torch.save(model.state_dict(), model_path)
    print("Fast AR-NSDE training complete and model cached.")
    return model
