# models/base_models.py

import torch
import torch.nn as nn
import torchsde
import numpy as np

def create_pytorch_windows(data_series, lookback, horizon):
    """Utility to create input/output windows for PyTorch models."""
    X, y = [], []
    for i in range(len(data_series) - lookback - horizon + 1):
        X.append(data_series[i:(i + lookback)])
        y.append(data_series[(i + lookback):(i + lookback + horizon)])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

class SDE(nn.Module):
    """Defines the core SDE with drift and diffusion networks."""
    noise_type = "general"
    sde_type = "ito"
    
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.features = None
        self.f_net = nn.Sequential(nn.Linear(feature_dim + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.g_net = nn.Sequential(nn.Linear(feature_dim + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def f(self, t, y):
        sde_input = torch.cat([y, self.features], dim=1)
        return self.f_net(sde_input)

    def g(self, t, y):
        sde_input = torch.cat([y, self.features], dim=1)
        return self.g_net(sde_input).unsqueeze(-1)

class NeuralSDE(nn.Module):
    """A wrapper for the SDE class to handle multi-step forecasting."""
    def __init__(self, lookback, horizon, sde_hidden, device):
        super().__init__()
        self.sde = SDE(feature_dim=lookback, hidden_dim=sde_hidden)
        self.ts = torch.linspace(0, 1, 2).to(device)
        self.horizon = horizon

    def forward(self, x_seq):
        batch_size = x_seq.size(0)
        current_sequence = x_seq.clone()
        predictions = []
        for _ in range(self.horizon):
            features = current_sequence.view(batch_size, -1)
            y0 = current_sequence[:, -1, :]
            self.sde.features = features
            solution = torchsde.sdeint(self.sde, y0, self.ts, method='euler', dt=1.0)
            next_pred = solution[-1]
            predictions.append(next_pred)
            current_sequence = torch.cat([current_sequence[:, 1:, :], next_pred.unsqueeze(1)], dim=1)
        self.sde.features = None
        return torch.stack(predictions, dim=1)

class LSTMBenchmark(nn.Module):
    """A simple two-layer LSTM benchmark model."""
    def __init__(self, lookback, horizon, hidden_dim, num_layers):
        super().__init__()
        self.lstm1 = nn.LSTM(1, hidden_dim, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return self.linear(x[:, -1, :])
