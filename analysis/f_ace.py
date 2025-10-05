# analysis/f_ace.py

import os
import pickle
import numpy as np
from sklearn.neighbors import KernelDensity
from joblib import Parallel, delayed
from tqdm import tqdm

def run_face_benchmark(config, historical_data, current_window_data):
    """
    Executes the F-ACE benchmark model and caches its results.
    """
    # Create cache path using the start and end dates from config
    start_date = config['TRAIN_START_DATE'].replace('-', '')
    end_date = config['TRAIN_END_DATE'].replace('-', '')
    cache_path = os.path.join(config['CACHE_DIR'], f"face_cache_{config['STOCK_TICKER']}_{start_date}_to_{end_date}.pkl")
    if os.path.exists(cache_path):
        print("\n--- Stage 2: Loading cached F-ACE results ---")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print("\n--- Stage 2: Running F-ACE Benchmark ---")
    
    def create_windows_face(data_series, lookback):
        X, indices = [], []
        for i in range(len(data_series) - lookback + 1):
            X.append(data_series.iloc[i:(i + lookback)].values)
            indices.append(data_series.index[i + lookback - 1])
        return np.array(X), np.array(indices)

    def hellinger_distance(p, q):
        epsilon = 1e-10; p += epsilon; q += epsilon; p /= p.sum(); q /= q.sum()
        return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q))**2)) / np.sqrt(2)

    def find_top_analogues_face(current_window_returns, historical_returns, k, lookback, min_distance_days=0):
        X_hist, end_indices_hist = create_windows_face(historical_returns, lookback)
        kde_bw = 1.06 * np.std(X_hist) * (X_hist.shape[0])**(-1/5)
        kde_current = KernelDensity(kernel='gaussian', bandwidth=kde_bw).fit(current_window_returns.reshape(-1, 1))
        grid = np.linspace(np.min(X_hist), np.max(X_hist), 1000).reshape(-1, 1)
        pdf_current = np.exp(kde_current.score_samples(grid)); pdf_current /= pdf_current.sum()

        def fit_and_score_kde(win):
            pdf_win = np.exp(KernelDensity(bandwidth=kde_bw).fit(win.reshape(-1, 1)).score_samples(grid))
            return pdf_win / pdf_win.sum()

        pdfs_hist = Parallel(n_jobs=-1)(delayed(fit_and_score_kde)(win) for win in tqdm(X_hist, desc="F-ACE KDEs"))
        distances = np.array([hellinger_distance(pdf_current, pdf_hist) for pdf_hist in pdfs_hist])
        
        sorted_indices = np.argsort(distances)
        selected_end_dates, selected_distances = [], []
        for idx in sorted_indices:
            end_date = end_indices_hist[idx]
            if all(abs((end_date - sel_date).days) >= min_distance_days for sel_date in selected_end_dates):
                selected_end_dates.append(end_date)
                selected_distances.append(distances[idx])
                if len(selected_end_dates) >= k: break
        return {"analogue_end_dates": np.array(selected_end_dates), "distances": np.array(selected_distances)}

    analogue_results = find_top_analogues_face(current_window_data['log_return'].values, historical_data['log_return'], k=config['K_ANALOGUES_FACE'], lookback=config['LOOKBACK_WINDOW'], min_distance_days=config['MIN_ANALOGUE_DISTANCE_DAYS'])

    analogue_futures = []
    valid_distances = []
    for i, end_date in enumerate(analogue_results['analogue_end_dates']):
        start_loc = historical_data.index.get_loc(end_date) + 1
        future = historical_data['log_return'].iloc[start_loc : start_loc + config['FORECAST_HORIZON']]
        if len(future) == config['FORECAST_HORIZON']:
            analogue_futures.append(future.values)
            valid_distances.append(analogue_results['distances'][i])

    if len(analogue_futures) > 0:
        weights = 1 / (np.array(valid_distances) + 1e-8)
        weights /= np.sum(weights)
        face_forecast_returns = np.sum(np.array(analogue_futures) * weights[:, np.newaxis], axis=0)
    else:
        face_forecast_returns = np.zeros(config['FORECAST_HORIZON'])
    
    with open(cache_path, 'wb') as f:
        pickle.dump(face_forecast_returns, f)
    print("F-ACE benchmark complete and results cached.")
    return face_forecast_returns
