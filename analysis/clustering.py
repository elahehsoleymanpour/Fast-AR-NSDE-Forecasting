# analysis/clustering.py

import os
import pickle
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from tqdm import tqdm
from models.base_models import create_pytorch_windows

def get_market_regime_codebook(config, training_df):
    """
    Performs offline clustering to identify market regimes and caches the resulting codebook.
    """
    # Create cache path using the start and end dates from config
    start_date = config['TRAIN_START_DATE'].replace('-', '')
    end_date = config['TRAIN_END_DATE'].replace('-', '')
    cache_path = os.path.join(config['CACHE_DIR'], f"codebook_{config['STOCK_TICKER']}_{start_date}_to_{end_date}_k{config['K_CLUSTERS']}.pkl")
    if os.path.exists(cache_path):
        print("\n--- Stage 4: Loading cached market regime codebook ---")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print("\n--- Stage 4: Performing offline clustering for market regimes ---")
    X_hist, y_hist = create_pytorch_windows(training_df['scaled_return'].values, config['LOOKBACK_WINDOW'], config['FORECAST_HORIZON'])
    kde_bw = 1.06 * np.std(X_hist) * (len(X_hist))**(-1/5)
    grid = np.linspace(np.min(X_hist), np.max(X_hist), 500).reshape(-1, 1)

    def fit_and_score(win):
        pdf = np.exp(KernelDensity(bandwidth=kde_bw).fit(win.reshape(-1, 1)).score_samples(grid))
        return pdf / pdf.sum()
        
    pdfs_hist = np.array(Parallel(n_jobs=-1)(delayed(fit_and_score)(win) for win in tqdm(X_hist, desc="Clustering KDEs")))
    
    kmeans = KMeans(n_clusters=config['K_CLUSTERS'], random_state=config['SEED'], n_init=10).fit(pdfs_hist)
    
    codebook = {
        'centroids': kmeans.cluster_centers_,
        'futures': {i: y_hist[np.where(kmeans.labels_ == i)[0]] for i in range(config['K_CLUSTERS'])}
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(codebook, f)
    print("Clustering complete and codebook cached.")
    return codebook
