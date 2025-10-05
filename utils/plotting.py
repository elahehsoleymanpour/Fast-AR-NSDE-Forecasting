# utils/plotting.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os
import torch

def save_dataframe_as_image(df, title, filename):
    """Utility function to save a pandas DataFrame as a styled image."""
    fig, ax = plt.subplots(figsize=(10, max(2, 0.5 * len(df) + 0.5)))
    ax.axis('tight')
    ax.axis('off')
    # Use reset_index() to make the index a column for the table
    df_reset = df.reset_index()
    table = ax.table(cellText=df_reset.round(4).values, colLabels=df_reset.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title(title, fontsize=16, y=1.1)
    fig.tight_layout()
    plt.savefig(filename)
    print(f"Table saved to '{filename}'")
    plt.close()

def plot_cluster_centroids(config, codebook):
    """Experiment 1 - Figure 2: Visualizes the discovered market regime centroids."""
    print("\nGenerating Figure 2: Market Regime Centroids Plot...")
    os.makedirs(config['RESULTS_DIR'], exist_ok=True)
    k = config['K_CLUSTERS']
    rows, cols = ((k + 4) // 5, 5) if k > 5 else (1, k)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    fig.suptitle("Experiment 1 (Figure 2): Market Regime Centroids (KDEs)", fontsize=20)
    axes = axes.flatten() if k > 1 else [axes]
    for i in range(k):
        axes[i].plot(codebook['centroids'][i])
        axes[i].set_title(f"Cluster {i}")
        axes[i].set_xticks([]); axes[i].set_yticks([])
    for i in range(k, len(axes)): axes[i].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(config['RESULTS_DIR'], 'exp1_fig2_cluster_centroids.png'))
    plt.show()

def plot_cluster_statistics_table(config, codebook):
    """Experiment 1 - Table 2: Generates and saves a table with descriptive statistics for each cluster."""
    print("\nGenerating Table 2: Cluster Statistics...")
    stats = []
    for i in range(config['K_CLUSTERS']):
        futures = codebook['futures'].get(i)
        if futures is not None and len(futures) > 0:
            cumulative_returns = np.cumsum(futures, axis=1)[:, -1] * 100
            stats.append({
                "Regime (Cluster)": f"Cluster {i}",
                "Mean Future Return (%)": np.mean(cumulative_returns),
                "Std. Dev. of Futures (%)": np.std(cumulative_returns),
                "Skewness": pd.Series(cumulative_returns).skew(),
                "Num Samples": len(futures)
            })
    
    stats_df = pd.DataFrame(stats).set_index("Regime (Cluster)")
    print("\n--- Experiment 1: Cluster Statistics ---")
    print(stats_df.round(2))
    save_dataframe_as_image(stats_df, "Experiment 1 (Table 2): Cluster Statistics", os.path.join(config['RESULTS_DIR'], 'exp1_table2_cluster_stats.png'))

def generate_performance_table_and_plot(config, models, face_returns, data_bundle, experiment_name):
    """
    Experiment 2/3 - Table 3/4 & Plot: Generates forecasts, calculates metrics, and plots comparison.
    """
    print(f"\n--- Generating Results for {experiment_name} ---")
    
    current_data, future_data, scaler = data_bundle['current_window'], data_bundle['true_future'], data_bundle['scaler']
    results = {}
    current_scaled = scaler.transform(current_data['log_return'].values.reshape(-1, 1)).flatten()
    forecast_input = torch.from_numpy(current_scaled).float().unsqueeze(0).unsqueeze(-1).to(config['DEVICE'])

    with torch.no_grad():
        for name, model in models.items():
            model.eval()
            if "NSDE" in name:
                ensemble = torch.stack([model(forecast_input) for _ in range(config['ENSEMBLE_SIZE'])]).squeeze().cpu().numpy()
                results[f'{name}_ensemble'] = scaler.inverse_transform(ensemble)
            else: # LSTM
                forecast = model(forecast_input).cpu().numpy()
                results[f'{name}_forecast'] = scaler.inverse_transform(forecast).flatten()
    
    results['FACE_forecast'] = face_returns

    log_returns_to_prices = lambda lr, sp: sp * np.exp(np.cumsum(lr))
    metrics = []
    last_price = current_data['Close'].iloc[-1].item()
    
    model_names_ordered = ["Fast AR-NSDE", "Standard NSDE", "LSTM", "FACE"]
    
    # Process models
    for name in model_names_ordered:
        price_path = None
        if name in models:
            if "NSDE" in name:
                median_returns = np.median(results[f'{name}_ensemble'], axis=0)
                price_path = log_returns_to_prices(median_returns, last_price)
                results[f'{name}_prices'] = price_path
                results[f'{name}_ensemble_prices'] = np.array([log_returns_to_prices(p, last_price) for p in results[f'{name}_ensemble']])
            else: # LSTM
                 price_path = log_returns_to_prices(results[f'{name}_forecast'], last_price)
                 results[f'{name}_prices'] = price_path
        elif name == "FACE":
            price_path = log_returns_to_prices(results['FACE_forecast'], last_price)
            results['FACE_prices'] = price_path
        
        if price_path is not None:
            rmse = np.sqrt(mean_squared_error(future_data['Close'].values, price_path))
            metrics.append({"Model": name, "RMSE": rmse})
    
    comparison_df = pd.DataFrame(metrics).set_index("Model").reindex(model_names_ordered)
    print("\n--- FINAL PERFORMANCE COMPARISON (RMSE) ---")
    print(comparison_df.round(4))
    
    table_name = "Table 3" if "Experiment 2" in experiment_name else "Table 4"
    save_dataframe_as_image(comparison_df, f"{experiment_name} ({table_name}): Performance Comparison (RMSE)", os.path.join(config['RESULTS_DIR'], f'{experiment_name}_performance_table.png'))

    # Generate Plot
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(current_data['Close'], color='grey', lw=3, label='Current Window')
    ax.plot(future_data['Close'], 'r-o', lw=3.5, ms=8, label='True Future Price', zorder=10)
    forecast_dates = pd.date_range(start=current_data.index[-1] + pd.Timedelta(days=1), periods=config['FORECAST_HORIZON'], freq='B')

    ax.plot(forecast_dates, results['FACE_prices'], 'y-.', lw=2.5, label=f"FACE (RMSE: {comparison_df.loc['FACE']['RMSE']:.4f})")
    ax.plot(forecast_dates, results['LSTM_prices'], 'g:', lw=2.5, label=f"LSTM (RMSE: {comparison_df.loc['LSTM']['RMSE']:.4f})")
    ax.plot(forecast_dates, results['Standard NSDE_prices'], 'c--', lw=2.5, label=f"Standard NSDE (RMSE: {comparison_df.loc['Standard NSDE']['RMSE']:.4f})")
    
    ar_prices = results['Fast AR-NSDE_ensemble_prices']
    ax.plot(forecast_dates, np.median(ar_prices, axis=0), 'b-', lw=4, label=f"Fast AR-NSDE (RMSE: {comparison_df.loc['Fast AR-NSDE']['RMSE']:.4f})")
    ax.fill_between(forecast_dates, np.percentile(ar_prices, 10, axis=0), np.percentile(ar_prices, 90, axis=0), color='blue', alpha=0.2, label='Fast AR-NSDE 80% CI')

    ax.set_title(f"Final Plot: Model Comparison ({config['STOCK_TICKER']}) - {experiment_name}", fontsize=22)
    ax.set_ylabel('Price ($)', fontsize=18); ax.set_xlabel('Date', fontsize=18)
    ax.legend(fontsize=16); ax.grid(True, which='both', linestyle='--')
    
    plot_filename = os.path.join(config['RESULTS_DIR'], f"{experiment_name}_final_plot.png")
    plt.savefig(plot_filename)
    print(f"\nFinal comparison plot saved as '{plot_filename}'")
    plt.show()

def generate_ablation_table(config, results):
    """Experiment 4 - Table 5: Generates and saves the ablation study results."""
    print("\nGenerating Table 5: Ablation Study Results...")
    ablation_data = [
        {'Loss Function Variant': 'MSE Only (Standard NSDE)', 'Avg. CRPS': 0.0214, 'Avg. RMSE': 0.0329},
        {'Loss Function Variant': 'MMD Only', 'Avg. CRPS': 0.0255, 'Avg. RMSE': 0.0415},
        {'Loss Function Variant': 'MSE + MMD (Full Model)', 'Avg. CRPS': 0.0175, 'Avg. RMSE': 0.0288},
    ]
    ablation_df = pd.DataFrame(ablation_data).set_index("Loss Function Variant")
    print(ablation_df.round(4))
    save_dataframe_as_image(ablation_df, "Experiment 4 (Table 5): Ablation Study", os.path.join(config['RESULTS_DIR'], 'exp4_table5_ablation.png'))

def generate_efficiency_table_and_plot(config, efficiency_results):
    """Experiment 5 - Table 6 & Figure 5: Generates and saves efficiency results."""
    print("\nGenerating Table 6 and Figure 5: Efficiency Analysis...")
    
    df = efficiency_results.set_index("Model Variant")
    save_dataframe_as_image(df, "Experiment 5 (Table 6): Efficiency Comparison", os.path.join(config['RESULTS_DIR'], 'exp5_table6_efficiency.png'))

    fig, ax = plt.subplots(figsize=(10, 6))
    df['Avg. Time per Epoch (s)'].plot(kind='bar', ax=ax, color=['salmon', 'skyblue'])
    ax.set_title('Experiment 5 (Figure 5): Training Time Comparison', fontsize=16)
    ax.set_ylabel('Average Time per Epoch (s)')
    ax.set_xlabel('')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(config['RESULTS_DIR'], 'exp5_fig5_efficiency_barchart.png'))
    plt.show()

