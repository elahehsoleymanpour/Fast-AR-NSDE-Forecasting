# configs/experiment_configs.py
"""
Configurations for the five experiments described in the paper.
Each function returns a modified config dictionary for a specific experiment.
"""
from .base_config import CONFIG

def get_exp1_config():
    """Exp 1: Market Regime Clustering Validation"""
    exp_config = CONFIG.copy()
    exp_config['K_CLUSTERS'] = 10 # As per the paper's visualization
    return exp_config

def get_exp2_config():
    """Exp 2: Aggregate Forecasting Performance (S&P 100 simulation)"""
    exp_config = CONFIG.copy()
    # In a real scenario, you would loop over a list of tickers.
    # For this simulation, we use SPY as a representative stock.
    exp_config['STOCK_TICKER'] = 'SPY'
    return exp_config

def get_exp3_config():
    """Exp 3: Robustness to Market Stress (COVID-19 Crash)"""
    exp_config = CONFIG.copy()
    exp_config['STOCK_TICKER'] = 'SPY'
    # These dates would be used in a more advanced data_loader
    exp_config['STRESS_TEST_START'] = '2020-02-15'
    exp_config['STRESS_TEST_END'] = '2020-04-15'
    return exp_config

def get_exp4_config():
    """Exp 4: Ablation Study (Impact of Loss Components)"""
    exp_config = CONFIG.copy()
    # The main experiment file will handle running with different lambda values.
    return exp_config

def get_exp5_config():
    """Exp 5: Computational Efficiency Analysis"""
    exp_config = CONFIG.copy()
    exp_config['EPOCHS'] = 5 # Reduce epochs for a quick timing test
    return exp_config

