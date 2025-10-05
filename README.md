Fast Analogy-Regularized Neural SDEs for Financial Forecasting
This repository provides the official implementation for the paper "Fast Analogy-Regularized Neural SDEs", a hybrid framework for financial time series forecasting. The project introduces the Fast AR-NSDE model, which combines Neural Stochastic Differential Equations with historical market patterns to improve prediction accuracy.

Quick Start
Follow these steps to set up and run the project.

1. Clone the Repository
git clone [https://github.com/your-username/Fast-AR-NSDE-Forecasting.git](https://github.com/your-username/Fast-AR-NSDE-Forecasting.git)
cd Fast-AR-NSDE-Forecasting

2. Install Dependencies
It is recommended to use a virtual environment.

# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install torch torchsde yfinance pandas numpy matplotlib scikit-learn joblib tqdm

3. Run the Experiments
The experiments.py script is the main entry point to replicate the paper's findings.

To run all five experiments sequentially:

python experiments.py

To run a specific experiment (e.g., Experiment 1):

python experiments.py --experiment 1

You can also run multiple experiments at once:

python experiments.py --experiment 1 2 5

All results, including plots and tables, will be saved in the results/ directory.

Project Structure
The code is organized into several modules for clarity:

financial_forecasting_suite/
│
├── experiments.py          # Main script to run all experiments
├── configs/                # Configuration files
├── data/                   # Data loading and preprocessing
├── models/                 # Model definitions and training logic
├── analysis/               # Analysis modules (F-ACE, clustering)
└── utils/                  # Plotting and helper functions
