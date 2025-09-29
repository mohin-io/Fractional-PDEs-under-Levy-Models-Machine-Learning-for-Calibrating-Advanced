"""
Script to generate a large synthetic dataset for training the calibration model.

This script performs the following steps:
1. Defines realistic parameter ranges for a chosen Lévy model.
2. Samples a large number of parameter sets from these ranges using a quasi-random
   Sobol sequence to ensure uniform coverage.
3. For each parameter set, it prices the corresponding option surface using the
   Fourier-based pricer.
4. The resulting (price surface, parameter set) pairs are saved to a Parquet file.

This is a computationally intensive, offline process designed to be run once to
create the training data.
"""
import pandas as pd
import numpy as np
from scipy.stats import qmc
from scipy.stats.qmc import Sobol
import os

from models.pricing_engine.fourier_pricer import price_surface

# --- Configuration ---

# Dataset size
NUM_SAMPLES = 100_000 # Use a smaller number for quick tests, 1M+ for final training

# Model to use
MODEL_NAME = 'VarianceGamma'

# Define realistic parameter ranges for the Variance Gamma model
# [sigma, nu, theta]
PARAM_RANGES = {
    'sigma': [0.1, 0.6],  # Volatility
    'nu':    [0.1, 1.0],  # Kurtosis control
    'theta': [-0.5, 0.0]   # Skew control (often negative for equities)
}

# Fixed parameters for the pricing grid
S0 = 100.0  # Spot price
R = 0.05    # Risk-free rate

# The fixed grid for the option surface (features for the ML model)
# This grid MUST match the one used in the feature engineering phase.
GRID_STRIKES = np.linspace(80, 120, 20) # 20 strikes
GRID_MATURITIES = np.linspace(0.1, 2.0, 10) # 10 maturities (e.g., ~1 month to 2 years)

# Output file path
OUTPUT_DIR = 'data/synthetic'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'training_data.parquet')

def generate_dataset():
    """Main function to generate and save the dataset."""
    print(f"Starting dataset generation for {NUM_SAMPLES} samples...")

    param_names = list(PARAM_RANGES.keys())
    param_dim = len(param_names)
    param_lower_bounds = [PARAM_RANGES[p][0] for p in param_names]
    param_upper_bounds = [PARAM_RANGES[p][1] for p in param_names]

    # 1. Initialize Sobol sequence generator
    sobol_sampler = Sobol(d=param_dim, scramble=True)

    # 2. Generate scaled parameter samples
    unit_samples = sobol_sampler.random(n=NUM_SAMPLES)
    param_samples = qmc.scale(unit_samples, param_lower_bounds, param_upper_bounds)

    results = []
    surface_feature_names = [f'price_{i}_{j}' for i in range(len(GRID_STRIKES)) for j in range(len(GRID_MATURITIES))]

    # 3. Loop through samples and price surfaces
    for i, params_array in enumerate(param_samples):
        if (i + 1) % 1000 == 0:
            print(f"Processing sample {i + 1}/{NUM_SAMPLES}...")

        params_dict = {name: val for name, val in zip(param_names, params_array)}

        # Price the surface for the current parameter set
        surface = price_surface(
            params=params_dict,
            model_name=MODEL_NAME,
            s0=S0,
            grid_strikes=GRID_STRIKES,
            grid_maturities=GRID_MATURITIES,
            r=R
        )

        # Flatten the 2D surface into a 1D feature vector
        flat_surface = surface.flatten()

        # Store results
        record = {name: val for name, val in zip(param_names, params_array)}
        record.update({name: val for name, val in zip(surface_feature_names, flat_surface)})
        results.append(record)

    # 4. Save the dataset to a Parquet file
    print("Saving dataset to file...")
    df = pd.DataFrame(results)
    
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_parquet(OUTPUT_FILE)

    print(f"Dataset generation complete. Saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    generate_dataset()
