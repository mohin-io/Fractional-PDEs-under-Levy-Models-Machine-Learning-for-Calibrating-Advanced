"""
Utility functions for dataset generation.

This module provides common functions for generating synthetic option pricing datasets
across different Lévy models.
"""

import pandas as pd
import numpy as np
from scipy.stats import qmc
from scipy.stats.qmc import Sobol
import os
from tqdm import tqdm

from models.pricing_engine.fourier_pricer import price_surface


def generate_sobol_samples(param_ranges, num_samples, seed=None):
    """
    Generate parameter samples using Sobol quasi-random sequence.

    Args:
        param_ranges (dict): Dictionary mapping parameter names to [min, max] ranges.
        num_samples (int): Number of samples to generate.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: Array of shape (num_samples, num_params) with parameter samples.
        list: List of parameter names in order.
    """
    param_names = list(param_ranges.keys())
    param_dim = len(param_names)
    param_lower_bounds = [param_ranges[p][0] for p in param_names]
    param_upper_bounds = [param_ranges[p][1] for p in param_names]

    # Initialize Sobol sequence generator
    sobol_sampler = Sobol(d=param_dim, scramble=True, seed=seed)

    # Generate scaled parameter samples
    unit_samples = sobol_sampler.random(n=num_samples)
    param_samples = qmc.scale(unit_samples, param_lower_bounds, param_upper_bounds)

    return param_samples, param_names


def add_market_noise(prices, noise_type='bid_ask', noise_level=0.005):
    """
    Add realistic market microstructure noise to option prices.

    Args:
        prices (np.ndarray): Clean option prices.
        noise_type (str): Type of noise - 'bid_ask', 'gaussian', or 'mixed'.
        noise_level (float): Noise magnitude (default: 0.5% for bid-ask spread).

    Returns:
        np.ndarray: Noisy prices.
    """
    if noise_type == 'bid_ask':
        # Simulate bid-ask spread
        spread = np.random.uniform(-noise_level, noise_level, prices.shape)
        noisy_prices = prices * (1 + spread)
    elif noise_type == 'gaussian':
        # Add Gaussian measurement noise
        noise = np.random.normal(0, noise_level * prices, prices.shape)
        noisy_prices = prices + noise
    elif noise_type == 'mixed':
        # Combine bid-ask spread and measurement noise
        spread = np.random.uniform(-noise_level/2, noise_level/2, prices.shape)
        noise = np.random.normal(0, noise_level/2 * prices, prices.shape)
        noisy_prices = prices * (1 + spread) + noise
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    # Ensure positive prices
    return np.maximum(noisy_prices, 1e-6)


def generate_synthetic_dataset(model_name, param_ranges, num_samples, grid_strikes,
                               grid_maturities, S0=100.0, r=0.05, q=0.0,
                               add_noise=False, noise_level=0.005, progress_bar=True):
    """
    Generate synthetic dataset for a given Lévy model.

    Args:
        model_name (str): 'VarianceGamma' or 'CGMY'.
        param_ranges (dict): Parameter ranges for the model.
        num_samples (int): Number of samples to generate.
        grid_strikes (np.ndarray): Strike price grid.
        grid_maturities (np.ndarray): Maturity grid.
        S0 (float): Spot price (default: 100.0).
        r (float): Risk-free rate (default: 0.05).
        q (float): Dividend yield (default: 0.0).
        add_noise (bool): Whether to add market noise (default: False).
        noise_level (float): Noise magnitude if add_noise=True (default: 0.005).
        progress_bar (bool): Show progress bar (default: True).

    Returns:
        pd.DataFrame: Dataset with parameters and price features.
    """
    # Generate parameter samples
    param_samples, param_names = generate_sobol_samples(param_ranges, num_samples)

    results = []
    surface_feature_names = [
        f"price_{i}_{j}"
        for i in range(len(grid_strikes))
        for j in range(len(grid_maturities))
    ]

    # Loop through samples with optional progress bar
    iterator = enumerate(param_samples)
    if progress_bar:
        iterator = tqdm(iterator, total=num_samples, desc=f"Generating {model_name} dataset")

    for i, params_array in iterator:
        params_dict = {name: val for name, val in zip(param_names, params_array)}

        # Price the surface for the current parameter set
        surface = price_surface(
            params=params_dict,
            model_name=model_name,
            s0=S0,
            grid_strikes=grid_strikes,
            grid_maturities=grid_maturities,
            r=r,
            q=q
        )

        # Add market noise if requested
        if add_noise:
            surface = add_market_noise(surface, noise_type='mixed', noise_level=noise_level)

        # Flatten the 2D surface into a 1D feature vector
        flat_surface = surface.flatten()

        # Store results
        record = {name: val for name, val in zip(param_names, params_array)}
        record.update(
            {name: val for name, val in zip(surface_feature_names, flat_surface)}
        )
        results.append(record)

    return pd.DataFrame(results)


def save_dataset(df, output_path, format='parquet'):
    """
    Save dataset to file.

    Args:
        df (pd.DataFrame): Dataset to save.
        output_path (str): Output file path.
        format (str): File format - 'parquet', 'csv', or 'hdf5' (default: 'parquet').
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if format == 'parquet':
        df.to_parquet(output_path, index=False)
    elif format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'hdf5':
        df.to_hdf(output_path, key='data', mode='w')
    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"Dataset saved to {output_path}")
