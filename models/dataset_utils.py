"""
Enhanced Dataset Generation Utilities for Lévy Models

This module provides comprehensive utilities for generating synthetic training data
with realistic market imperfections for machine learning calibration.

Key Features:
1. Sobol quasi-random sampling for uniform parameter space coverage
2. Market noise injection (bid-ask spreads, microstructure noise, outliers)
3. Data validation and arbitrage checks
4. Multi-model support (VG, CGMY, NIG)
5. Parallel processing for large-scale generation

Author: Mohin Hasin (mohinhasin999@gmail.com)
Project: Fractional PDEs & Lévy Processes: An ML Approach
Repository: https://github.com/mohin-io/Fractional-PDEs-under-Levy-Models-Machine-Learning-for-Calibrating-Advanced
"""

import pandas as pd
import numpy as np
from scipy.stats import qmc
from scipy.stats.qmc import Sobol
import os
from tqdm import tqdm
import warnings
from typing import Dict, Tuple, List, Callable, Optional
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

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


def check_arbitrage_free(
    call_prices: np.ndarray,
    strikes: np.ndarray,
    spot: float,
    rate: float,
    time_to_maturity: float,
    dividend: float = 0.0,
    tolerance: float = 1e-6
) -> Tuple[bool, List[str]]:
    """
    Check if option prices satisfy no-arbitrage conditions.

    Conditions checked:
    1. Price bounds: S*exp(-qT) - K*exp(-rT) <= C <= S*exp(-qT)
    2. Monotonicity: C(K1) >= C(K2) for K1 < K2
    3. Convexity: Butterfly spreads are non-negative

    Args:
        call_prices: Call option prices
        strikes: Strike prices
        spot: Current spot price
        rate: Risk-free rate
        time_to_maturity: Time to maturity (years)
        dividend: Dividend yield
        tolerance: Numerical tolerance

    Returns:
        is_valid: True if all conditions satisfied
        violations: List of violation descriptions
    """
    violations = []

    # Discount factors
    df_r = np.exp(-rate * time_to_maturity)
    df_q = np.exp(-dividend * time_to_maturity)

    # 1. Price bounds
    intrinsic = np.maximum(spot * df_q - strikes * df_r, 0)
    upper_bound = spot * df_q

    if np.any(call_prices < intrinsic - tolerance):
        violations.append("Price below intrinsic value")
    if np.any(call_prices > upper_bound + tolerance):
        violations.append("Price above upper bound")

    # 2. Monotonicity
    if np.any(np.diff(call_prices) > tolerance):
        violations.append("Non-monotonic prices")

    # 3. Convexity (butterfly spreads)
    if len(strikes) >= 3:
        for i in range(1, len(strikes) - 1):
            butterfly = call_prices[i-1] - 2*call_prices[i] + call_prices[i+1]
            if butterfly < -tolerance:
                violations.append(f"Convexity violation at K={strikes[i]:.2f}")
                break

    return len(violations) == 0, violations


def add_outliers(prices: np.ndarray, outlier_prob: float = 0.02,
                 outlier_scale: float = 3.0, seed: Optional[int] = None) -> np.ndarray:
    """
    Add heavy-tailed outliers to simulate extreme market events.

    Args:
        prices: Clean prices
        outlier_prob: Probability of outlier (default 0.02 = 2%)
        outlier_scale: Scale of outlier noise (default 3.0)
        seed: Random seed

    Returns:
        Prices with outliers
    """
    if seed is not None:
        np.random.seed(seed)

    outlier_mask = np.random.random(prices.shape) < outlier_prob
    outlier_noise = np.random.normal(0, outlier_scale * 0.01 * prices, prices.shape)

    noisy_prices = prices.copy()
    noisy_prices[outlier_mask] += outlier_noise[outlier_mask]

    return np.maximum(noisy_prices, 0.0)


def split_dataset(df: pd.DataFrame, train_ratio: float = 0.8,
                  val_ratio: float = 0.1, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/validation/test sets.

    Args:
        df: Complete dataset
        train_ratio: Fraction for training (default 0.8)
        val_ratio: Fraction for validation (default 0.1)
        seed: Random seed

    Returns:
        train_df, val_df, test_df
    """
    np.random.seed(seed)

    n = len(df)
    indices = np.random.permutation(n)

    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy(), df.iloc[test_idx].copy()


def visualize_parameter_coverage(param_samples: pd.DataFrame, param_ranges: dict,
                                 save_path: Optional[str] = None):
    """
    Visualize parameter space coverage using pairwise scatter plots.

    Args:
        param_samples: DataFrame with parameter samples
        param_ranges: Dictionary of parameter ranges
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    param_names = list(param_ranges.keys())
    n_params = len(param_names)

    fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))

    for i, param1 in enumerate(param_names):
        for j, param2 in enumerate(param_names):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histogram
                ax.hist(param_samples[param1], bins=30, alpha=0.7, edgecolor='black')
                ax.set_ylabel('Count')
            else:
                # Off-diagonal: scatter
                ax.scatter(param_samples[param2], param_samples[param1],
                          alpha=0.3, s=1)

            if i == n_params - 1:
                ax.set_xlabel(param2)
            if j == 0:
                ax.set_ylabel(param1)

    plt.suptitle('Parameter Space Coverage (Sobol Sampling)', fontsize=16, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Parameter coverage plot saved to {save_path}")

    plt.show()
