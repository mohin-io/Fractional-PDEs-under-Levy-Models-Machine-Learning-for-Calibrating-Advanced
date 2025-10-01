"""
Uncertainty propagation from parameter posterior to option pricing.

This module quantifies how parameter uncertainty affects option price predictions,
providing prediction intervals and risk estimates.
"""

import numpy as np
import json
from typing import Dict, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.pricing_engine.fourier_pricer import carr_madan_pricer, price_surface
from models.pricing_engine.levy_models import variance_gamma_char_func, cgmy_char_func


def load_posterior_samples(results_path='models/bayesian_calibration/mcmc_results.json'):
    """
    Load posterior samples from MCMC results.

    Args:
        results_path (str): Path to MCMC results JSON file.

    Returns:
        dict: Posterior samples.
        str: Model name.
    """
    with open(results_path, 'r') as f:
        results = json.load(f)

    posterior_samples = {k: np.array(v) for k, v in results['posterior_samples'].items()}
    model_name = results['model_name']

    return posterior_samples, model_name


def propagate_uncertainty_single_option(posterior_samples, model_name, S0, K, T, r, q=0.0,
                                       option_type='call', num_samples=1000):
    """
    Propagate parameter uncertainty to single option price.

    Args:
        posterior_samples (dict): Posterior parameter samples.
        model_name (str): 'VarianceGamma' or 'CGMY'.
        S0 (float): Spot price.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        q (float): Dividend yield.
        option_type (str): 'call' or 'put'.
        num_samples (int): Number of posterior samples to use.

    Returns:
        dict: Predictive statistics (mean, std, credible interval).
    """
    # Extract parameter samples
    param_names = list(posterior_samples.keys())
    n_total = len(posterior_samples[param_names[0]].flatten())

    # Randomly sample from posterior
    indices = np.random.choice(n_total, size=min(num_samples, n_total), replace=False)

    prices = []
    for idx in indices:
        # Extract parameters for this sample
        if model_name == 'VarianceGamma':
            sigma = posterior_samples['sigma'].flatten()[idx]
            nu = posterior_samples['nu'].flatten()[idx]
            theta = posterior_samples['theta'].flatten()[idx]
            char_func = lambda u, t, r_: variance_gamma_char_func(u, t, r_, sigma, nu, theta)
        else:  # CGMY
            C = posterior_samples['C'].flatten()[idx]
            G = posterior_samples['G'].flatten()[idx]
            M = posterior_samples['M'].flatten()[idx]
            Y = posterior_samples['Y'].flatten()[idx]
            char_func = lambda u, t, r_: cgmy_char_func(u, t, r_, C, G, M, Y)

        # Price option with these parameters
        price = carr_madan_pricer(S0, K, T, r, char_func, q=q, option_type=option_type)
        prices.append(price[0])

    prices = np.array(prices)

    return {
        'mean': float(np.mean(prices)),
        'std': float(np.std(prices)),
        'median': float(np.median(prices)),
        'hdi_95_lower': float(np.percentile(prices, 2.5)),
        'hdi_95_upper': float(np.percentile(prices, 97.5)),
        'hdi_50_lower': float(np.percentile(prices, 25)),
        'hdi_50_upper': float(np.percentile(prices, 75)),
        'samples': prices.tolist()
    }


def propagate_uncertainty_surface(posterior_samples, model_name, S0, strikes, maturities,
                                 r, q=0.0, num_samples=1000):
    """
    Propagate uncertainty to entire option surface.

    Args:
        posterior_samples (dict): Posterior parameter samples.
        model_name (str): 'VarianceGamma' or 'CGMY'.
        S0 (float): Spot price.
        strikes (np.ndarray): Strike prices.
        maturities (np.ndarray): Maturities.
        r (float): Risk-free rate.
        q (float): Dividend yield.
        num_samples (int): Number of posterior samples.

    Returns:
        dict: Surface statistics (mean, std, credible intervals).
    """
    param_names = list(posterior_samples.keys())
    n_total = len(posterior_samples[param_names[0]].flatten())

    # Randomly sample from posterior
    indices = np.random.choice(n_total, size=min(num_samples, n_total), replace=False)

    # Store surfaces
    surfaces = []

    for idx in indices:
        # Extract parameters
        params_dict = {name: posterior_samples[name].flatten()[idx]
                      for name in param_names}

        # Price surface
        surface = price_surface(
            params=params_dict,
            model_name=model_name,
            s0=S0,
            grid_strikes=strikes,
            grid_maturities=maturities,
            r=r,
            q=q
        )
        surfaces.append(surface)

    surfaces = np.array(surfaces)  # Shape: (num_samples, n_strikes, n_maturities)

    return {
        'mean': surfaces.mean(axis=0),
        'std': surfaces.std(axis=0),
        'median': np.median(surfaces, axis=0),
        'hdi_95_lower': np.percentile(surfaces, 2.5, axis=0),
        'hdi_95_upper': np.percentile(surfaces, 97.5, axis=0)
    }


def compute_prediction_interval_coverage(posterior_samples, model_name, true_params,
                                        S0, strikes, maturities, r, q=0.0,
                                        confidence_level=0.95, num_samples=1000):
    """
    Compute prediction interval coverage probability.

    Tests if credible intervals have correct frequentist coverage.

    Args:
        posterior_samples (dict): Posterior parameter samples.
        model_name (str): Model name.
        true_params (dict): True parameter values.
        S0 (float): Spot price.
        strikes (np.ndarray): Strikes.
        maturities (np.ndarray): Maturities.
        r (float): Risk-free rate.
        q (float): Dividend yield.
        confidence_level (float): Confidence level (e.g., 0.95).
        num_samples (int): Number of posterior samples.

    Returns:
        dict: Coverage statistics.
    """
    # Compute true option surface
    true_surface = price_surface(
        params=true_params,
        model_name=model_name,
        s0=S0,
        grid_strikes=strikes,
        grid_maturities=maturities,
        r=r,
        q=q
    )

    # Propagate uncertainty
    surface_stats = propagate_uncertainty_surface(
        posterior_samples, model_name, S0, strikes, maturities, r, q, num_samples
    )

    # Check coverage
    alpha = 1 - confidence_level
    lower = np.percentile(alpha / 2 * 100, axis=0)
    upper = np.percentile((1 - alpha / 2) * 100, axis=0)

    within_interval = (true_surface >= surface_stats['hdi_95_lower']) & \
                     (true_surface <= surface_stats['hdi_95_upper'])

    coverage = np.mean(within_interval)

    return {
        'coverage': float(coverage),
        'target_coverage': confidence_level,
        'n_strikes': len(strikes),
        'n_maturities': len(maturities),
        'n_options': true_surface.size
    }


def visualize_predictive_uncertainty(posterior_samples, model_name, S0, K, T_range, r, q=0.0,
                                     option_type='call', num_samples=500):
    """
    Create data for fan chart showing predictive uncertainty across maturities.

    Args:
        posterior_samples (dict): Posterior samples.
        model_name (str): Model name.
        S0 (float): Spot price.
        K (float): Strike price (fixed).
        T_range (np.ndarray): Range of maturities.
        r (float): Risk-free rate.
        q (float): Dividend yield.
        option_type (str): 'call' or 'put'.
        num_samples (int): Posterior samples to use.

    Returns:
        dict: Data for plotting (maturities, mean, intervals).
    """
    results = {
        'maturities': T_range.tolist(),
        'mean': [],
        'median': [],
        'hdi_50_lower': [],
        'hdi_50_upper': [],
        'hdi_95_lower': [],
        'hdi_95_upper': []
    }

    for T in T_range:
        stats = propagate_uncertainty_single_option(
            posterior_samples, model_name, S0, K, T, r, q, option_type, num_samples
        )

        results['mean'].append(stats['mean'])
        results['median'].append(stats['median'])
        results['hdi_50_lower'].append(stats['hdi_50_lower'])
        results['hdi_50_upper'].append(stats['hdi_50_upper'])
        results['hdi_95_lower'].append(stats['hdi_95_lower'])
        results['hdi_95_upper'].append(stats['hdi_95_upper'])

    return results


if __name__ == "__main__":
    print("="*80)
    print("UNCERTAINTY PROPAGATION")
    print("="*80)
    print("\nThis module propagates parameter uncertainty to option prices.")
    print("\nUsage:")
    print("  1. Run MCMC calibration first to get posterior samples")
    print("  2. Use propagate_uncertainty_single_option() for individual options")
    print("  3. Use propagate_uncertainty_surface() for entire surface")
    print("  4. Use visualize_predictive_uncertainty() for fan charts")
    print("\nExample:")
    print("""
from models.bayesian_calibration.uncertainty_propagation import *

# Load posterior samples
posterior, model_name = load_posterior_samples()

# Predict single option with uncertainty
stats = propagate_uncertainty_single_option(
    posterior, model_name, S0=100, K=100, T=1.0, r=0.05
)
print(f"Option price: {stats['mean']:.2f} ± {stats['std']:.2f}")
print(f"95% credible interval: [{stats['hdi_95_lower']:.2f}, {stats['hdi_95_upper']:.2f}]")
    """)
