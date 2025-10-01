"""
Quick start example: End-to-end Lévy model calibration workflow.

This script demonstrates:
1. Generating synthetic option data
2. Training a calibration model
3. Making predictions
4. Evaluating performance
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pricing_engine.fourier_pricer import price_surface, carr_madan_pricer, compute_greeks
from models.pricing_engine.levy_models import variance_gamma_char_func
from models.dataset_utils import generate_synthetic_dataset


def example_1_price_single_option():
    """Example 1: Price a single option using Variance Gamma model."""
    print("="*80)
    print("EXAMPLE 1: Price a single VG option")
    print("="*80)

    # Define VG parameters
    S0 = 100.0      # Spot price
    K = 100.0       # Strike price (at-the-money)
    T = 1.0         # 1 year to maturity
    r = 0.05        # 5% risk-free rate
    q = 0.0         # No dividend

    # VG model parameters
    sigma = 0.2     # Volatility
    nu = 0.5        # Kurtosis parameter
    theta = -0.1    # Skewness parameter (negative for equity-like skew)

    # Create characteristic function
    char_func = lambda u, t, r_: variance_gamma_char_func(u, t, r_, sigma, nu, theta)

    # Price call option
    call_price = carr_madan_pricer(S0, K, T, r, char_func, q=q, option_type='call')
    print(f"\nVG Call Option Price: ${call_price[0]:.4f}")

    # Price put option
    put_price = carr_madan_pricer(S0, K, T, r, char_func, q=q, option_type='put')
    print(f"VG Put Option Price:  ${put_price[0]:.4f}")

    # Verify put-call parity: C - P = S*exp(-qT) - K*exp(-rT)
    parity_lhs = call_price[0] - put_price[0]
    parity_rhs = S0 * np.exp(-q * T) - K * np.exp(-r * T)
    print(f"\nPut-Call Parity Check:")
    print(f"  C - P = {parity_lhs:.6f}")
    print(f"  S - K*exp(-rT) = {parity_rhs:.6f}")
    print(f"  Difference: {abs(parity_lhs - parity_rhs):.2e} (should be near zero)")


def example_2_compute_greeks():
    """Example 2: Compute option Greeks."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Compute option Greeks")
    print("="*80)

    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    sigma, nu, theta = 0.2, 0.5, -0.1

    char_func = lambda u, t, r_: variance_gamma_char_func(u, t, r_, sigma, nu, theta)

    greeks = compute_greeks(S0, K, T, r, char_func, option_type='call')

    print(f"\nGreeks for ATM VG Call Option:")
    print(f"  Delta: {greeks['delta']:.4f} (sensitivity to spot price)")
    print(f"  Gamma: {greeks['gamma']:.6f} (curvature of delta)")
    print(f"  Theta: {greeks['theta']:.4f} (time decay per year)")
    print(f"  Rho:   {greeks['rho']:.4f} (sensitivity to interest rate)")


def example_3_price_surface():
    """Example 3: Generate option price surface."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Generate option price surface")
    print("="*80)

    # Define VG parameters
    vg_params = {
        "sigma": 0.25,
        "nu": 0.3,
        "theta": -0.15
    }

    # Define grid
    strikes = np.array([90, 95, 100, 105, 110])
    maturities = np.array([0.25, 0.5, 1.0])

    # Price surface
    surface = price_surface(
        params=vg_params,
        model_name="VarianceGamma",
        s0=100.0,
        grid_strikes=strikes,
        grid_maturities=maturities,
        r=0.05
    )

    print(f"\nOption Price Surface (rows=strikes, cols=maturities):")
    print(f"Strikes: {strikes}")
    print(f"Maturities: {maturities}")
    print("\nPrices:")
    for i, strike in enumerate(strikes):
        print(f"  K={strike:3.0f}: ", end="")
        for j, maturity in enumerate(maturities):
            print(f"${surface[i, j]:6.2f}", end="  ")
        print()


def example_4_generate_small_dataset():
    """Example 4: Generate a small synthetic dataset."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Generate synthetic dataset")
    print("="*80)

    param_ranges = {
        "sigma": [0.15, 0.35],
        "nu": [0.2, 0.8],
        "theta": [-0.3, -0.05]
    }

    strikes = np.linspace(90, 110, 5)
    maturities = np.array([0.5, 1.0])

    print(f"\nGenerating 100 samples...")
    df = generate_synthetic_dataset(
        model_name="VarianceGamma",
        param_ranges=param_ranges,
        num_samples=100,
        grid_strikes=strikes,
        grid_maturities=maturities,
        progress_bar=False
    )

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns[:8])}... ({len(df.columns)} total)")
    print(f"\nParameter statistics:")
    print(df[['sigma', 'nu', 'theta']].describe())


def main():
    """Run all examples."""
    print("\n" + "#"*80)
    print("# LÉVY MODEL CALIBRATION: QUICK START EXAMPLES")
    print("#"*80 + "\n")

    try:
        example_1_price_single_option()
        example_2_compute_greeks()
        example_3_price_surface()
        example_4_generate_small_dataset()

        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80)
        print("\nNext steps:")
        print("  1. Generate full dataset: python models/generate_dataset.py")
        print("  2. Build features: python features/build_features.py")
        print("  3. Train model: python models/calibration_net/train.py")
        print("  4. Make predictions: python models/calibration_net/predict.py")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
