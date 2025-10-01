"""
Script to generate synthetic CGMY dataset for training.

The CGMY (Carr-Geman-Madan-Yor) model is a generalization of the Variance Gamma
model with more flexible control over jump activity via the Y parameter.

This script generates training data by:
1. Sampling CGMY parameters (C, G, M, Y) uniformly using Sobol sequences
2. Pricing option surfaces via Fourier methods
3. Saving (surface, params) pairs for ML training
"""

import numpy as np
import argparse
from models.dataset_utils import generate_synthetic_dataset, save_dataset

# --- Configuration ---

# CGMY parameter ranges
# C: overall activity level of jumps
# G: rate of exponential decay of right tail (positive jumps)
# M: rate of exponential decay of left tail (negative jumps)
# Y: controls fine structure (Y < 2 required for finite variation)
PARAM_RANGES = {
    "C": [0.01, 0.5],   # Jump activity
    "G": [1.0, 10.0],   # Right tail decay
    "M": [1.0, 10.0],   # Left tail decay
    "Y": [0.1, 1.8],    # Fine structure (must be < 2)
}

# Fixed parameters for pricing grid
S0 = 100.0  # Spot price
R = 0.05    # Risk-free rate
Q = 0.0     # Dividend yield

# Option grid (must match VG dataset for consistency)
GRID_STRIKES = np.linspace(80, 120, 20)      # 20 strikes
GRID_MATURITIES = np.linspace(0.1, 2.0, 10)  # 10 maturities

# Default dataset size
NUM_SAMPLES = 100_000

# Output configuration
OUTPUT_DIR = "data/synthetic"
OUTPUT_FILE = "cgmy_training_data.parquet"


def main(num_samples=NUM_SAMPLES, add_noise=False, noise_level=0.005):
    """
    Generate CGMY synthetic dataset.

    Args:
        num_samples (int): Number of samples to generate.
        add_noise (bool): Whether to add market microstructure noise.
        noise_level (float): Noise magnitude (percentage).
    """
    print(f"Generating CGMY dataset with {num_samples:,} samples...")
    print(f"Parameter ranges: {PARAM_RANGES}")
    print(f"Grid: {len(GRID_STRIKES)} strikes × {len(GRID_MATURITIES)} maturities")
    print(f"Market noise: {'Enabled' if add_noise else 'Disabled'}")

    # Generate dataset
    df = generate_synthetic_dataset(
        model_name="CGMY",
        param_ranges=PARAM_RANGES,
        num_samples=num_samples,
        grid_strikes=GRID_STRIKES,
        grid_maturities=GRID_MATURITIES,
        S0=S0,
        r=R,
        q=Q,
        add_noise=add_noise,
        noise_level=noise_level,
        progress_bar=True
    )

    # Save dataset
    output_path = f"{OUTPUT_DIR}/{OUTPUT_FILE}"
    if add_noise:
        # Save noisy variant with different filename
        output_path = output_path.replace('.parquet', '_noisy.parquet')

    save_dataset(df, output_path, format='parquet')

    # Print summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Total samples: {len(df):,}")
    print(f"Features: {len([c for c in df.columns if c.startswith('price_')])} price features")
    print(f"Parameters: {list(PARAM_RANGES.keys())}")
    print("\nParameter Statistics:")
    print(df[list(PARAM_RANGES.keys())].describe())

    # Check for any NaN or inf values
    if df.isnull().any().any():
        print("\n⚠ Warning: Dataset contains NaN values!")
    if np.isinf(df.select_dtypes(include=[np.number]).values).any():
        print("\n⚠ Warning: Dataset contains infinite values!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CGMY synthetic dataset")
    parser.add_argument('--num_samples', type=int, default=NUM_SAMPLES,
                       help=f'Number of samples to generate (default: {NUM_SAMPLES:,})')
    parser.add_argument('--add_noise', action='store_true',
                       help='Add market microstructure noise')
    parser.add_argument('--noise_level', type=float, default=0.005,
                       help='Noise level as percentage (default: 0.005 = 0.5%%)')

    args = parser.parse_args()

    main(num_samples=args.num_samples, add_noise=args.add_noise, noise_level=args.noise_level)
