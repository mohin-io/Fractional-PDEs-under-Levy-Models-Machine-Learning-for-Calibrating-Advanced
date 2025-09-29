import pandas as pd
import numpy as np
import os

# --- Configuration ---
INPUT_FILE = 'data/synthetic/training_data.parquet'
OUTPUT_DIR = 'data/processed'
FEATURES_FILE = os.path.join(OUTPUT_DIR, 'features.parquet')
TARGETS_FILE = os.path.join(OUTPUT_DIR, 'targets.parquet')

def build_features():
    """
    Loads the synthetic dataset, extracts features (option price surface)
    and targets (Levy model parameters), and saves them to separate files.
    """
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)

    # Identify feature columns (price_X_Y) and target columns (sigma, nu, theta, C, G, M, Y)
    feature_cols = [col for col in df.columns if col.startswith('price_')]
    target_cols = ['sigma', 'nu', 'theta', 'C', 'G', 'M', 'Y'] # Include all possible params

    # Filter out target columns that are not present in the dataframe (e.g., if only VG was generated)
    actual_target_cols = [col for col in target_cols if col in df.columns]

    features_df = df[feature_cols]
    targets_df = df[actual_target_cols]

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Saving features to {FEATURES_FILE}...")
    features_df.to_parquet(FEATURES_FILE)

    print(f"Saving targets to {TARGETS_FILE}...")
    targets_df.to_parquet(TARGETS_FILE)

    print("Feature engineering complete.")

if __name__ == '__main__':
    build_features()
