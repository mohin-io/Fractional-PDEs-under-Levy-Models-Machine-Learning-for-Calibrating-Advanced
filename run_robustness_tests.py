"""
Run comprehensive robustness tests on trained calibration model.
"""
import sys
import os
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from sklearn.model_selection import train_test_split

from analysis.robustness_tests import (
    test_noise_robustness,
    detect_out_of_distribution,
    test_missing_data,
    test_extreme_values,
    plot_robustness_results
)

print("="*80)
print("RUNNING COMPREHENSIVE ROBUSTNESS TESTS")
print("="*80)

# Load data
print("\n1. Loading data...")
X = pd.read_parquet('data/processed/features.parquet').values
y = pd.read_parquet('data/processed/targets.parquet').values

# Load model and scaler
print("2. Loading model and scaler...")
model = keras.models.load_model('models/calibration_net/mlp_calibration_model.h5', compile=False)
scaler = joblib.load('models/calibration_net/scaler_X.pkl')

# Split data (same as training)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
X_test_scaled = scaler.transform(X_test)

param_names = ['sigma', 'nu', 'theta']

# Use subset for efficiency
n_samples = 1000
X_subset = X_test_scaled[:n_samples]
y_subset = y_test[:n_samples]

print(f"\n3. Testing noise robustness ({n_samples} samples)...")
noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
noise_types = ['gaussian', 'uniform', 'salt_pepper']

all_noise_results = []
for noise_type in noise_types:
    print(f"   Testing {noise_type} noise...")
    noise_results = test_noise_robustness(
        model, X_subset, y_subset, param_names,
        noise_levels=noise_levels,
        noise_type=noise_type
    )
    all_noise_results.append(noise_results)

# Combine all results
noise_results = pd.concat(all_noise_results, ignore_index=True)

print("   Noise robustness tested.")
os.makedirs('outputs/reports', exist_ok=True)
noise_results.to_csv('outputs/reports/noise_robustness.csv', index=False)
print("   Saved to: outputs/reports/noise_robustness.csv")

print(f"\n4. Testing out-of-distribution detection ({n_samples} samples)...")
# OOD detection requires training set for Mahalanobis distance
X_train_scaled = scaler.transform(X_temp[:5000])  # Use subset of training data
ood_result = detect_out_of_distribution(X_train_scaled, X_subset, threshold_percentile=95)

if isinstance(ood_result, dict):
    ood_mask = ood_result.get('is_ood', np.zeros(len(X_subset), dtype=bool))
    ood_distances = ood_result.get('distances', np.zeros(len(X_subset)))
else:
    ood_mask = ood_result
    ood_distances = np.zeros(len(X_subset))

ood_rate = np.mean(ood_mask)

print(f"   OOD detection: {ood_rate*100:.2f}% of test samples flagged as OOD")
# Save results
ood_df = pd.DataFrame({
    'Sample Index': range(len(ood_mask)),
    'Is OOD': ood_mask,
    'Mahalanobis Distance': ood_distances
})
ood_df.to_csv('outputs/reports/ood_detection.csv', index=False)
print("   Saved to: outputs/reports/ood_detection.csv")

print(f"\n5. Testing missing data robustness ({n_samples} samples)...")
missing_rates = [0.0, 0.05, 0.1, 0.2, 0.3]

missing_results = test_missing_data(
    model, X_subset, y_subset, param_names,
    missing_pcts=missing_rates
)

print("   Missing data robustness tested.")
missing_results.to_csv('outputs/reports/missing_data_robustness.csv', index=False)
print("   Saved to: outputs/reports/missing_data_robustness.csv")

print("\n6. Generating robustness visualization...")
os.makedirs('outputs/figures/validation', exist_ok=True)
plot_robustness_results(
    noise_results, extreme_results=None, missing_results=missing_results,
    save_path='outputs/figures/validation/robustness_tests.png'
)

print("\n7. Summary Statistics...")
print("\n--- Noise Robustness Summary ---")
print(f"   Total tests: {len(noise_results)}")
print(f"   Noise types tested: gaussian, uniform, salt_pepper")
print(f"   Noise levels tested: 0%, 1%, 5%, 10%, 20%")

print("\n--- Missing Data Summary ---")
print(f"   Total tests: {len(missing_results)}")
print(f"   Missing rates tested: 0%, 5%, 10%, 20%, 30%")

print("\n--- OOD Detection Summary ---")
print(f"   OOD rate: {ood_rate*100:.2f}%")

print("\n" + "="*80)
print("ROBUSTNESS TESTS COMPLETE")
print("="*80)
print("\nGenerated outputs:")
print("  - outputs/reports/noise_robustness.csv")
print("  - outputs/reports/ood_detection.csv")
print("  - outputs/reports/missing_data_robustness.csv")
print("  - outputs/figures/validation/robustness_tests.png")
