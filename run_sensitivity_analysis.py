"""
Run enhanced sensitivity analysis on trained calibration model.
"""
import sys
import os
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from sklearn.model_selection import train_test_split

from analysis.sensitivity_analysis_enhanced import (
    compute_jacobian,
    plot_jacobian_heatmap,
    feature_importance_permutation,
    perturbation_analysis,
    plot_feature_importance
)

print("="*80)
print("RUNNING ENHANCED SENSITIVITY ANALYSIS")
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

# Sample a subset for Jacobian computation (expensive operation)
n_samples_jac = 100
X_sample = X_test_scaled[:n_samples_jac]
y_sample = y_test[:n_samples_jac]

print(f"\n3. Computing Jacobian matrix for {n_samples_jac} samples...")
print("   (This may take a few minutes...)")
jacobian = compute_jacobian(model, X_sample, param_names)
print(f"   Jacobian shape: {jacobian.shape}")

# Average Jacobian across samples
avg_jacobian = np.mean(np.abs(jacobian), axis=0)  # (n_params, n_features)

print("\n4. Generating Jacobian heatmaps...")
os.makedirs('outputs/figures/validation', exist_ok=True)
plot_jacobian_heatmap(avg_jacobian, param_names, grid_shape=(20, 10),
                     save_path='outputs/figures/validation/sensitivity_jacobian_heatmap.png')

print("\n5. Computing feature importance via permutation...")
n_samples_perm = 1000
X_perm = X_test_scaled[:n_samples_perm]
y_perm = y_test[:n_samples_perm]

feature_importance = feature_importance_permutation(model, X_perm, y_perm, param_names,
                                                   n_repeats=5)
print("   Feature importance computed.")

# Save feature importance
os.makedirs('outputs/reports', exist_ok=True)
feature_importance.to_csv('outputs/reports/feature_importance.csv', index=False)
print("   Saved to: outputs/reports/feature_importance.csv")

print("\n6. Running perturbation analysis...")
perturbation_levels = [0.01, 0.05, 0.1, 0.2]
n_samples_pert = 500
X_pert = X_test_scaled[:n_samples_pert]
y_pert = y_test[:n_samples_pert]

perturbation_results = perturbation_analysis(model, X_pert, param_names,
                                            perturbation_pcts=perturbation_levels)

# Save perturbation results
perturbation_results.to_csv('outputs/reports/perturbation_analysis.csv', index=False)
print("   Saved to: outputs/reports/perturbation_analysis.csv")

print("\n7. Generating feature importance plot...")
plot_feature_importance(feature_importance, top_k=20,
                       save_path='outputs/figures/validation/feature_importance.png')

print("\n8. Summary statistics...")
print("\n--- Average Absolute Jacobian (Top 5 features per parameter) ---")
for i, param in enumerate(param_names):
    top_indices = np.argsort(avg_jacobian[i, :])[-5:][::-1]
    print(f"\n{param.upper()}:")
    for idx in top_indices:
        print(f"  Feature {idx}: {avg_jacobian[i, idx]:.6f}")

print("\n" + "="*80)
print("SENSITIVITY ANALYSIS COMPLETE")
print("="*80)
print("\nGenerated outputs:")
print("  - outputs/figures/validation/sensitivity_jacobian_heatmap.png")
print("  - outputs/figures/validation/feature_importance.png")
print("  - outputs/reports/feature_importance.csv")
print("  - outputs/reports/perturbation_analysis.csv")
