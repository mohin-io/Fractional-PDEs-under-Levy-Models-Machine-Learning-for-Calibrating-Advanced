"""
Run comprehensive residual analysis on trained calibration model.
"""
import sys
import os
sys.path.insert(0, '.')

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from analysis.residual_analysis import (
    test_normality,
    plot_residual_diagnostics,
    plot_autocorrelation,
    test_heteroscedasticity,
    print_diagnostic_summary
)

print("="*80)
print("RUNNING COMPREHENSIVE RESIDUAL ANALYSIS")
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

# Make predictions
print("3. Generating predictions...")
y_pred = model.predict(X_test_scaled, verbose=0)

# Calculate residuals
residuals = y_test - y_pred

param_names = ['sigma', 'nu', 'theta']

print("\n4. Running normality tests...")
normality_results = test_normality(residuals, param_names=param_names)
print(normality_results.to_string())
# Save to file
os.makedirs('outputs/reports', exist_ok=True)
normality_results.to_csv('outputs/reports/normality_tests.csv', index=False)

print("\n5. Generating residual diagnostic plots...")
plot_residual_diagnostics(y_test, y_pred, param_names=param_names,
                         save_path='outputs/figures/validation/residual_diagnostics.png')

print("\n6. Generating Q-Q plots...")
# Q-Q plot already created, skip

print("\n7. Plotting autocorrelation...")
plot_autocorrelation(residuals, param_names=param_names,
                    save_path='outputs/figures/validation/residual_autocorrelation.png')

print("\n8. Testing heteroscedasticity...")
hetero_results = test_heteroscedasticity(y_pred, residuals, param_names=param_names)
print(hetero_results.to_string())
hetero_results.to_csv('outputs/reports/heteroscedasticity_tests.csv', index=False)

print("\n9. Generating comprehensive diagnostic summary...")
print_diagnostic_summary(y_test, y_pred, param_names=param_names)

print("\n" + "="*80)
print("RESIDUAL ANALYSIS COMPLETE")
print("="*80)
print("\nGenerated outputs:")
print("  - outputs/figures/validation/residual_diagnostics.png")
print("  - outputs/figures/validation/residual_autocorrelation.png")
print("  - outputs/figures/validation/residual_qq_plot.png")
