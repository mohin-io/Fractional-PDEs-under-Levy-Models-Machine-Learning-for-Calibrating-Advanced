"""
Run comprehensive model comparison for all trained architectures.
"""
import sys
import os
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json

from analysis.model_comparison import (
    evaluate_model,
    compare_models,
    benchmark_inference_speed,
    test_robustness,
    save_comparison_table,
    print_comparison_summary
)

print("="*80)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*80)

# Load data
print("\n1. Loading data...")
X = pd.read_parquet('data/processed/features.parquet').values
y = pd.read_parquet('data/processed/targets.parquet').values

# Split data (same as training)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_names = ['sigma', 'nu', 'theta']

# Load available models
print("\n2. Loading trained models...")
models_dict = {}
scalers_dict = {}

# MLP model
if os.path.exists('models/calibration_net/mlp_calibration_model.h5'):
    print("   - Loading MLP model...")
    models_dict['MLP'] = keras.models.load_model(
        'models/calibration_net/mlp_calibration_model.h5', compile=False
    )
    scalers_dict['MLP'] = joblib.load('models/calibration_net/scaler_X.pkl')

# CNN model
if os.path.exists('models/calibration_net/mlp_calibration_model_cnn.h5'):
    print("   - Loading CNN model...")
    models_dict['CNN'] = keras.models.load_model(
        'models/calibration_net/mlp_calibration_model_cnn.h5', compile=False
    )
    if os.path.exists('models/calibration_net/scaler_X_cnn.pkl'):
        scalers_dict['CNN'] = joblib.load('models/calibration_net/scaler_X_cnn.pkl')
    else:
        scalers_dict['CNN'] = scalers_dict['MLP']  # Use MLP scaler if CNN scaler not found

# ResNet model
if os.path.exists('models/calibration_net/mlp_calibration_model_resnet.h5'):
    print("   - Loading ResNet model...")
    models_dict['ResNet'] = keras.models.load_model(
        'models/calibration_net/mlp_calibration_model_resnet.h5', compile=False
    )
    if os.path.exists('models/calibration_net/scaler_X_resnet.pkl'):
        scalers_dict['ResNet'] = joblib.load('models/calibration_net/scaler_X_resnet.pkl')
    else:
        scalers_dict['ResNet'] = scalers_dict['MLP']  # Use MLP scaler

print(f"   Loaded {len(models_dict)} models: {list(models_dict.keys())}")

# Scale test data for each model
X_test_dict = {}
for name, scaler in scalers_dict.items():
    X_test_dict[name] = scaler.transform(X_test)

print("\n3. Running comprehensive model comparison...")
comparison_results = compare_models(models_dict, X_test_dict['MLP'], y_test, param_names)

print("\n4. Benchmarking inference speed...")
X_sample = X_test_dict['MLP'][:1]  # Single sample
speed_results = benchmark_inference_speed(models_dict, X_sample, num_runs=100)

print("\n5. Testing robustness to noise...")
X_subset = X_test_dict['MLP'][:500]
y_subset = y_test[:500]
noise_levels = [0.01, 0.05, 0.1]
robustness_results = test_robustness(models_dict, X_subset, y_subset, noise_levels)

print("\n6. Saving results...")
os.makedirs('outputs/tables', exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)

# Save comparison table
save_comparison_table(comparison_results, 'outputs/tables/model_comparison.csv')

# Save speed benchmark
speed_results.to_csv('outputs/tables/speed_benchmark.csv', index=False)
print("   Speed benchmark saved to: outputs/tables/speed_benchmark.csv")

# Save robustness results
robustness_results.to_csv('outputs/tables/robustness_comparison.csv', index=False)
print("   Robustness comparison saved to: outputs/tables/robustness_comparison.csv")

# Load training histories
print("\n7. Analyzing training histories...")
histories = {}
for arch in ['cnn', 'resnet']:
    history_file = f'models/calibration_net/training_history_{arch}.json'
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            histories[arch.upper()] = json.load(f)

# Create comprehensive comparison report
print("\n8. Generating comparison visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Test performance comparison
ax = axes[0, 0]
metrics = ['MSE', 'MAE', 'R² (mean)']
x_pos = np.arange(len(models_dict))
width = 0.25

for i, metric in enumerate(metrics):
    values = comparison_results[metric].values
    ax.bar(x_pos + i*width, values, width, label=metric, alpha=0.8)

ax.set_xlabel('Model Architecture')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x_pos + width)
ax.set_xticklabels(comparison_results['Model'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 2: Inference speed comparison
ax = axes[0, 1]
mean_times = speed_results['Mean (ms)'].values
models_list = speed_results['Model'].values
colors = sns.color_palette('husl', len(models_list))
bars = ax.barh(models_list, mean_times, color=colors, alpha=0.8)
ax.set_xlabel('Inference Time (ms)')
ax.set_title('Inference Speed Comparison')
ax.grid(axis='x', alpha=0.3)

# Add value labels
for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2,
            f'{width:.2f}ms', ha='left', va='center', fontsize=9)

# Plot 3: Model complexity (parameters)
ax = axes[1, 0]
params = comparison_results['Parameters'].values / 1000  # Convert to thousands
bars = ax.bar(comparison_results['Model'], params, color=colors, alpha=0.8)
ax.set_ylabel('Parameters (thousands)')
ax.set_title('Model Complexity')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height,
            f'{height:.0f}K', ha='center', va='bottom', fontsize=9)

# Plot 4: Robustness to noise
ax = axes[1, 1]
for model_name in models_dict.keys():
    model_data = robustness_results[robustness_results['Model'] == model_name]
    if not model_data.empty:
        noise_lvls = (model_data['Noise Level'].values * 100).astype(str)
        mae_values = model_data['MAE'].values
        ax.plot(noise_lvls, mae_values, marker='o', label=model_name, linewidth=2)

ax.set_xlabel('Noise Level (%)')
ax.set_ylabel('Mean Absolute Error')
ax.set_title('Robustness to Input Noise')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/validation/model_comparison_summary.png', dpi=300, bbox_inches='tight')
print("   Comparison summary saved to: outputs/figures/validation/model_comparison_summary.png")

# Print comprehensive summary
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)
print_comparison_summary(comparison_results)

print("\n--- Speed Benchmark ---")
print(speed_results[['Model', 'Mean (ms)', 'P95 (ms)']].to_string(index=False))

print("\n" + "="*80)
print("MODEL COMPARISON COMPLETE")
print("="*80)
print("\nGenerated outputs:")
print("  - outputs/tables/model_comparison.csv")
print("  - outputs/tables/model_comparison.md")
print("  - outputs/tables/speed_benchmark.csv")
print("  - outputs/tables/robustness_comparison.csv")
print("  - outputs/figures/validation/model_comparison_summary.png")
