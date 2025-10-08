"""
Model comparison framework for calibration networks.

Compares different architectures (MLP, CNN, ResNet, Ensemble) on:
- Accuracy metrics (MSE, MAE, R²)
- Inference speed
- Model complexity
- Robustness to noise
"""

import numpy as np
import pandas as pd
import time
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.calibration_net.model import build_mlp_model
from models.calibration_net.architectures import build_cnn_model, build_resnet_model, CalibrationEnsemble


def evaluate_model(model, X_test, y_test, model_name='Model'):
    """
    Evaluate a single model on test set.

    Args:
        model: Trained Keras model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test targets.
        model_name (str): Name for display.

    Returns:
        dict: Evaluation metrics.
    """
    # Make predictions
    start_time = time.time()
    y_pred = model.predict(X_test, verbose=0)
    inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample

    # Compute metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # R² per parameter
    r2_scores = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
    r2_mean = np.mean(r2_scores)

    # MAE per parameter
    mae_per_param = [mean_absolute_error(y_test[:, i], y_pred[:, i])
                     for i in range(y_test.shape[1])]

    # Model complexity
    num_params = model.count_params()

    return {
        'model_name': model_name,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2_mean': r2_mean,
        'r2_per_param': r2_scores,
        'mae_per_param': mae_per_param,
        'inference_time_ms': inference_time,
        'num_parameters': num_params,
        'predictions': y_pred
    }


def compare_models(models_dict, X_test, y_test, param_names=None):
    """
    Compare multiple models on the same test set.

    Args:
        models_dict (dict): Dictionary mapping model names to trained models.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test targets.
        param_names (list): List of parameter names (e.g., ['sigma', 'nu', 'theta']).

    Returns:
        pd.DataFrame: Comparison table with metrics for each model.
    """
    if param_names is None:
        param_names = [f'param_{i}' for i in range(y_test.shape[1])]

    results = []

    print("Evaluating models...")
    for model_name, model in models_dict.items():
        print(f"  - {model_name}...")
        metrics = evaluate_model(model, X_test, y_test, model_name)

        # Store summary metrics
        result = {
            'Model': model_name,
            'MSE': metrics['mse'],
            'MAE': metrics['mae'],
            'RMSE': metrics['rmse'],
            'R² (mean)': metrics['r2_mean'],
            'Inference (ms)': metrics['inference_time_ms'],
            'Parameters': metrics['num_parameters']
        }

        # Add per-parameter metrics
        for i, param in enumerate(param_names):
            result[f'R² ({param})'] = metrics['r2_per_param'][i]
            result[f'MAE ({param})'] = metrics['mae_per_param'][i]

        results.append(result)

    # Create comparison DataFrame
    df = pd.DataFrame(results)

    return df


def benchmark_inference_speed(models_dict, X_sample, num_runs=100):
    """
    Benchmark inference speed for multiple models.

    Args:
        models_dict (dict): Dictionary of models.
        X_sample (np.ndarray): Sample input (single instance).
        num_runs (int): Number of inference runs for averaging.

    Returns:
        pd.DataFrame: Timing results.
    """
    results = []

    for model_name, model in models_dict.items():
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            model.predict(X_sample, verbose=0)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        results.append({
            'Model': model_name,
            'Mean (ms)': np.mean(times),
            'Std (ms)': np.std(times),
            'Min (ms)': np.min(times),
            'Max (ms)': np.max(times),
            'P50 (ms)': np.percentile(times, 50),
            'P95 (ms)': np.percentile(times, 95),
            'P99 (ms)': np.percentile(times, 99)
        })

    return pd.DataFrame(results)


def test_robustness(models_dict, X_test, y_test, noise_levels=[0.01, 0.05, 0.1]):
    """
    Test model robustness to input noise.

    Args:
        models_dict (dict): Dictionary of models.
        X_test (np.ndarray): Clean test features.
        y_test (np.ndarray): Test targets.
        noise_levels (list): List of noise levels to test.

    Returns:
        pd.DataFrame: Robustness results.
    """
    results = []

    for noise_level in noise_levels:
        print(f"\nTesting with {noise_level*100}% noise...")

        # Add Gaussian noise
        noise = np.random.normal(0, noise_level * np.abs(X_test), X_test.shape)
        X_noisy = X_test + noise

        for model_name, model in models_dict.items():
            y_pred = model.predict(X_noisy, verbose=0)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            results.append({
                'Model': model_name,
                'Noise Level': f'{noise_level*100}%',
                'MAE': mae,
                'MSE': mse,
                'RMSE': np.sqrt(mse)
            })

    return pd.DataFrame(results)


def save_comparison_table(df, output_path='outputs/tables/model_comparison.csv'):
    """
    Save comparison table to file.

    Args:
        df (pd.DataFrame): Comparison results.
        output_path (str): Output file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nComparison table saved to {output_path}")

    # Also save as markdown for GitHub
    md_path = output_path.replace('.csv', '.md')
    try:
        with open(md_path, 'w') as f:
            f.write(df.to_markdown(index=False))
        print(f"Markdown table saved to {md_path}")
    except ImportError:
        # Fallback: create simple markdown table manually
        with open(md_path, 'w') as f:
            # Header
            f.write('| ' + ' | '.join(df.columns) + ' |\n')
            f.write('| ' + ' | '.join(['---'] * len(df.columns)) + ' |\n')
            # Rows
            for _, row in df.iterrows():
                f.write('| ' + ' | '.join(str(v) for v in row.values) + ' |\n')
        print(f"Markdown table saved to {md_path} (simple format)")


def print_comparison_summary(df):
    """
    Print formatted comparison summary.

    Args:
        df (pd.DataFrame): Comparison results.
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)

    # Main metrics
    cols = ['Model', 'MSE', 'MAE', 'R² (mean)', 'Inference (ms)', 'Parameters']
    print("\n" + df[cols].to_string(index=False))

    # Best models
    print("\n" + "-"*80)
    print("BEST MODELS BY METRIC:")
    print("-"*80)
    print(f"  Best Accuracy (lowest MSE):  {df.loc[df['MSE'].idxmin(), 'Model']}")
    print(f"  Best Speed (fastest):        {df.loc[df['Inference (ms)'].idxmin(), 'Model']}")
    print(f"  Most Compact (fewest params): {df.loc[df['Parameters'].idxmin(), 'Model']}")

    # Speed-accuracy tradeoff
    df['Speed-Accuracy Score'] = (1 - df['MSE'] / df['MSE'].max()) * (1 - df['Inference (ms)'] / df['Inference (ms)'].max())
    print(f"  Best Tradeoff:               {df.loc[df['Speed-Accuracy Score'].idxmax(), 'Model']}")


if __name__ == "__main__":
    # Example usage (requires trained models and test data)
    print("Model comparison framework loaded.")
    print("To use this module:")
    print("  1. Train models using train.py")
    print("  2. Load models and test data")
    print("  3. Call compare_models(models_dict, X_test, y_test)")
    print("\nExample:")
    print("""
    from analysis.model_comparison import compare_models

    models = {
        'MLP': mlp_model,
        'CNN': cnn_model,
        'ResNet': resnet_model,
        'Ensemble': ensemble
    }

    results_df = compare_models(models, X_test, y_test, param_names=['sigma', 'nu', 'theta'])
    print_comparison_summary(results_df)
    save_comparison_table(results_df)
    """)
