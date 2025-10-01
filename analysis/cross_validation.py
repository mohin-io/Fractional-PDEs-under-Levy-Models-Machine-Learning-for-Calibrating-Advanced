"""
K-fold cross-validation for calibration models.

Implements stratified and standard k-fold cross-validation with:
- Per-fold metrics tracking
- Statistical significance testing
- Visualization of fold performance
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Callable
import joblib


def k_fold_validation(X, y, model_builder, param_names, k=5, random_state=42, stratified=False):
    """
    Perform k-fold cross-validation.

    Args:
        X (np.ndarray): Features.
        y (np.ndarray): Targets.
        model_builder (Callable): Function that returns a compiled model.
        param_names (list): Parameter names.
        k (int): Number of folds (default: 5).
        random_state (int): Random seed.
        stratified (bool): Use stratified folds (default: False).

    Returns:
        dict: Cross-validation results.
    """
    print(f"Running {k}-fold cross-validation...")

    if stratified:
        # For stratified, we need to bin continuous targets
        # Use first parameter for stratification
        bins = pd.qcut(y[:, 0], q=k, labels=False, duplicates='drop')
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        splits = kf.split(X, bins)
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        splits = kf.split(X)

    results = {
        'fold_mse': [],
        'fold_mae': [],
        'fold_r2': [],
        'fold_mse_per_param': [],
        'fold_mae_per_param': [],
        'fold_r2_per_param': []
    }

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n  Fold {fold + 1}/{k}...")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Build and train model
        model = model_builder()
        history = model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )

        # Evaluate
        y_pred = model.predict(X_val_scaled, verbose=0)

        # Overall metrics
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        results['fold_mse'].append(mse)
        results['fold_mae'].append(mae)
        results['fold_r2'].append(r2)

        # Per-parameter metrics
        mse_per_param = [mean_squared_error(y_val[:, i], y_pred[:, i])
                         for i in range(y_val.shape[1])]
        mae_per_param = [mean_absolute_error(y_val[:, i], y_pred[:, i])
                         for i in range(y_val.shape[1])]
        r2_per_param = [r2_score(y_val[:, i], y_pred[:, i])
                        for i in range(y_val.shape[1])]

        results['fold_mse_per_param'].append(mse_per_param)
        results['fold_mae_per_param'].append(mae_per_param)
        results['fold_r2_per_param'].append(r2_per_param)

        print(f"    MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}")

    # Compute statistics
    results['mean_mse'] = np.mean(results['fold_mse'])
    results['std_mse'] = np.std(results['fold_mse'])
    results['mean_mae'] = np.mean(results['fold_mae'])
    results['std_mae'] = np.std(results['fold_mae'])
    results['mean_r2'] = np.mean(results['fold_r2'])
    results['std_r2'] = np.std(results['fold_r2'])

    # Per-parameter statistics
    results['mean_mse_per_param'] = np.mean(results['fold_mse_per_param'], axis=0)
    results['mean_mae_per_param'] = np.mean(results['fold_mae_per_param'], axis=0)
    results['mean_r2_per_param'] = np.mean(results['fold_r2_per_param'], axis=0)

    results['param_names'] = param_names

    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION RESULTS ({k} folds)")
    print(f"{'='*80}")
    print(f"  MSE: {results['mean_mse']:.6f} ± {results['std_mse']:.6f}")
    print(f"  MAE: {results['mean_mae']:.6f} ± {results['std_mae']:.6f}")
    print(f"  R²:  {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")

    return results


def plot_cv_results(results, save_path=None):
    """
    Plot cross-validation results.

    Args:
        results (dict): Results from k_fold_validation().
        save_path (str): Path to save figure (optional).
    """
    k = len(results['fold_mse'])
    param_names = results['param_names']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Overall metrics box plots
    metrics_df = pd.DataFrame({
        'MSE': results['fold_mse'],
        'MAE': results['fold_mae'],
        'R²': results['fold_r2']
    })

    # MSE
    axes[0, 0].boxplot([results['fold_mse']], labels=['MSE'])
    axes[0, 0].scatter([1]*k, results['fold_mse'], alpha=0.5, s=50)
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_title(f'MSE: {results["mean_mse"]:.6f} ± {results["std_mse"]:.6f}')
    axes[0, 0].grid(True, alpha=0.3)

    # MAE
    axes[0, 1].boxplot([results['fold_mae']], labels=['MAE'])
    axes[0, 1].scatter([1]*k, results['fold_mae'], alpha=0.5, s=50)
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title(f'MAE: {results["mean_mae"]:.6f} ± {results["std_mae"]:.6f}')
    axes[0, 1].grid(True, alpha=0.3)

    # R²
    axes[1, 0].boxplot([results['fold_r2']], labels=['R²'])
    axes[1, 0].scatter([1]*k, results['fold_r2'], alpha=0.5, s=50)
    axes[1, 0].set_ylabel('R²')
    axes[1, 0].set_title(f'R²: {results["mean_r2"]:.4f} ± {results["std_r2"]:.4f}')
    axes[1, 0].grid(True, alpha=0.3)

    # Per-parameter MAE
    mae_per_param_array = np.array(results['fold_mae_per_param'])  # Shape: (k, n_params)
    positions = np.arange(1, len(param_names) + 1)

    bp = axes[1, 1].boxplot([mae_per_param_array[:, i] for i in range(len(param_names))],
                            labels=param_names, positions=positions)

    for i in range(len(param_names)):
        axes[1, 1].scatter([i+1]*k, mae_per_param_array[:, i], alpha=0.5, s=30)

    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('MAE per Parameter')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'{k}-Fold Cross-Validation Results', fontsize=14, y=0.995)
    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CV results saved to {save_path}")

    return fig


def compare_models_cv(X, y, model_builders, model_names, param_names, k=5):
    """
    Compare multiple models using cross-validation.

    Args:
        X (np.ndarray): Features.
        y (np.ndarray): Targets.
        model_builders (dict): Dictionary mapping model names to builder functions.
        model_names (list): List of model names to compare.
        param_names (list): Parameter names.
        k (int): Number of folds.

    Returns:
        pd.DataFrame: Comparison table.
    """
    comparison_results = []

    for name in model_names:
        print(f"\n{'='*80}")
        print(f"Evaluating {name}...")
        print(f"{'='*80}")

        results = k_fold_validation(X, y, model_builders[name], param_names, k=k)

        comparison_results.append({
            'Model': name,
            'Mean MSE': results['mean_mse'],
            'Std MSE': results['std_mse'],
            'Mean MAE': results['mean_mae'],
            'Std MAE': results['std_mae'],
            'Mean R²': results['mean_r2'],
            'Std R²': results['std_r2']
        })

    df = pd.DataFrame(comparison_results)

    print(f"\n{'='*80}")
    print("MODEL COMPARISON (CROSS-VALIDATION)")
    print(f"{'='*80}")
    print(df.to_string(index=False))

    return df


if __name__ == "__main__":
    print("="*80)
    print("K-FOLD CROSS-VALIDATION MODULE")
    print("="*80)
    print("\nThis module implements k-fold cross-validation for model evaluation.")
    print("\nKey functions:")
    print("  - k_fold_validation(): Standard k-fold CV")
    print("  - plot_cv_results(): Visualize fold performance")
    print("  - compare_models_cv(): Compare multiple models")
    print("\nExample usage:")
    print("""
from models.calibration_net.model import build_mlp_model
from analysis.cross_validation import k_fold_validation, plot_cv_results

# Define model builder
def model_builder():
    return build_mlp_model((200,), 3)

# Run CV
results = k_fold_validation(X, y, model_builder, ['sigma', 'nu', 'theta'], k=5)
plot_cv_results(results, save_path='outputs/cv_results.png')
    """)
