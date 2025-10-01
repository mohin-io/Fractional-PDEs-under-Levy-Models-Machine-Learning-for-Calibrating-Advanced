"""
Robustness testing framework for calibration models.

Tests model performance under:
- Input noise (Gaussian, uniform, adversarial)
- Out-of-distribution detection
- Extreme parameter values
- Missing data patterns
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


def test_noise_robustness(model, X_test, y_test, param_names,
                          noise_levels=[0.01, 0.05, 0.10], noise_type='gaussian'):
    """
    Test model robustness to input noise.

    Args:
        model: Trained model.
        X_test (np.ndarray): Clean test features.
        y_test (np.ndarray): Test targets.
        param_names (list): Parameter names.
        noise_levels (list): Noise levels (as fractions).
        noise_type (str): 'gaussian', 'uniform', or 'salt_pepper'.

    Returns:
        pd.DataFrame: Results for each noise level.
    """
    results = []

    # Baseline (no noise)
    y_pred_clean = model.predict(X_test, verbose=0)
    mse_clean = mean_squared_error(y_test, y_pred_clean)
    mae_clean = mean_absolute_error(y_test, y_pred_clean)

    results.append({
        'Noise Level': '0% (Clean)',
        'MSE': mse_clean,
        'MAE': mae_clean,
        'MSE Increase': 0.0,
        'MAE Increase': 0.0,
        'MSE Increase %': 0.0,
        'MAE Increase %': 0.0
    })

    print(f"Testing robustness to {noise_type} noise...")
    print(f"Baseline (clean) - MSE: {mse_clean:.6f}, MAE: {mae_clean:.6f}")

    for noise_level in noise_levels:
        print(f"\n  Noise level: {noise_level*100:.0f}%")

        # Add noise
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level * np.abs(X_test), X_test.shape)
        elif noise_type == 'uniform':
            noise = np.random.uniform(-noise_level, noise_level, X_test.shape) * np.abs(X_test)
        elif noise_type == 'salt_pepper':
            # Random spikes
            mask = np.random.binomial(1, noise_level, X_test.shape)
            noise = mask * np.random.choice([-1, 1], X_test.shape) * np.abs(X_test)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        X_noisy = X_test + noise

        # Predict
        y_pred_noisy = model.predict(X_noisy, verbose=0)

        # Metrics
        mse_noisy = mean_squared_error(y_test, y_pred_noisy)
        mae_noisy = mean_absolute_error(y_test, y_pred_noisy)

        mse_increase = mse_noisy - mse_clean
        mae_increase = mae_noisy - mae_clean

        results.append({
            'Noise Level': f'{noise_level*100:.0f}%',
            'MSE': mse_noisy,
            'MAE': mae_noisy,
            'MSE Increase': mse_increase,
            'MAE Increase': mae_increase,
            'MSE Increase %': (mse_increase / mse_clean) * 100,
            'MAE Increase %': (mae_increase / mae_clean) * 100
        })

        print(f"    MSE: {mse_noisy:.6f} (+{(mse_increase/mse_clean)*100:.1f}%)")
        print(f"    MAE: {mae_noisy:.6f} (+{(mae_increase/mae_clean)*100:.1f}%)")

    return pd.DataFrame(results)


def detect_out_of_distribution(X_train, X_test, threshold_percentile=95):
    """
    Detect out-of-distribution samples using Mahalanobis distance.

    Args:
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Test features.
        threshold_percentile (float): Percentile for OOD threshold.

    Returns:
        dict: OOD detection results.
    """
    from scipy.spatial.distance import mahalanobis

    # Compute mean and covariance of training data
    mean_train = np.mean(X_train, axis=0)
    cov_train = np.cov(X_train, rowvar=False)

    # Add regularization to avoid singular matrix
    cov_train += np.eye(cov_train.shape[0]) * 1e-6
    cov_inv = np.linalg.inv(cov_train)

    # Compute Mahalanobis distance for test samples
    distances = np.array([
        mahalanobis(x, mean_train, cov_inv) for x in X_test
    ])

    # Threshold from training data
    train_distances = np.array([
        mahalanobis(x, mean_train, cov_inv) for x in X_train[:1000]  # Sample for speed
    ])
    threshold = np.percentile(train_distances, threshold_percentile)

    # Flag OOD samples
    is_ood = distances > threshold

    return {
        'distances': distances,
        'threshold': threshold,
        'is_ood': is_ood,
        'n_ood': np.sum(is_ood),
        'pct_ood': (np.sum(is_ood) / len(distances)) * 100
    }


def test_extreme_values(model, X_test, y_test, param_names, scale_factors=[0.5, 2.0, 5.0]):
    """
    Test model on extreme input values (scaled).

    Args:
        model: Trained model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test targets.
        param_names (list): Parameter names.
        scale_factors (list): Scaling factors to apply.

    Returns:
        pd.DataFrame: Results for each scale factor.
    """
    results = []

    # Baseline
    y_pred_base = model.predict(X_test, verbose=0)
    mse_base = mean_squared_error(y_test, y_pred_base)

    results.append({
        'Scale Factor': '1.0x (Normal)',
        'MSE': mse_base,
        'MAE': mean_absolute_error(y_test, y_pred_base)
    })

    for scale in scale_factors:
        X_scaled = X_test * scale

        y_pred_scaled = model.predict(X_scaled, verbose=0)
        mse_scaled = mean_squared_error(y_test, y_pred_scaled)
        mae_scaled = mean_absolute_error(y_test, y_pred_scaled)

        results.append({
            'Scale Factor': f'{scale}x',
            'MSE': mse_scaled,
            'MAE': mae_scaled
        })

    return pd.DataFrame(results)


def test_missing_data(model, X_test, y_test, param_names, missing_pcts=[0.05, 0.10, 0.20]):
    """
    Test model robustness to missing data.

    Randomly set features to zero (simulating missing prices).

    Args:
        model: Trained model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test targets.
        param_names (list): Parameter names.
        missing_pcts (list): Percentages of missing features.

    Returns:
        pd.DataFrame: Results for each missing percentage.
    """
    results = []

    # Baseline
    y_pred_base = model.predict(X_test, verbose=0)
    mse_base = mean_squared_error(y_test, y_pred_base)

    results.append({
        'Missing %': '0%',
        'MSE': mse_base,
        'MAE': mean_absolute_error(y_test, y_pred_base)
    })

    for missing_pct in missing_pcts:
        X_missing = X_test.copy()

        # Randomly set features to zero
        n_features = X_test.shape[1]
        n_missing = int(missing_pct * n_features)

        for i in range(len(X_test)):
            missing_idx = np.random.choice(n_features, n_missing, replace=False)
            X_missing[i, missing_idx] = 0

        y_pred_missing = model.predict(X_missing, verbose=0)
        mse_missing = mean_squared_error(y_test, y_pred_missing)
        mae_missing = mean_absolute_error(y_test, y_pred_missing)

        results.append({
            'Missing %': f'{missing_pct*100:.0f}%',
            'MSE': mse_missing,
            'MAE': mae_missing
        })

    return pd.DataFrame(results)


def plot_robustness_results(noise_results, extreme_results=None, missing_results=None,
                            save_path=None):
    """
    Plot robustness test results.

    Args:
        noise_results (pd.DataFrame): From test_noise_robustness().
        extreme_results (pd.DataFrame): From test_extreme_values() (optional).
        missing_results (pd.DataFrame): From test_missing_data() (optional).
        save_path (str): Save path (optional).
    """
    n_plots = 1 + (extreme_results is not None) + (missing_results is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Noise robustness
    ax = axes[plot_idx]
    ax.plot(range(len(noise_results)), noise_results['MSE'], 'o-', linewidth=2, markersize=8)
    ax.set_xticks(range(len(noise_results)))
    ax.set_xticklabels(noise_results['Noise Level'], rotation=45)
    ax.set_ylabel('MSE')
    ax.set_title('Robustness to Input Noise')
    ax.grid(True, alpha=0.3)
    plot_idx += 1

    # Extreme values
    if extreme_results is not None:
        ax = axes[plot_idx]
        ax.plot(range(len(extreme_results)), extreme_results['MSE'], 's-',
               linewidth=2, markersize=8, color='orange')
        ax.set_xticks(range(len(extreme_results)))
        ax.set_xticklabels(extreme_results['Scale Factor'], rotation=45)
        ax.set_ylabel('MSE')
        ax.set_title('Robustness to Extreme Values')
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Missing data
    if missing_results is not None:
        ax = axes[plot_idx]
        ax.plot(range(len(missing_results)), missing_results['MSE'], '^-',
               linewidth=2, markersize=8, color='red')
        ax.set_xticks(range(len(missing_results)))
        ax.set_xticklabels(missing_results['Missing %'], rotation=45)
        ax.set_ylabel('MSE')
        ax.set_title('Robustness to Missing Data')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Robustness Tests', fontsize=14)
    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Robustness results saved to {save_path}")

    return fig


if __name__ == "__main__":
    print("="*80)
    print("ROBUSTNESS TESTING FRAMEWORK")
    print("="*80)
    print("\nThis module tests model robustness under various conditions.")
    print("\nKey tests:")
    print("  - test_noise_robustness(): Gaussian/uniform/salt-pepper noise")
    print("  - detect_out_of_distribution(): Mahalanobis distance OOD detection")
    print("  - test_extreme_values(): Scaled inputs")
    print("  - test_missing_data(): Random feature dropout")
    print("\nThese tests help identify model weaknesses and deployment risks.")
