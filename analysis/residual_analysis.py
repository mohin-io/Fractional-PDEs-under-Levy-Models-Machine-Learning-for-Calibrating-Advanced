"""
Comprehensive residual analysis for model diagnostics.

Provides statistical tests and visualizations to assess model quality:
- Normality tests (Shapiro-Wilk, Kolmogorov-Smirnov)
- Q-Q plots
- Residuals vs fitted values
- Autocorrelation analysis
- Heteroscedasticity tests
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, kstest, normaltest
import pandas as pd
import os


def compute_residuals(y_true, y_pred):
    """
    Compute residuals and standardized residuals.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        dict: Residuals and standardized residuals.
    """
    residuals = y_true - y_pred
    std_residuals = residuals / np.std(residuals, axis=0, keepdims=True)

    return {
        'residuals': residuals,
        'std_residuals': std_residuals
    }


def test_normality(residuals, param_names):
    """
    Test residual normality using multiple tests.

    Args:
        residuals (np.ndarray): Residuals array (n_samples, n_params).
        param_names (list): Parameter names.

    Returns:
        pd.DataFrame: Test results for each parameter.
    """
    results = []

    for i, param in enumerate(param_names):
        res = residuals[:, i]

        # Shapiro-Wilk test (good for small-medium samples)
        shapiro_stat, shapiro_p = shapiro(res)

        # Kolmogorov-Smirnov test
        ks_stat, ks_p = kstest(res, 'norm', args=(np.mean(res), np.std(res)))

        # D'Agostino-Pearson test
        dagostino_stat, dagostino_p = normaltest(res)

        results.append({
            'Parameter': param,
            'Shapiro-Wilk Stat': shapiro_stat,
            'Shapiro-Wilk p-value': shapiro_p,
            'KS Stat': ks_stat,
            'KS p-value': ks_p,
            "D'Agostino Stat": dagostino_stat,
            "D'Agostino p-value": dagostino_p,
            'Normal (α=0.05)': (shapiro_p > 0.05) and (ks_p > 0.05)
        })

    return pd.DataFrame(results)


def plot_residual_diagnostics(y_true, y_pred, param_names, save_path=None):
    """
    Create comprehensive residual diagnostic plots.

    For each parameter:
    - Q-Q plot (normality check)
    - Residuals vs fitted values (homoscedasticity check)
    - Histogram of residuals (distribution check)

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        param_names (list): Parameter names.
        save_path (str): Path to save figure (optional).
    """
    residuals = y_true - y_pred
    n_params = len(param_names)

    fig, axes = plt.subplots(n_params, 3, figsize=(15, 4*n_params))

    if n_params == 1:
        axes = axes.reshape(1, -1)

    for i, param in enumerate(param_names):
        res = residuals[:, i]
        fitted = y_pred[:, i]

        # Q-Q plot
        stats.probplot(res, dist="norm", plot=axes[i, 0])
        axes[i, 0].set_title(f'Q-Q Plot: {param}')
        axes[i, 0].grid(True, alpha=0.3)

        # Residuals vs Fitted
        axes[i, 1].scatter(fitted, res, alpha=0.3, s=10)
        axes[i, 1].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[i, 1].set_xlabel(f'Fitted {param}')
        axes[i, 1].set_ylabel('Residuals')
        axes[i, 1].set_title(f'Residuals vs Fitted: {param}')
        axes[i, 1].grid(True, alpha=0.3)

        # Add LOESS smoother
        try:
            from scipy.signal import savgol_filter
            sorted_idx = np.argsort(fitted)
            smoothed = savgol_filter(res[sorted_idx], window_length=min(51, len(res)//10*2+1), polyorder=3)
            axes[i, 1].plot(fitted[sorted_idx], smoothed, color='blue', linewidth=2, label='Trend')
            axes[i, 1].legend()
        except:
            pass

        # Histogram
        axes[i, 2].hist(res, bins=50, density=True, alpha=0.7, edgecolor='black')

        # Overlay normal distribution
        mu, sigma = np.mean(res), np.std(res)
        x = np.linspace(res.min(), res.max(), 100)
        axes[i, 2].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')
        axes[i, 2].set_xlabel('Residuals')
        axes[i, 2].set_ylabel('Density')
        axes[i, 2].set_title(f'Residual Distribution: {param}')
        axes[i, 2].legend()
        axes[i, 2].grid(True, alpha=0.3)

    plt.suptitle('Residual Diagnostics', fontsize=14, y=0.995)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residual diagnostics saved to {save_path}")

    return fig


def compute_autocorrelation(residuals, max_lag=20):
    """
    Compute autocorrelation function for residuals.

    Args:
        residuals (np.ndarray): Residuals (1D array).
        max_lag (int): Maximum lag to compute.

    Returns:
        np.ndarray: Autocorrelation values for each lag.
    """
    n = len(residuals)
    mean = np.mean(residuals)
    c0 = np.sum((residuals - mean) ** 2) / n

    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0

    for lag in range(1, max_lag + 1):
        c_lag = np.sum((residuals[:-lag] - mean) * (residuals[lag:] - mean)) / n
        acf[lag] = c_lag / c0

    return acf


def plot_autocorrelation(residuals, param_names, max_lag=20, save_path=None):
    """
    Plot autocorrelation function for residuals.

    Args:
        residuals (np.ndarray): Residuals (n_samples, n_params).
        param_names (list): Parameter names.
        max_lag (int): Maximum lag.
        save_path (str): Save path (optional).
    """
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 3*n_params))

    if n_params == 1:
        axes = [axes]

    for i, param in enumerate(param_names):
        acf = compute_autocorrelation(residuals[:, i], max_lag)
        lags = np.arange(len(acf))

        axes[i].stem(lags, acf, basefmt=' ')
        axes[i].axhline(0, color='black', linewidth=0.8)

        # Confidence interval (95%)
        conf_int = 1.96 / np.sqrt(len(residuals))
        axes[i].axhline(conf_int, color='blue', linestyle='--', linewidth=1, label='95% CI')
        axes[i].axhline(-conf_int, color='blue', linestyle='--', linewidth=1)

        axes[i].set_xlabel('Lag')
        axes[i].set_ylabel('ACF')
        axes[i].set_title(f'Autocorrelation: {param}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Autocorrelation plot saved to {save_path}")

    return fig


def test_heteroscedasticity(y_pred, residuals, param_names):
    """
    Breusch-Pagan test for heteroscedasticity.

    H0: Homoscedastic (constant variance)
    H1: Heteroscedastic (variance depends on fitted values)

    Args:
        y_pred (np.ndarray): Predicted values.
        residuals (np.ndarray): Residuals.
        param_names (list): Parameter names.

    Returns:
        pd.DataFrame: Test results.
    """
    from scipy.stats import chi2

    results = []

    for i, param in enumerate(param_names):
        fitted = y_pred[:, i]
        res = residuals[:, i]

        # Squared residuals
        res_sq = res ** 2

        # Fit linear model: res_sq ~ fitted
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(fitted.reshape(-1, 1), res_sq)

        # Predicted squared residuals
        res_sq_pred = model.predict(fitted.reshape(-1, 1))

        # Test statistic
        SSR = np.sum((res_sq_pred - np.mean(res_sq)) ** 2)
        SST = np.sum((res_sq - np.mean(res_sq)) ** 2)
        R_sq = SSR / SST

        n = len(fitted)
        p = 1  # Number of predictors

        LM = n * R_sq  # Lagrange Multiplier statistic
        p_value = 1 - chi2.cdf(LM, p)

        results.append({
            'Parameter': param,
            'LM Statistic': LM,
            'p-value': p_value,
            'Homoscedastic (α=0.05)': p_value > 0.05
        })

    return pd.DataFrame(results)


def print_diagnostic_summary(y_true, y_pred, param_names):
    """
    Print comprehensive diagnostic summary.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        param_names (list): Parameter names.
    """
    residuals = y_true - y_pred

    print("\n" + "="*80)
    print("RESIDUAL DIAGNOSTICS SUMMARY")
    print("="*80)

    # Normality tests
    print("\n--- Normality Tests ---")
    normality_results = test_normality(residuals, param_names)
    print(normality_results.to_string(index=False))

    # Heteroscedasticity test
    print("\n--- Heteroscedasticity Tests (Breusch-Pagan) ---")
    hetero_results = test_heteroscedasticity(y_pred, residuals, param_names)
    print(hetero_results.to_string(index=False))

    # Summary statistics
    print("\n--- Residual Summary Statistics ---")
    for i, param in enumerate(param_names):
        res = residuals[:, i]
        print(f"\n{param}:")
        print(f"  Mean:     {np.mean(res):>10.6f} (should be ≈ 0)")
        print(f"  Std:      {np.std(res):>10.6f}")
        print(f"  Min:      {np.min(res):>10.6f}")
        print(f"  Max:      {np.max(res):>10.6f}")
        print(f"  Skewness: {stats.skew(res):>10.6f} (should be ≈ 0)")
        print(f"  Kurtosis: {stats.kurtosis(res):>10.6f} (should be ≈ 0 for normal)")


if __name__ == "__main__":
    print("="*80)
    print("RESIDUAL ANALYSIS MODULE")
    print("="*80)
    print("\nThis module provides comprehensive residual diagnostics.")
    print("\nKey functions:")
    print("  - test_normality(): Shapiro-Wilk, KS, D'Agostino tests")
    print("  - plot_residual_diagnostics(): Q-Q, res vs fitted, histograms")
    print("  - plot_autocorrelation(): ACF plots")
    print("  - test_heteroscedasticity(): Breusch-Pagan test")
    print("  - print_diagnostic_summary(): Complete report")
