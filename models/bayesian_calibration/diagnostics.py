"""
MCMC diagnostics and posterior analysis tools.

Provides convergence diagnostics, summary statistics, and visualization helpers
for Bayesian calibration results.
"""

import numpy as np
import json
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


def compute_rhat(chains):
    """
    Compute Gelman-Rubin R-hat statistic for convergence diagnosis.

    R-hat compares within-chain and between-chain variance.
    Values close to 1.0 indicate convergence (typically < 1.01 is good).

    Args:
        chains (np.ndarray): MCMC samples with shape (n_chains, n_samples).

    Returns:
        float: R-hat statistic.
    """
    n_chains, n_samples = chains.shape

    # Within-chain variance
    W = np.mean(np.var(chains, axis=1, ddof=1))

    # Between-chain variance
    chain_means = np.mean(chains, axis=1)
    B = n_samples * np.var(chain_means, ddof=1)

    # Variance estimate
    var_hat = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B

    # R-hat
    rhat = np.sqrt(var_hat / W)

    return rhat


def compute_ess(chains):
    """
    Compute Effective Sample Size (ESS).

    ESS accounts for autocorrelation in MCMC samples.
    Higher ESS indicates more independent samples.

    Args:
        chains (np.ndarray): MCMC samples with shape (n_chains, n_samples).

    Returns:
        float: Effective sample size.
    """
    n_chains, n_samples = chains.shape

    # Flatten chains
    samples_flat = chains.flatten()

    # Compute autocorrelation
    def autocorr(x, lag):
        """Compute autocorrelation at given lag."""
        n = len(x)
        if lag >= n:
            return 0.0
        x_mean = np.mean(x)
        c0 = np.sum((x - x_mean) ** 2) / n
        c_lag = np.sum((x[:-lag] - x_mean) * (x[lag:] - x_mean)) / n
        return c_lag / c0

    # Sum autocorrelations until they become negative
    max_lag = min(n_samples // 2, 1000)
    rho = []
    for lag in range(1, max_lag):
        rho_lag = autocorr(samples_flat, lag)
        if rho_lag < 0:
            break
        rho.append(rho_lag)

    # ESS formula
    tau = 1 + 2 * np.sum(rho)
    ess = len(samples_flat) / tau

    return ess


def compute_mcse(chains):
    """
    Compute Monte Carlo Standard Error (MCSE).

    MCSE quantifies uncertainty in posterior mean estimate due to finite MCMC samples.

    Args:
        chains (np.ndarray): MCMC samples.

    Returns:
        float: Monte Carlo standard error.
    """
    ess = compute_ess(chains)
    std = np.std(chains.flatten())
    mcse = std / np.sqrt(ess)
    return mcse


def diagnose_posterior(posterior_samples, param_names=None):
    """
    Compute comprehensive diagnostics for posterior samples.

    Args:
        posterior_samples (dict): Dictionary mapping parameter names to samples.
                                 Each value has shape (n_chains, n_samples).
        param_names (list): List of parameter names to diagnose (default: all).

    Returns:
        dict: Diagnostic statistics for each parameter.
    """
    if param_names is None:
        param_names = list(posterior_samples.keys())

    diagnostics = {}

    for param in param_names:
        samples = posterior_samples[param]

        # Ensure 2D array (chains × samples)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        diagnostics[param] = {
            'rhat': float(compute_rhat(samples)),
            'ess': float(compute_ess(samples)),
            'mcse': float(compute_mcse(samples)),
            'mean': float(np.mean(samples)),
            'std': float(np.std(samples)),
            'n_chains': samples.shape[0],
            'n_samples_per_chain': samples.shape[1],
            'total_samples': samples.size
        }

    return diagnostics


def print_diagnostics_table(diagnostics):
    """
    Print formatted diagnostics table.

    Args:
        diagnostics (dict): Diagnostics from diagnose_posterior().
    """
    print("\n" + "="*80)
    print("MCMC DIAGNOSTICS")
    print("="*80)

    print(f"\n{'Parameter':<10} {'R-hat':<10} {'ESS':<10} {'MCSE':<12} {'Mean':<12} {'Std':<12}")
    print("-"*80)

    for param, stats in diagnostics.items():
        print(f"{param:<10} {stats['rhat']:<10.4f} {stats['ess']:<10.0f} "
              f"{stats['mcse']:<12.6f} {stats['mean']:<12.6f} {stats['std']:<12.6f}")

    print("-"*80)

    # Convergence warnings
    warnings = []
    for param, stats in diagnostics.items():
        if stats['rhat'] > 1.01:
            warnings.append(f"⚠ {param}: R-hat = {stats['rhat']:.4f} > 1.01 (poor convergence)")
        if stats['ess'] < 1000:
            warnings.append(f"⚠ {param}: ESS = {stats['ess']:.0f} < 1000 (low effective samples)")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  {w}")
    else:
        print("\n✓ All diagnostics look good!")


def plot_trace(posterior_samples, param_names=None, save_path=None):
    """
    Plot MCMC trace plots for visual convergence diagnosis.

    Args:
        posterior_samples (dict): Posterior samples.
        param_names (list): Parameters to plot (default: all).
        save_path (str): Path to save figure (optional).
    """
    if param_names is None:
        param_names = list(posterior_samples.keys())

    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 1, figsize=(12, 3*n_params))

    if n_params == 1:
        axes = [axes]

    for i, param in enumerate(param_names):
        samples = posterior_samples[param]

        # Ensure 2D
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        # Plot each chain
        for chain_idx in range(samples.shape[0]):
            axes[i].plot(samples[chain_idx, :], alpha=0.7, linewidth=0.5,
                        label=f'Chain {chain_idx + 1}')

        axes[i].set_ylabel(param)
        axes[i].set_xlabel('Iteration')
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.3)

    plt.suptitle('MCMC Trace Plots', fontsize=14, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trace plots saved to {save_path}")

    return fig


def plot_posterior_distributions(posterior_samples, param_names=None, true_values=None,
                                 save_path=None):
    """
    Plot posterior distributions with credible intervals.

    Args:
        posterior_samples (dict): Posterior samples.
        param_names (list): Parameters to plot.
        true_values (dict): True parameter values for comparison (optional).
        save_path (str): Path to save figure.
    """
    if param_names is None:
        param_names = list(posterior_samples.keys())

    n_params = len(param_names)
    fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 4))

    if n_params == 1:
        axes = [axes]

    for i, param in enumerate(param_names):
        samples = posterior_samples[param].flatten()

        # Histogram
        axes[i].hist(samples, bins=50, density=True, alpha=0.6, edgecolor='black')

        # Credible interval
        hdi_lower = np.percentile(samples, 2.5)
        hdi_upper = np.percentile(samples, 97.5)
        axes[i].axvline(hdi_lower, color='red', linestyle='--', linewidth=2,
                       label=f'95% HDI: [{hdi_lower:.3f}, {hdi_upper:.3f}]')
        axes[i].axvline(hdi_upper, color='red', linestyle='--', linewidth=2)

        # Mean
        mean_val = np.mean(samples)
        axes[i].axvline(mean_val, color='blue', linestyle='-', linewidth=2,
                       label=f'Mean: {mean_val:.3f}')

        # True value if provided
        if true_values and param in true_values:
            axes[i].axvline(true_values[param], color='green', linestyle='-',
                           linewidth=2, label=f'True: {true_values[param]:.3f}')

        axes[i].set_xlabel(param)
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.suptitle('Posterior Distributions', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Posterior distributions saved to {save_path}")

    return fig


def plot_parameter_correlations(posterior_samples, param_names=None, save_path=None):
    """
    Plot pairwise parameter correlations (corner plot).

    Args:
        posterior_samples (dict): Posterior samples.
        param_names (list): Parameters to plot.
        save_path (str): Path to save figure.
    """
    if param_names is None:
        param_names = list(posterior_samples.keys())

    # Flatten samples
    samples_dict = {param: posterior_samples[param].flatten() for param in param_names}

    # Create DataFrame for seaborn
    import pandas as pd
    df = pd.DataFrame(samples_dict)

    # Pair plot
    g = sns.pairplot(df, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 5})
    g.fig.suptitle('Parameter Correlations', y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation plot saved to {save_path}")

    return g


def save_diagnostics_report(posterior_samples, output_dir='outputs/bayesian'):
    """
    Generate and save comprehensive diagnostics report.

    Args:
        posterior_samples (dict): Posterior samples.
        output_dir (str): Output directory.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Compute diagnostics
    diagnostics = diagnose_posterior(posterior_samples)

    # Print table
    print_diagnostics_table(diagnostics)

    # Save JSON
    with open(f'{output_dir}/diagnostics.json', 'w') as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\nDiagnostics saved to {output_dir}/diagnostics.json")

    # Generate plots
    plot_trace(posterior_samples, save_path=f'{output_dir}/trace_plots.png')
    plot_posterior_distributions(posterior_samples, save_path=f'{output_dir}/posteriors.png')
    plot_parameter_correlations(posterior_samples, save_path=f'{output_dir}/correlations.png')

    print(f"\n✓ Full diagnostics report saved to {output_dir}/")


if __name__ == "__main__":
    print("="*80)
    print("BAYESIAN DIAGNOSTICS")
    print("="*80)
    print("\nThis module provides MCMC diagnostics and visualization tools.")
    print("\nKey functions:")
    print("  - compute_rhat(): Gelman-Rubin convergence diagnostic")
    print("  - compute_ess(): Effective sample size")
    print("  - diagnose_posterior(): Comprehensive diagnostics")
    print("  - plot_trace(): Trace plots for visual inspection")
    print("  - plot_posterior_distributions(): Posterior histograms with HDI")
    print("  - save_diagnostics_report(): Generate full report")
