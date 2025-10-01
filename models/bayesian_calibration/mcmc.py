"""
Bayesian calibration using MCMC (Hamiltonian Monte Carlo).

This module provides full Bayesian inference for Lévy model parameters using
TensorFlow Probability's No-U-Turn Sampler (NUTS), enabling:
- Full posterior distributions (not just point estimates)
- Credible intervals for uncertainty quantification
- Parameter correlations
- Model comparison via Bayes factors
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import argparse
import json
import os
from datetime import datetime

tfd = tfp.distributions
tfb = tfp.bijectors

# Import pricing functions
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.pricing_engine.fourier_pricer import price_surface


class BayesianCalibrator:
    """
    Bayesian calibration using Hamiltonian Monte Carlo.

    This class implements full Bayesian inference for Lévy model parameters,
    providing posterior distributions and uncertainty quantification.
    """

    def __init__(self, model_name='VarianceGamma', S0=100.0, r=0.05, q=0.0):
        """
        Initialize Bayesian calibrator.

        Args:
            model_name (str): 'VarianceGamma' or 'CGMY'.
            S0 (float): Spot price.
            r (float): Risk-free rate.
            q (float): Dividend yield.
        """
        self.model_name = model_name
        self.S0 = S0
        self.r = r
        self.q = q
        self.posterior_samples = None
        self.diagnostics = None

    def build_prior(self):
        """
        Build prior distributions for model parameters.

        Priors are based on financial domain knowledge:
        - Volatility (sigma): LogNormal centered around 20% annual vol
        - Kurtosis (nu): Gamma distribution (positive values)
        - Skew (theta): Normal distribution (typically negative for equities)

        Returns:
            tfd.JointDistribution: Joint prior distribution.
        """
        if self.model_name == 'VarianceGamma':
            # Prior for sigma: LogNormal(log(0.2), 0.5)
            # Prior for nu: Gamma(2, 2) -> mean=1, variance=0.5
            # Prior for theta: Normal(-0.2, 0.2)
            return tfd.JointDistributionNamed({
                'sigma': tfd.LogNormal(loc=np.log(0.2), scale=0.5),
                'nu': tfd.Gamma(concentration=2.0, rate=2.0),
                'theta': tfd.Normal(loc=-0.2, scale=0.2)
            })
        elif self.model_name == 'CGMY':
            return tfd.JointDistributionNamed({
                'C': tfd.LogNormal(loc=np.log(0.1), scale=0.5),
                'G': tfd.Gamma(concentration=2.0, rate=0.5),
                'M': tfd.Gamma(concentration=2.0, rate=0.5),
                'Y': tfd.TruncatedNormal(loc=1.0, scale=0.3, low=0.1, high=1.9)
            })
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    @tf.function
    def log_likelihood(self, params, observed_prices, strikes, maturities):
        """
        Compute log-likelihood of observed prices given parameters.

        Likelihood: observed_prices ~ Normal(model_prices(params), sigma_obs)

        Args:
            params (dict): Model parameters.
            observed_prices (tf.Tensor): Observed option prices (flattened).
            strikes (np.ndarray): Strike prices.
            maturities (np.ndarray): Maturities.

        Returns:
            tf.Tensor: Log-likelihood value.
        """
        # Price options using the model
        params_dict = {k: v.numpy() if hasattr(v, 'numpy') else v
                      for k, v in params.items()}

        # Use numpy for pricing (TF doesn't support complex FFT well)
        model_prices = price_surface(
            params=params_dict,
            model_name=self.model_name,
            s0=self.S0,
            grid_strikes=strikes,
            grid_maturities=maturities,
            r=self.r,
            q=self.q
        ).flatten()

        model_prices = tf.convert_to_tensor(model_prices, dtype=tf.float32)
        observed_prices = tf.convert_to_tensor(observed_prices, dtype=tf.float32)

        # Observation noise (estimate from bid-ask spread, typically 0.5-1% of price)
        sigma_obs = 0.01 * tf.reduce_mean(observed_prices)

        # Log-likelihood: sum of log P(observed | model)
        log_prob = tfd.Normal(loc=model_prices, scale=sigma_obs).log_prob(observed_prices)

        return tf.reduce_sum(log_prob)

    def target_log_prob_fn(self, observed_prices, strikes, maturities):
        """
        Create target log probability function for MCMC.

        Target = log(prior) + log(likelihood)

        Args:
            observed_prices (np.ndarray): Observed prices.
            strikes (np.ndarray): Strikes.
            maturities (np.ndarray): Maturities.

        Returns:
            Callable: Function that computes target log prob.
        """
        def target_log_prob(*params):
            if self.model_name == 'VarianceGamma':
                sigma, nu, theta = params
                params_dict = {'sigma': sigma, 'nu': nu, 'theta': theta}
            else:  # CGMY
                C, G, M, Y = params
                params_dict = {'C': C, 'G': G, 'M': M, 'Y': Y}

            # Prior log probability
            log_prior = self.build_prior().log_prob(params_dict)

            # Likelihood log probability
            log_lik = self.log_likelihood(params_dict, observed_prices, strikes, maturities)

            return log_prior + log_lik

        return target_log_prob

    def run_mcmc(self, observed_prices, strikes, maturities,
                 num_samples=5000, num_burnin=2000, num_chains=4,
                 initial_state=None):
        """
        Run MCMC sampling using No-U-Turn Sampler (NUTS).

        Args:
            observed_prices (np.ndarray): Observed option prices.
            strikes (np.ndarray): Strike prices.
            maturities (np.ndarray): Maturities.
            num_samples (int): Number of samples per chain.
            num_burnin (int): Number of burn-in samples.
            num_chains (int): Number of parallel chains.
            initial_state (list): Initial parameter values.

        Returns:
            dict: Posterior samples and diagnostics.
        """
        print(f"Running MCMC for {self.model_name} model...")
        print(f"  Chains: {num_chains}, Samples: {num_samples}, Burn-in: {num_burnin}")

        # Flatten observed prices
        observed_prices_flat = observed_prices.flatten()

        # Target log probability function
        target_log_prob = self.target_log_prob_fn(observed_prices_flat, strikes, maturities)

        # Initial state
        if initial_state is None:
            if self.model_name == 'VarianceGamma':
                initial_state = [
                    tf.ones(num_chains, dtype=tf.float32) * 0.2,  # sigma
                    tf.ones(num_chains, dtype=tf.float32) * 0.5,  # nu
                    tf.ones(num_chains, dtype=tf.float32) * -0.1  # theta
                ]
            else:  # CGMY
                initial_state = [
                    tf.ones(num_chains, dtype=tf.float32) * 0.1,  # C
                    tf.ones(num_chains, dtype=tf.float32) * 5.0,  # G
                    tf.ones(num_chains, dtype=tf.float32) * 5.0,  # M
                    tf.ones(num_chains, dtype=tf.float32) * 1.0   # Y
                ]

        # NUTS kernel
        kernel = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=target_log_prob,
            step_size=0.01
        )

        # Adaptive step size
        kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=kernel,
            num_adaptation_steps=int(0.8 * num_burnin),
            target_accept_prob=0.8
        )

        # Run MCMC
        print("Sampling...")
        samples, kernel_results = tfp.mcmc.sample_chain(
            num_results=num_samples,
            num_burnin_steps=num_burnin,
            current_state=initial_state,
            kernel=kernel,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted
        )

        # Extract samples
        if self.model_name == 'VarianceGamma':
            param_names = ['sigma', 'nu', 'theta']
        else:
            param_names = ['C', 'G', 'M', 'Y']

        self.posterior_samples = {
            name: samples[i].numpy() for i, name in enumerate(param_names)
        }

        # Compute diagnostics
        acceptance_rate = tf.reduce_mean(tf.cast(kernel_results, tf.float32)).numpy()

        self.diagnostics = {
            'acceptance_rate': float(acceptance_rate),
            'num_chains': num_chains,
            'num_samples': num_samples
        }

        print(f"✓ MCMC completed. Acceptance rate: {acceptance_rate:.2%}")

        return self.posterior_samples, self.diagnostics

    def summarize_posterior(self):
        """
        Compute posterior summary statistics.

        Returns:
            dict: Summary statistics for each parameter.
        """
        if self.posterior_samples is None:
            raise ValueError("No posterior samples. Run MCMC first.")

        summary = {}
        for param_name, samples in self.posterior_samples.items():
            # Flatten across chains
            samples_flat = samples.flatten()

            summary[param_name] = {
                'mean': float(np.mean(samples_flat)),
                'std': float(np.std(samples_flat)),
                'median': float(np.median(samples_flat)),
                '2.5%': float(np.percentile(samples_flat, 2.5)),
                '97.5%': float(np.percentile(samples_flat, 97.5)),
                'hdi_95_lower': float(np.percentile(samples_flat, 2.5)),
                'hdi_95_upper': float(np.percentile(samples_flat, 97.5))
            }

        return summary

    def save_results(self, output_path='models/bayesian_calibration/mcmc_results.json'):
        """
        Save posterior samples and diagnostics to file.

        Args:
            output_path (str): Output file path.
        """
        if self.posterior_samples is None:
            raise ValueError("No results to save. Run MCMC first.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        results = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'summary': self.summarize_posterior(),
            'diagnostics': self.diagnostics,
            'posterior_samples': {k: v.tolist() for k, v in self.posterior_samples.items()}
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {output_path}")


def run_bayesian_calibration(observed_prices, strikes, maturities,
                            model_name='VarianceGamma', num_samples=5000,
                            num_burnin=2000, num_chains=4):
    """
    Convenience function to run Bayesian calibration.

    Args:
        observed_prices (np.ndarray): Observed option prices (2D array).
        strikes (np.ndarray): Strike prices.
        maturities (np.ndarray): Maturities.
        model_name (str): 'VarianceGamma' or 'CGMY'.
        num_samples (int): MCMC samples per chain.
        num_burnin (int): Burn-in samples.
        num_chains (int): Number of chains.

    Returns:
        dict: Posterior samples.
        dict: Summary statistics.
    """
    calibrator = BayesianCalibrator(model_name=model_name)
    posterior, diagnostics = calibrator.run_mcmc(
        observed_prices, strikes, maturities,
        num_samples=num_samples, num_burnin=num_burnin, num_chains=num_chains
    )
    summary = calibrator.summarize_posterior()
    calibrator.save_results()

    return posterior, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian MCMC calibration")
    parser.add_argument('--model', type=str, default='VarianceGamma',
                       choices=['VarianceGamma', 'CGMY'],
                       help='Lévy model to calibrate')
    parser.add_argument('--samples', type=int, default=5000,
                       help='Number of MCMC samples per chain')
    parser.add_argument('--burnin', type=int, default=2000,
                       help='Number of burn-in samples')
    parser.add_argument('--chains', type=int, default=4,
                       help='Number of parallel chains')

    args = parser.parse_args()

    print("="*80)
    print("BAYESIAN MCMC CALIBRATION")
    print("="*80)
    print("\nThis is a demonstration. In practice, you would:")
    print("1. Load real market option prices")
    print("2. Run MCMC calibration")
    print("3. Analyze posterior distributions")
    print("\nFor a working example, generate synthetic data first:")
    print("  python models/generate_dataset.py --num_samples 1")
    print("\nThen use that data for calibration.")
