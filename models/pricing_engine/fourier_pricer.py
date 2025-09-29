import numpy as np
from scipy.integrate import quad
from scipy.fft import fft, ifft

# Import the characteristic functions
from models.pricing_engine.levy_models import variance_gamma_char_func, cgmy_char_func

def carr_madan_pricer(
    S0, K, T, r, char_func, alpha=1.5, N=2**10, eta=0.25
):
    """
    Prices a European call option using the Carr-Madan FFT method.

    Args:
        S0 (float): Spot price.
        K (float or np.ndarray): Strike price(s).
        T (float): Time to maturity.
        r (float): Risk-free rate.
        char_func (callable): Characteristic function of the log-price process.
                              It should accept (u, T, r, *model_params).
        alpha (float): Damping factor for the characteristic function.
                       Typically between 1 and 2.
        N (int): Number of points in the FFT. Must be a power of 2.
        eta (float): Step size for the integration.

    Returns:
        float or np.ndarray: Call option price(s).
    """
    # Ensure K is an array for vectorized operations
    K = np.atleast_1d(K)

    # Parameters for the FFT
    lambda_ = 2 * np.pi / (N * eta)
    b = N * lambda_ / 2
    v = np.arange(N) * eta
    k = -b + np.arange(N) * lambda_

    # Evaluate the damped characteristic function
    psi = char_func(v - 1j * alpha, T, r)
    
    # Compute the integrand
    integrand = np.exp(-r * T) * psi / (alpha**2 + alpha - v**2 + 1j * v * (2 * alpha + 1))

    # Apply FFT
    fft_result = np.real(ifft(integrand * np.exp(-1j * b * v))) * N

    # Interpolate or select the relevant FFT results for the given strikes
    # This is a simplified interpolation. For production, a more robust interpolation
    # or direct mapping to k values matching log(K/S0) would be needed.
    log_K_S0 = np.log(K / S0)
    call_prices = np.interp(log_K_S0, k, fft_result) * S0

    return call_prices

def price_surface(params, model_name, s0, grid_strikes, grid_maturities, r):
    """
    Prices a surface of European call options for a given model and parameters.

    Args:
        params (dict): Dictionary of model parameters.
        model_name (str): Name of the Levy model ('VarianceGamma' or 'CGMY').
        s0 (float): Spot price.
        grid_strikes (np.ndarray): Array of strike prices.
        grid_maturities (np.ndarray): Array of maturities.
        r (float): Risk-free rate.

    Returns:
        np.ndarray: 2D array of call option prices (strikes x maturities).
    """
    price_matrix = np.zeros((len(grid_strikes), len(grid_maturities)))

    for i, T in enumerate(grid_maturities):
        if model_name == 'VarianceGamma':
            char_func_t = lambda u, t, r: variance_gamma_char_func(
                u, t, r, params['sigma'], params['nu'], params['theta']
            )
        elif model_name == 'CGMY':
            char_func_t = lambda u, t, r: cgmy_char_func(
                u, t, r, params['C'], params['G'], params['M'], params['Y']
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Price options for the current maturity
        call_prices_for_T = carr_madan_pricer(S0=s0, K=grid_strikes, T=T, r=r, char_func=char_func_t)
        price_matrix[:, i] = call_prices_for_T

    return price_matrix

if __name__ == '__main__':
    # Example Usage for Variance Gamma
    S0_val = 100.0
    r_val = 0.05
    strikes = np.array([90, 100, 110])
    maturities = np.array([0.5, 1.0])

    vg_params = {'sigma': 0.2, 'nu': 0.5, 'theta': -0.1}
    vg_surface = price_surface(vg_params, 'VarianceGamma', S0_val, strikes, maturities, r_val)
    print("Variance Gamma Option Surface:")
    print(vg_surface)

    # Example Usage for CGMY
    cgmy_params = {'C': 0.1, 'G': 5.0, 'M': 5.0, 'Y': 0.8}
    cgmy_surface = price_surface(cgmy_params, 'CGMY', S0_val, strikes, maturities, r_val)
    print("\nCGMY Option Surface:")
    print(cgmy_surface)