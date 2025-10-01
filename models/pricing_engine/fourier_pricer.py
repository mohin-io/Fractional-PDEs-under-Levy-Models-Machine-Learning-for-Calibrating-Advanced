import numpy as np
from scipy.integrate import quad
from scipy.fft import fft, ifft
from scipy.interpolate import CubicSpline

# Import the characteristic functions
from models.pricing_engine.levy_models import variance_gamma_char_func, cgmy_char_func


def carr_madan_pricer(S0, K, T, r, char_func, q=0.0, option_type='call',
                       alpha=1.5, N=2**12, eta=0.1):
    """
    Prices European options using the Carr-Madan FFT method.

    Args:
        S0 (float): Spot price.
        K (float or np.ndarray): Strike price(s).
        T (float): Time to maturity.
        r (float): Risk-free rate.
        char_func (callable): Characteristic function of the log-price process.
                              It should accept (u, T, r, *model_params).
        q (float): Dividend yield (default: 0.0).
        option_type (str): 'call' or 'put' (default: 'call').
        alpha (float): Damping factor for the characteristic function.
                       Typically between 1 and 2 (default: 1.5).
        N (int): Number of points in the FFT. Must be a power of 2 (default: 2^12).
        eta (float): Step size for the integration (default: 0.1).

    Returns:
        float or np.ndarray: Option price(s).

    Examples:
        >>> # Price a VG call option
        >>> char_func = lambda u, t, r: variance_gamma_char_func(u, t, r, 0.2, 0.5, -0.1)
        >>> price = carr_madan_pricer(100, 100, 1.0, 0.05, char_func)
    """
    # Input validation
    if option_type not in ['call', 'put']:
        raise ValueError("option_type must be 'call' or 'put'")
    if N & (N - 1) != 0:
        raise ValueError("N must be a power of 2")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

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
    integrand = (
        np.exp(-r * T) * psi / (alpha**2 + alpha - v**2 + 1j * v * (2 * alpha + 1))
    )

    # Apply FFT
    fft_result = np.real(ifft(integrand * np.exp(-1j * b * v))) * N

    # Use CubicSpline for improved interpolation
    log_K_S0 = np.log(K / S0)

    # Filter valid k values (avoid extreme tails)
    valid_mask = (k > -5) & (k < 5)
    k_valid = k[valid_mask]
    fft_valid = fft_result[valid_mask]

    # Create cubic spline interpolator
    spline = CubicSpline(k_valid, fft_valid, extrapolate=False)
    call_prices = spline(log_K_S0) * S0

    # Handle extrapolation for strikes outside the range
    call_prices = np.where(np.isnan(call_prices),
                           np.interp(log_K_S0, k_valid, fft_valid) * S0,
                           call_prices)

    # Ensure non-negative prices
    call_prices = np.maximum(call_prices, 0.0)

    # Convert to put prices using put-call parity if needed
    if option_type == 'put':
        # Put-Call Parity: P = C - S0*exp(-qT) + K*exp(-rT)
        put_prices = call_prices - S0 * np.exp(-q * T) + K * np.exp(-r * T)
        return np.maximum(put_prices, 0.0)

    return call_prices


def compute_greeks(S0, K, T, r, char_func, q=0.0, option_type='call', epsilon=0.01):
    """
    Compute option Greeks using finite differences.

    Args:
        S0 (float): Spot price.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        char_func (callable): Characteristic function of the log-price process.
        q (float): Dividend yield (default: 0.0).
        option_type (str): 'call' or 'put' (default: 'call').
        epsilon (float): Step size for finite differences (default: 0.01).

    Returns:
        dict: Dictionary containing 'delta', 'gamma', 'vega', 'theta', 'rho'.

    Examples:
        >>> char_func = lambda u, t, r: variance_gamma_char_func(u, t, r, 0.2, 0.5, -0.1)
        >>> greeks = compute_greeks(100, 100, 1.0, 0.05, char_func)
        >>> print(greeks['delta'])
    """
    greeks = {}

    # Base price
    V = carr_madan_pricer(S0, K, T, r, char_func, q, option_type)

    # Delta: ∂V/∂S (first derivative w.r.t. spot)
    V_up = carr_madan_pricer(S0 + epsilon, K, T, r, char_func, q, option_type)
    V_down = carr_madan_pricer(S0 - epsilon, K, T, r, char_func, q, option_type)
    greeks['delta'] = (V_up - V_down) / (2 * epsilon)

    # Gamma: ∂²V/∂S² (second derivative w.r.t. spot)
    greeks['gamma'] = (V_up - 2 * V + V_down) / (epsilon ** 2)

    # Vega: ∂V/∂σ (derivative w.r.t. volatility)
    # Note: This is approximate as we perturb the characteristic function
    # For true Vega, need to perturb model-specific vol parameter
    # Here we use a proxy by scaling the spot volatility
    greeks['vega'] = None  # Model-specific implementation needed

    # Theta: ∂V/∂T (derivative w.r.t. time)
    epsilon_t = min(0.001, T / 10)  # Smaller time step
    if T > epsilon_t:
        V_theta = carr_madan_pricer(S0, K, T - epsilon_t, r, char_func, q, option_type)
        greeks['theta'] = -(V - V_theta) / epsilon_t  # Negative because time decays
    else:
        greeks['theta'] = None

    # Rho: ∂V/∂r (derivative w.r.t. risk-free rate)
    epsilon_r = 0.0001
    V_rho_up = carr_madan_pricer(S0, K, T, r + epsilon_r, char_func, q, option_type)
    greeks['rho'] = (V_rho_up - V) / epsilon_r

    return greeks


def price_surface(params, model_name, s0, grid_strikes, grid_maturities, r, q=0.0, option_type='call'):
    """
    Prices a surface of European options for a given model and parameters.

    Args:
        params (dict): Dictionary of model parameters.
        model_name (str): Name of the Levy model ('VarianceGamma' or 'CGMY').
        s0 (float): Spot price.
        grid_strikes (np.ndarray): Array of strike prices.
        grid_maturities (np.ndarray): Array of maturities.
        r (float): Risk-free rate.
        q (float): Dividend yield (default: 0.0).
        option_type (str): 'call' or 'put' (default: 'call').

    Returns:
        np.ndarray: 2D array of option prices (strikes x maturities).
    """
    price_matrix = np.zeros((len(grid_strikes), len(grid_maturities)))

    for i, T in enumerate(grid_maturities):
        if model_name == "VarianceGamma":
            char_func_t = lambda u, t, r: variance_gamma_char_func(
                u, t, r, params["sigma"], params["nu"], params["theta"]
            )
        elif model_name == "CGMY":
            char_func_t = lambda u, t, r: cgmy_char_func(
                u, t, r, params["C"], params["G"], params["M"], params["Y"]
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Price options for the current maturity
        option_prices_for_T = carr_madan_pricer(
            S0=s0, K=grid_strikes, T=T, r=r, char_func=char_func_t, q=q, option_type=option_type
        )
        price_matrix[:, i] = option_prices_for_T

    return price_matrix


if __name__ == "__main__":
    # Example Usage for Variance Gamma
    S0_val = 100.0
    r_val = 0.05
    strikes = np.array([90, 100, 110])
    maturities = np.array([0.5, 1.0])

    vg_params = {"sigma": 0.2, "nu": 0.5, "theta": -0.1}
    vg_surface = price_surface(
        vg_params, "VarianceGamma", S0_val, strikes, maturities, r_val
    )
    print("Variance Gamma Option Surface:")
    print(vg_surface)

    # Example Usage for CGMY
    cgmy_params = {"C": 0.1, "G": 5.0, "M": 5.0, "Y": 0.8}
    cgmy_surface = price_surface(
        cgmy_params, "CGMY", S0_val, strikes, maturities, r_val
    )
    print("\nCGMY Option Surface:")
    print(cgmy_surface)
