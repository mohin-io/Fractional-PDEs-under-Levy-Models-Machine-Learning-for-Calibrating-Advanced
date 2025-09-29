"""
Module for pricing European options using Fourier methods, specifically the
Carr-Madan algorithm with Fast Fourier Transform (FFT).
"""
import numpy as np
from scipy.interpolate import interp1d

# Import the model definitions
from models.pricing_engine.levy_models import VarianceGamma

def price_european_call_fft(model, s0: float, k: float, t: float, r: float) -> float:
    """
    Prices a European call option using the FFT-based Carr-Madan method.

    Args:
        model: An instance of a Lévy model class (e.g., VarianceGamma).
        s0: Initial stock price.
        k: Strike price.
        t: Time to maturity in years.
        r: Risk-free interest rate (annualized).

    Returns:
        The price of the European call option.
    """
    # FFT parameters
    N = 2**14  # Number of points, should be a power of 2
    alpha = 1.5  # Dampening factor for integrability
    
    # Grid spacing in Fourier domain (eta)
    eta = 0.1
    # Grid spacing in log-strike domain (lambda)
    lambda_ = (2 * np.pi) / (N * eta)

    # Create the grid for the log-strike k
    # The grid is centered around log(s0)
    k_u = np.arange(N) * lambda_ - (N * lambda_) / 2

    # Create the grid for the Fourier domain u
    u = np.arange(N) * eta

    # --- Compute the modified characteristic function ---
    # We need to evaluate the characteristic function of the log-price process.
    # Let X_t be the Lévy process. The log-price is S_t = s0 * exp(r*t + X_t).
    # The characteristic function of log(S_t) is phi_log_S(u) = exp(i*u*(log(s0)+r*t)) * phi_X(u)
    # However, the Carr-Madan formula uses a modified, dampened characteristic function.
    
    # Characteristic function of the Lévy process X_t
    phi_X = model.characteristic_function(u - (alpha + 1) * 1j, t)
    
    # The term to be transformed
    # This is the Fourier transform of the dampened call price
    psi = np.exp(-r * t) * phi_X / ((alpha + 1j * u) * (alpha + 1 + 1j * u))

    # --- Perform the FFT ---
    fft_result = np.fft.fft(psi) * eta

    # --- Calculate the call prices from the FFT result ---
    # The result of the FFT needs to be scaled and shifted
    call_prices_on_grid = (np.exp(-alpha * k_u) / np.pi) * np.real(fft_result)

    # --- Interpolate to find the price for the desired strike k ---
    # The strikes corresponding to our grid of call prices
    strike_grid = s0 * np.exp(k_u)

    # Use linear interpolation to find the price at the specific strike k
    # We use fill_value='extrapolate' to handle strikes outside the grid, though
    # a well-chosen grid should avoid this.
    interpolator = interp1d(strike_grid, call_prices_on_grid, kind='linear', fill_value="extrapolate")
    
    call_price = interpolator(k)

    return float(call_price)

def price_surface(params: dict, model_name: str, s0: float, grid_strikes: np.ndarray, grid_maturities: np.ndarray, r: float) -> np.ndarray:
    """
    Prices a full surface of options for given grids of strikes and maturities.

    This is a convenience wrapper around the core FFT pricer to be used for
    generating the synthetic dataset.

    Args:
        params: Dictionary of parameters for the specified model.
        model_name: The name of the model to use (e.g., 'VarianceGamma').
        s0: Initial stock price.
        grid_strikes: A numpy array of strike prices.
        grid_maturities: A numpy array of maturities in years.
        r: Risk-free interest rate.

    Returns:
        A 2D numpy array where rows correspond to strikes and columns to maturities.
    """
    if model_name == 'VarianceGamma':
        model = VarianceGamma(**params)
    else:
        raise NotImplementedError(f"Model '{model_name}' is not implemented.")

    price_matrix = np.zeros((len(grid_strikes), len(grid_maturities)))

    for j, t in enumerate(grid_maturities):
        # For each maturity, we can price all strikes in a vectorized way if the
        # pricer is adapted, but for clarity, we loop here.
        # The core FFT pricer already gives us prices on a grid, so a more
        # efficient implementation would compute the FFT once per maturity.
        # For this implementation, we call the pricer for each strike.
        for i, k in enumerate(grid_strikes):
            price_matrix[i, j] = price_european_call_fft(model, s0, k, t, r)

    return price_matrix
