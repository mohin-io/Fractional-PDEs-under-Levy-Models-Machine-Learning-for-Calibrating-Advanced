"""
Spectral Methods for Lévy Option Pricing

This module implements Fourier-based methods for option pricing under Lévy processes.
Spectral methods exploit the FFT (Fast Fourier Transform) to achieve O(N log N) complexity
instead of O(N²) for traditional finite difference methods.

Key Methods:
1. Carr-Madan FFT: Fourier transform of option payoff
2. Cosine expansion (COS method): More accurate for deep ITM/OTM options
3. CONV method: Convolution approach for path-dependent options

Mathematical Foundation:
Given characteristic function φ(u) of log-returns, the call option price is:
    C(K,T) = exp(-rT) * Re[∫ φ(u-i) * exp(-iu*log(K)) / (u²-iu) du]

Using FFT, this integral can be computed efficiently on a grid.

Author: Mohin Hasin (mohinhasin999@gmail.com)
Reference: Carr & Madan (1999), Fang & Oosterlee (2008)
"""

import numpy as np
from scipy import interpolate
from scipy.fft import fft, ifft
from typing import Callable, Dict, Tuple, Optional
import warnings


def carr_madan_fft(
    S0: float,
    K: np.ndarray,
    T: float,
    r: float,
    q: float,
    char_func: Callable,
    alpha: float = 1.5,
    N: int = 2**12,
    eta: float = 0.1,
    option_type: str = 'call'
) -> np.ndarray:
    """
    Enhanced Carr-Madan FFT option pricing with improved accuracy.

    This implementation includes:
    - CubicSpline interpolation for arbitrary strikes
    - Put-call parity for put options
    - Adaptive damping parameter α
    - Higher FFT resolution (N=2^12 default)

    Parameters
    ----------
    S0 : float
        Current asset price
    K : np.ndarray
        Array of strike prices
    T : float
        Time to maturity (years)
    r : float
        Risk-free rate
    q : float
        Dividend yield (use q=0 for non-dividend assets)
    char_func : Callable
        Characteristic function φ(u, T) of log-returns
        Should accept frequency array and return complex values
    alpha : float, optional
        Damping parameter (default 1.5)
        - For calls: typically 1.5
        - For puts: typically -1.5 (or use put-call parity)
    N : int, optional
        Number of FFT points (default 4096)
        Must be power of 2 for FFT efficiency
    eta : float, optional
        Grid spacing in frequency domain (default 0.1)
        Smaller η → wider strike coverage, more computation
    option_type : str, optional
        'call' or 'put' (default 'call')

    Returns
    -------
    np.ndarray
        Option prices corresponding to strike array K

    Raises
    ------
    ValueError
        If parameters are invalid

    Notes
    -----
    The method computes:
        C_T(k) = exp(-αk) / π * Re[∫₀^∞ exp(-iuk) ψ_T(u) du]
    where ψ_T(u) = exp(-rT) φ_T(u - (α+1)i) / (α² + α - u² + i(2α+1)u)

    The integral is approximated using trapezoidal rule + FFT.

    Examples
    --------
    >>> from models.pde_solver.levy_processes import variance_gamma_char_func_enhanced
    >>> import functools
    >>>
    >>> # Create VG characteristic function
    >>> vg_cf = functools.partial(
    ...     variance_gamma_char_func_enhanced,
    ...     sigma=0.2, nu=0.3, theta=-0.1
    ... )
    >>>
    >>> # Price options
    >>> strikes = np.linspace(80, 120, 41)
    >>> prices = carr_madan_fft(
    ...     S0=100, K=strikes, T=1.0, r=0.05, q=0.0,
    ...     char_func=vg_cf, option_type='call'
    ... )
    """
    # Input validation
    if S0 <= 0:
        raise ValueError(f"S0 must be positive, got {S0}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if N & (N - 1) != 0:
        warnings.warn(f"N={N} is not a power of 2, FFT will be slower")
    if option_type not in ['call', 'put']:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    # Step 1: Set up frequency grid
    # u_j = η * j, j = 0, 1, ..., N-1
    lambda_ = 2 * np.pi / (N * eta)  # Strike spacing in log-space
    b = N * lambda_ / 2  # Upper log-strike bound

    # Frequency grid
    u = np.arange(N) * eta

    # Log-strike grid
    k = -b + lambda_ * np.arange(N)

    # Step 2: Compute modified characteristic function
    # ψ(u) = exp(-rT) * φ(u - (α+1)i) / (α² + α - u² + i(2α+1)u)
    char_values = char_func(u - (alpha + 1) * 1j, T)

    denominator = alpha**2 + alpha - u**2 + 1j * (2 * alpha + 1) * u
    psi = np.exp(-r * T) * char_values / denominator

    # Step 3: Apply Simpson's rule weights for trapezoidal integration
    # w_j = η * (3 + (-1)^j - δ_{j,0}) / 3
    weights = np.ones(N) * eta
    weights[0] = eta / 2  # First point

    # Simpson's 1/3 rule pattern: 1, 4, 2, 4, 2, ..., 4, 1
    simpson_weights = np.ones(N)
    simpson_weights[0] = 1
    simpson_weights[-1] = 1
    simpson_weights[1:-1:2] = 4  # Odd indices
    simpson_weights[2:-1:2] = 2  # Even indices
    weights = weights * simpson_weights / 3

    # Step 4: Compute FFT
    # x_k = exp(ikb) * Re[FFT(exp(-ikb) * ψ * weights)]
    fft_input = np.exp(-1j * b * u) * psi * weights
    fft_output = fft(fft_input)

    # Real part and apply exponential factor
    call_values_log = np.exp(-alpha * k) / np.pi * np.real(
        np.exp(-1j * b * k) * fft_output
    )

    # Step 5: Convert log-strikes to strikes and interpolate
    strike_grid = S0 * np.exp(k)

    # Use cubic spline for smooth interpolation
    # Only use middle 80% of grid to avoid boundary effects
    N_mid = int(0.8 * N)
    start_idx = (N - N_mid) // 2
    end_idx = start_idx + N_mid

    spline = interpolate.CubicSpline(
        strike_grid[start_idx:end_idx],
        call_values_log[start_idx:end_idx],
        bc_type='natural',
        extrapolate=False
    )

    # Interpolate to requested strikes
    K = np.asarray(K)
    call_prices = spline(K)

    # Handle extrapolation with intrinsic value bounds
    mask_low = K < strike_grid[start_idx]
    mask_high = K > strike_grid[end_idx]

    call_prices[mask_low] = np.maximum(S0 * np.exp(-q * T) - K[mask_low] * np.exp(-r * T), 0)
    call_prices[mask_high] = np.maximum(S0 * np.exp(-q * T) - K[mask_high] * np.exp(-r * T), 0)

    # Step 6: Convert to puts if requested using put-call parity
    if option_type == 'put':
        # Put-Call Parity: P = C - S*exp(-qT) + K*exp(-rT)
        put_prices = call_prices - S0 * np.exp(-q * T) + K * np.exp(-r * T)
        return put_prices

    return call_prices


def cos_method(
    S0: float,
    K: np.ndarray,
    T: float,
    r: float,
    q: float,
    char_func: Callable,
    N: int = 2**8,
    L: float = 10.0,
    option_type: str = 'call'
) -> np.ndarray:
    """
    COS (Fourier-Cosine) method for option pricing.

    The COS method expands the density in a Fourier-cosine series and
    computes option prices by integrating against the payoff function.

    Advantages over Carr-Madan:
    - Better accuracy for deep ITM/OTM options
    - Explicit truncation range [a, b]
    - No damping parameter needed

    Parameters
    ----------
    S0 : float
        Current asset price
    K : np.ndarray
        Strike prices
    T : float
        Time to maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    char_func : Callable
        Characteristic function of log-returns
    N : int, optional
        Number of terms in cosine expansion (default 256)
    L : float, optional
        Truncation parameter (default 10.0)
        Integration range: [mean - L*std, mean + L*std]
    option_type : str, optional
        'call' or 'put'

    Returns
    -------
    np.ndarray
        Option prices

    Notes
    -----
    The method approximates the density as:
        f(x) ≈ 2/(b-a) * Σ_{k=0}^{N-1} Re[φ(kπ/(b-a)) * exp(-ikπa/(b-a))] * cos(kπ(x-a)/(b-a))

    Then integrates against payoff: ∫ f(x) * max(S0*exp(x) - K, 0) dx

    References
    ----------
    Fang, F. and Oosterlee, C.W. (2008), "A Novel Pricing Method for
    European Options Based on Fourier-Cosine Series Expansions"
    """
    if S0 <= 0 or T <= 0:
        raise ValueError("S0 and T must be positive")

    K = np.asarray(K)

    # Step 1: Determine integration range [a, b]
    # Use first few cumulants (mean, variance) of log-returns
    # Mean = log(S0) + (r - q - σ²/2)T (approx)
    # For Lévy: use characteristic function to get cumulants

    # Cumulants from characteristic function
    c1 = np.log(S0) + (r - q) * T  # First cumulant (mean)

    # Estimate variance from char function at small u
    small_u = 0.001
    cf_val = char_func(small_u, T)
    # Var ≈ -φ''(0) = d²/du² log(φ(u))|_{u=0}
    # Rough estimate: use finite difference
    cf_val_2u = char_func(2 * small_u, T)
    c2 = np.abs(np.log(cf_val_2u) - 2 * np.log(cf_val)) / small_u**2  # Approximate variance
    c2 = max(c2, 0.01)  # Ensure positive

    a = c1 - L * np.sqrt(c2)
    b = c1 + L * np.sqrt(c2)

    # Step 2: Compute cosine coefficients
    k = np.arange(N)
    u_k = k * np.pi / (b - a)

    # Characteristic function values
    cf_values = char_func(u_k, T)

    # Fourier-cosine coefficients: χ_k = φ(kπ/(b-a)) * exp(-ikπa/(b-a))
    chi_k = np.real(cf_values * np.exp(-1j * u_k * a))
    chi_k[0] = 0.5 * chi_k[0]  # Adjust first term

    # Step 3: Compute option prices for each strike
    prices = np.zeros_like(K, dtype=float)

    for i, strike in enumerate(K):
        # Log-strike
        k_log = np.log(strike / S0)

        if option_type == 'call':
            # Call payoff coefficients
            # V_k = 2/(b-a) * ∫_a^b cos(kπ(x-a)/(b-a)) * max(exp(x) - K/S0, 0) dx
            U_k = 2 / (b - a) * (
                _xi_call(k, a, b, k_log, S0) * chi_k
            )
        else:  # put
            # Put payoff coefficients
            U_k = 2 / (b - a) * (
                _xi_put(k, a, b, k_log, S0) * chi_k
            )

        prices[i] = np.exp(-r * T) * S0 * np.sum(U_k)

    return prices


def _xi_call(k: np.ndarray, a: float, b: float, c: float, S0: float) -> np.ndarray:
    """Helper: Fourier coefficients for call payoff."""
    # Integral of cos(k*(x-a)/(b-a)) * (exp(x) - exp(c)) for x in [c, b]
    # Returns coefficients for COS method

    xi = np.zeros_like(k, dtype=float)

    # k = 0 case
    if k[0] == 0:
        xi[0] = (np.exp(b) - np.exp(c)) - (b - c) * np.exp(c)

    # k > 0 cases
    k_pos = k[k > 0]
    kappa = k_pos * np.pi / (b - a)

    term1 = (np.cos(kappa * (c - a)) * (kappa * np.sinh(c) - np.cosh(c)) +
             kappa * np.sin(kappa * (c - a)) * np.cosh(c))
    term2 = (np.cos(kappa * (b - a)) * (kappa * np.sinh(b) - np.cosh(b)) +
             kappa * np.sin(kappa * (b - a)) * np.cosh(b))

    xi[k > 0] = (term2 - term1) / (1 + kappa**2)

    return xi


def _xi_put(k: np.ndarray, a: float, b: float, c: float, S0: float) -> np.ndarray:
    """Helper: Fourier coefficients for put payoff."""
    # Integral of cos(k*(x-a)/(b-a)) * (exp(c) - exp(x)) for x in [a, c]

    xi = np.zeros_like(k, dtype=float)

    # k = 0 case
    if k[0] == 0:
        xi[0] = (c - a) * np.exp(c) - (np.exp(c) - np.exp(a))

    # k > 0 cases
    k_pos = k[k > 0]
    kappa = k_pos * np.pi / (b - a)

    term1 = (kappa * np.sin(kappa * (a - a)) * np.cosh(a) -
             np.cos(kappa * (a - a)) * (kappa * np.sinh(a) - np.cosh(a)))
    term2 = (kappa * np.sin(kappa * (c - a)) * np.cosh(c) -
             np.cos(kappa * (c - a)) * (kappa * np.sinh(c) - np.cosh(c)))

    xi[k > 0] = (term2 - term1) / (1 + kappa**2)

    return xi


def compute_implied_volatility_surface(
    S0: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    r: float,
    q: float,
    char_func_factory: Callable,
    params: Dict[str, float],
    method: str = 'carr_madan'
) -> np.ndarray:
    """
    Compute implied volatility surface from Lévy model parameters.

    This function:
    1. Prices options using spectral methods
    2. Inverts Black-Scholes to get implied volatility
    3. Returns full surface (strikes × maturities)

    Parameters
    ----------
    S0 : float
        Current asset price
    strikes : np.ndarray
        Strike grid (1D array)
    maturities : np.ndarray
        Maturity grid (1D array)
    r : float
        Risk-free rate
    q : float
        Dividend yield
    char_func_factory : Callable
        Function that creates characteristic function given params
        Should have signature: (params) -> char_func(u, T)
    params : dict
        Model parameters (e.g., {'sigma': 0.2, 'nu': 0.3, 'theta': -0.1})
    method : str, optional
        Pricing method: 'carr_madan' or 'cos' (default 'carr_madan')

    Returns
    -------
    np.ndarray
        Implied volatility surface (strikes × maturities)

    Examples
    --------
    >>> from models.pde_solver.levy_processes import variance_gamma_char_func_enhanced
    >>> import functools
    >>>
    >>> # Define characteristic function factory
    >>> def vg_factory(params):
    ...     return functools.partial(
    ...         variance_gamma_char_func_enhanced,
    ...         sigma=params['sigma'],
    ...         nu=params['nu'],
    ...         theta=params['theta']
    ...     )
    >>>
    >>> # Compute IV surface
    >>> strikes = np.linspace(80, 120, 21)
    >>> maturities = np.linspace(0.25, 2.0, 8)
    >>> params = {'sigma': 0.2, 'nu': 0.3, 'theta': -0.1}
    >>>
    >>> iv_surface = compute_implied_volatility_surface(
    ...     S0=100, strikes=strikes, maturities=maturities,
    ...     r=0.05, q=0.0, char_func_factory=vg_factory,
    ...     params=params, method='carr_madan'
    ... )
    """
    from scipy.optimize import brentq
    from scipy.stats import norm

    # Create characteristic function
    char_func = char_func_factory(params)

    # Initialize surface
    iv_surface = np.zeros((len(strikes), len(maturities)))

    # Price options for each maturity
    for j, T in enumerate(maturities):
        if method == 'carr_madan':
            prices = carr_madan_fft(S0, strikes, T, r, q, char_func)
        elif method == 'cos':
            prices = cos_method(S0, strikes, T, r, q, char_func)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Invert Black-Scholes for each strike
        for i, (K, price) in enumerate(zip(strikes, prices)):
            try:
                # Black-Scholes formula for call
                def bs_price_diff(sigma):
                    d1 = (np.log(S0/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                    d2 = d1 - sigma * np.sqrt(T)
                    bs_price = S0*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
                    return bs_price - price

                # Solve for IV using Brent's method
                iv = brentq(bs_price_diff, 1e-6, 5.0, xtol=1e-6)
                iv_surface[i, j] = iv

            except ValueError:
                # If no solution found (e.g., price < intrinsic), set to NaN
                iv_surface[i, j] = np.nan

    return iv_surface


if __name__ == "__main__":
    # Example: Price VG options using both methods
    import functools
    from models.pde_solver.levy_processes import variance_gamma_char_func_enhanced

    print("=" * 70)
    print("Spectral Methods for Option Pricing - Examples")
    print("=" * 70)

    # VG parameters
    params_vg = {'sigma': 0.2, 'nu': 0.3, 'theta': -0.1}

    # Create characteristic function
    vg_cf = functools.partial(
        variance_gamma_char_func_enhanced,
        sigma=params_vg['sigma'],
        nu=params_vg['nu'],
        theta=params_vg['theta']
    )

    # Market parameters
    S0, r, q, T = 100.0, 0.05, 0.0, 1.0
    strikes = np.array([80, 90, 100, 110, 120])

    print(f"\nMarket Setup:")
    print(f"  S0 = {S0}, r = {r}, q = {q}, T = {T}")
    print(f"  VG params: σ={params_vg['sigma']}, ν={params_vg['nu']}, θ={params_vg['theta']}")

    # Method 1: Carr-Madan FFT
    print(f"\n1. Carr-Madan FFT Method")
    print("-" * 70)
    prices_cm = carr_madan_fft(S0, strikes, T, r, q, vg_cf, option_type='call')
    print(f"{'Strike':<10} {'Call Price':<15}")
    for K, price in zip(strikes, prices_cm):
        print(f"{K:<10.1f} {price:<15.4f}")

    # Method 2: COS Method
    print(f"\n2. COS Method")
    print("-" * 70)
    prices_cos = cos_method(S0, strikes, T, r, q, vg_cf, option_type='call')
    print(f"{'Strike':<10} {'Call Price':<15} {'Difference':<15}")
    for K, p_cm, p_cos in zip(strikes, prices_cm, prices_cos):
        diff = abs(p_cm - p_cos)
        print(f"{K:<10.1f} {p_cos:<15.4f} {diff:<15.6f}")

    print("\n" + "=" * 70)
