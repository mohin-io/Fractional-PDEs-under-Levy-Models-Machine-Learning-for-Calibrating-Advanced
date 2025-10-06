"""
Enhanced Lévy Process Implementations

This module provides comprehensive implementations of Lévy processes including:
- Variance Gamma (VG)
- CGMY (Carr-Geman-Madan-Yor)
- Normal Inverse Gaussian (NIG)

Each process includes:
1. Characteristic function for Fourier pricing
2. Lévy density for jump distribution
3. Parameter validation
4. Moment computations

Mathematical Foundation:
A Lévy process X_t has the Lévy-Khintchine representation:
    E[exp(iuX_t)] = exp(t * ψ(u))
where ψ(u) is the characteristic exponent.

Author: Mohin Hasin (mohinhasin999@gmail.com)
Reference: Cont & Tankov (2004), "Financial Modelling with Jump Processes"
"""

import numpy as np
from typing import Dict, Tuple
from scipy import special


def variance_gamma_char_func_enhanced(
    u: np.ndarray,
    T: float,
    sigma: float,
    nu: float,
    theta: float
) -> np.ndarray:
    """
    Enhanced Variance Gamma characteristic function with validation.

    The VG process is a pure jump Lévy process obtained by evaluating
    Brownian motion with drift at a random time given by a Gamma process.

    Characteristic function:
        φ(u) = (1 - iuθν + 0.5σ²νu²)^(-T/ν)

    Parameters
    ----------
    u : np.ndarray
        Frequencies for Fourier transform
    T : float
        Time to maturity (years)
    sigma : float
        Volatility of the Brownian motion (σ > 0)
    nu : float
        Variance rate of Gamma time change (ν > 0)
    theta : float
        Drift parameter (controls skewness)
        - θ < 0: negative skew (common for equity indices)
        - θ > 0: positive skew
        - θ = 0: symmetric distribution

    Returns
    -------
    np.ndarray
        Complex characteristic function values

    Raises
    ------
    ValueError
        If parameters are outside valid ranges

    Notes
    -----
    The VG process has:
    - Mean: θT
    - Variance: σ²T + νθ²T
    - Skewness: (2θ³ν² + 3θσ²ν) / (σ²  + θ²ν)^(3/2) * sqrt(T)
    - Excess kurtosis: 3ν(σ⁴ + 2θ²σ²ν + 2θ⁴ν²) / (σ² + θ²ν)²

    Examples
    --------
    >>> u = np.linspace(-10, 10, 100)
    >>> cf = variance_gamma_char_func_enhanced(u, T=1.0, sigma=0.2, nu=0.3, theta=-0.1)
    >>> np.abs(cf[0])  # Should be 1.0 at u=0
    1.0
    """
    # Parameter validation
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if nu <= 0:
        raise ValueError(f"nu must be positive, got {nu}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")

    # Ensure u is complex for numerical stability
    u = np.asarray(u, dtype=complex)

    # Compute characteristic function
    # φ(u) = (1 - iuθν + 0.5σ²νu²)^(-T/ν)
    exponent = -T / nu
    base = 1.0 - 1j * u * theta * nu + 0.5 * sigma**2 * nu * u**2

    # Handle potential numerical issues
    char_func = np.power(base, exponent)

    # Risk-neutral drift correction (ensures E[S_T/S_0] = exp(rT))
    # ω = (1/ν) * log(1 - θν - 0.5σ²ν)
    omega = (1.0 / nu) * np.log(1.0 - theta * nu - 0.5 * sigma**2 * nu)
    drift_correction = np.exp(-1j * u * omega * T)

    return char_func * drift_correction


def cgmy_char_func_enhanced(
    u: np.ndarray,
    T: float,
    C: float,
    G: float,
    M: float,
    Y: float
) -> np.ndarray:
    """
    Enhanced CGMY (Carr-Geman-Madan-Yor) characteristic function.

    The CGMY process generalizes VG with finer control over tail behavior.
    It includes:
    - Y < 0: Finite activity (finitely many jumps in finite time)
    - 0 ≤ Y < 1: Infinite activity, finite variation
    - 1 ≤ Y < 2: Infinite activity, infinite variation

    Characteristic function:
        φ(u) = exp(T * C * Γ(-Y) * [(M-iu)^Y - M^Y + (G+iu)^Y - G^Y])

    Parameters
    ----------
    u : np.ndarray
        Frequencies
    T : float
        Time to maturity
    C : float
        Overall level of activity (C > 0)
    G : float
        Rate of exponential decay for negative jumps (G > 0)
    M : float
        Rate of exponential decay for positive jumps (M > 0)
    Y : float
        Fine structure parameter (Y < 2)
        - Controls tail heaviness
        - Y closer to 2 → heavier tails

    Returns
    -------
    np.ndarray
        Complex characteristic function values

    Raises
    ------
    ValueError
        If parameters violate constraints

    Notes
    -----
    Special cases:
    - When Y → 0: CGMY reduces to Variance Gamma
    - When Y = 0.5: Related to Normal Inverse Gaussian

    The Lévy measure has density:
        ν(dx) = C * exp(-G|x|) / |x|^(1+Y) * 1_{x<0}
              + C * exp(-M*x) / x^(1+Y) * 1_{x>0}

    Examples
    --------
    >>> u = np.linspace(-5, 5, 50)
    >>> cf = cgmy_char_func_enhanced(u, T=1.0, C=0.1, G=5.0, M=5.0, Y=1.5)
    """
    # Parameter validation
    if C <= 0:
        raise ValueError(f"C must be positive, got {C}")
    if G <= 0:
        raise ValueError(f"G must be positive, got {G}")
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")
    if Y >= 2:
        raise ValueError(f"Y must be < 2 for integrability, got {Y}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")

    u = np.asarray(u, dtype=complex)

    # Compute characteristic exponent ψ(u)
    # Using Gamma function: Γ(-Y)
    gamma_neg_Y = special.gamma(-Y)

    # ψ(u) = C * Γ(-Y) * [(M-iu)^Y - M^Y + (G+iu)^Y - G^Y]
    term1 = np.power(M - 1j * u, Y) - M**Y
    term2 = np.power(G + 1j * u, Y) - G**Y

    psi_u = C * gamma_neg_Y * (term1 + term2)

    # Characteristic function: φ(u) = exp(T * ψ(u))
    char_func = np.exp(T * psi_u)

    # Risk-neutral drift correction
    # ω = -ψ(-i) to ensure martingale property
    psi_neg_i = C * gamma_neg_Y * (
        np.power(M + 1, Y) - M**Y + np.power(G - 1, Y) - G**Y
    )
    omega = -psi_neg_i
    drift_correction = np.exp(-1j * u * omega * T)

    return char_func * drift_correction


def nig_char_func(
    u: np.ndarray,
    T: float,
    alpha: float,
    beta: float,
    delta: float,
    mu: float = 0.0
) -> np.ndarray:
    """
    Normal Inverse Gaussian (NIG) characteristic function.

    The NIG distribution is a normal mean-variance mixture where the mixing
    distribution is inverse Gaussian. It provides a flexible model for
    asset returns with semi-heavy tails.

    Characteristic function:
        φ(u) = exp(iuμT + δT(√(α²-β²) - √(α²-(β+iu)²)))

    Parameters
    ----------
    u : np.ndarray
        Frequencies
    T : float
        Time to maturity
    alpha : float
        Tail heaviness parameter (α > 0)
        Larger α → lighter tails
    beta : float
        Asymmetry parameter (-α < β < α)
        β < 0 → negative skew
        β > 0 → positive skew
    delta : float
        Scale parameter (δ > 0)
    mu : float, optional
        Location parameter (drift), default 0.0

    Returns
    -------
    np.ndarray
        Complex characteristic function values

    Raises
    ------
    ValueError
        If parameters violate NIG constraints

    Notes
    -----
    Parameter constraints:
    - α > 0 (tail parameter)
    - δ > 0 (scale)
    - |β| < α (ensures integrability)

    Moments:
    - Mean: μ + δβ/√(α²-β²)
    - Variance: δα²/(α²-β²)^(3/2)
    - Skewness: 3β/(α*√(δ*√(α²-β²)))
    - Kurtosis: 3(1 + 4β²/α²)/(δ*√(α²-β²))

    Examples
    --------
    >>> u = np.linspace(-10, 10, 100)
    >>> cf = nig_char_func(u, T=1.0, alpha=15.0, beta=-3.0, delta=0.5, mu=0.0)
    """
    # Parameter validation
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta}")
    if abs(beta) >= alpha:
        raise ValueError(f"|beta| must be < alpha, got beta={beta}, alpha={alpha}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")

    u = np.asarray(u, dtype=complex)

    # Compute characteristic function
    # φ(u) = exp(iuμT + δT(√(α²-β²) - √(α²-(β+iu)²)))
    sqrt_alpha_beta = np.sqrt(alpha**2 - beta**2)
    sqrt_term = np.sqrt(alpha**2 - (beta + 1j * u)**2)

    char_func = np.exp(
        1j * u * mu * T + delta * T * (sqrt_alpha_beta - sqrt_term)
    )

    return char_func


def levy_density_vg(
    x: np.ndarray,
    sigma: float,
    nu: float,
    theta: float
) -> np.ndarray:
    """
    Variance Gamma Lévy density (jump distribution).

    The VG Lévy density describes the distribution of jumps:
        ν(x) = C/|x| * exp(θx/σ² ± √(θ²+2σ²/ν) * |x| / σ²)

    where C is a normalization constant and ± depends on sign of x.

    Parameters
    ----------
    x : np.ndarray
        Jump sizes
    sigma : float
        Volatility parameter (σ > 0)
    nu : float
        Variance rate (ν > 0)
    theta : float
        Drift/skewness parameter

    Returns
    -------
    np.ndarray
        Lévy density values

    Notes
    -----
    This is used in PIDE formulations where the jump integral is explicit:
        ∫ [V(S+x) - V(S)] ν(dx)

    Examples
    --------
    >>> x = np.linspace(-1, 1, 100)
    >>> density = levy_density_vg(x, sigma=0.2, nu=0.3, theta=-0.1)
    """
    if sigma <= 0 or nu <= 0:
        raise ValueError("sigma and nu must be positive")

    x = np.asarray(x, dtype=float)

    # Avoid division by zero at x=0
    x_safe = np.where(np.abs(x) < 1e-10, 1e-10, x)

    # Compute constants
    sqrt_term = np.sqrt(theta**2 + 2 * sigma**2 / nu)

    # Positive jumps (x > 0)
    pos_jumps = np.exp(
        theta * x_safe / sigma**2 - sqrt_term * np.abs(x_safe) / sigma**2
    ) / (nu * np.abs(x_safe))

    # Negative jumps (x < 0)
    neg_jumps = np.exp(
        theta * x_safe / sigma**2 + sqrt_term * np.abs(x_safe) / sigma**2
    ) / (nu * np.abs(x_safe))

    # Combine based on sign
    levy_dens = np.where(x >= 0, pos_jumps, neg_jumps)

    # Normalize (optional, for visualization)
    # Total mass should integrate to infinity for infinite activity

    return levy_dens


def levy_density_cgmy(
    x: np.ndarray,
    C: float,
    G: float,
    M: float,
    Y: float
) -> np.ndarray:
    """
    CGMY Lévy density.

    The CGMY Lévy measure has density:
        ν(x) = C * exp(-G|x|) / |x|^(1+Y)  for x < 0
        ν(x) = C * exp(-M*x) / x^(1+Y)     for x > 0

    Parameters
    ----------
    x : np.ndarray
        Jump sizes
    C : float
        Activity level (C > 0)
    G : float
        Decay for negative jumps (G > 0)
    M : float
        Decay for positive jumps (M > 0)
    Y : float
        Fine structure (Y < 2)

    Returns
    -------
    np.ndarray
        Lévy density values

    Notes
    -----
    The density blows up at x=0 when Y ≥ 0 (infinite activity).
    For Y < 0, there are finitely many jumps in finite time.

    Examples
    --------
    >>> x = np.linspace(-0.5, 0.5, 100)
    >>> x = x[x != 0]  # Exclude x=0
    >>> density = levy_density_cgmy(x, C=0.1, G=5.0, M=5.0, Y=1.5)
    """
    if C <= 0 or G <= 0 or M <= 0:
        raise ValueError("C, G, M must be positive")
    if Y >= 2:
        raise ValueError("Y must be < 2")

    x = np.asarray(x, dtype=float)
    x_safe = np.where(np.abs(x) < 1e-10, 1e-10, x)

    # Negative jumps (x < 0)
    neg_density = C * np.exp(-G * np.abs(x_safe)) / np.abs(x_safe)**(1 + Y)

    # Positive jumps (x > 0)
    pos_density = C * np.exp(-M * x_safe) / x_safe**(1 + Y)

    levy_dens = np.where(x < 0, neg_density, pos_density)

    return levy_dens


def compute_vg_moments(
    T: float,
    sigma: float,
    nu: float,
    theta: float
) -> Dict[str, float]:
    """
    Compute moments of the Variance Gamma distribution.

    Parameters
    ----------
    T : float
        Time horizon
    sigma : float
        Volatility
    nu : float
        Variance rate
    theta : float
        Drift

    Returns
    -------
    dict
        Dictionary containing mean, variance, skewness, kurtosis

    Examples
    --------
    >>> moments = compute_vg_moments(T=1.0, sigma=0.2, nu=0.3, theta=-0.1)
    >>> print(f"Mean: {moments['mean']:.4f}")
    >>> print(f"Variance: {moments['variance']:.4f}")
    """
    mean = theta * T
    variance = (sigma**2 + nu * theta**2) * T
    skewness = (
        (2 * theta**3 * nu**2 + 3 * theta * sigma**2 * nu)
        / (sigma**2 + theta**2 * nu)**(3/2)
        * np.sqrt(T)
    )

    # Excess kurtosis (kurtosis - 3 for normal)
    excess_kurtosis = (
        3 * nu * (sigma**4 + 2 * theta**2 * sigma**2 * nu + 2 * theta**4 * nu**2)
        / (sigma**2 + theta**2 * nu)**2
    )

    return {
        'mean': mean,
        'variance': variance,
        'std_dev': np.sqrt(variance),
        'skewness': skewness,
        'excess_kurtosis': excess_kurtosis
    }


def validate_parameters(model_type: str, params: Dict[str, float]) -> Tuple[bool, str]:
    """
    Validate parameters for a given Lévy model.

    Parameters
    ----------
    model_type : str
        One of: 'VG', 'CGMY', 'NIG'
    params : dict
        Parameter dictionary

    Returns
    -------
    tuple
        (is_valid, error_message)

    Examples
    --------
    >>> is_valid, msg = validate_parameters('VG', {'sigma': 0.2, 'nu': 0.3, 'theta': -0.1})
    >>> assert is_valid
    """
    try:
        if model_type == 'VG':
            if params['sigma'] <= 0:
                return False, "sigma must be positive"
            if params['nu'] <= 0:
                return False, "nu must be positive"
            # theta can be any real number
            return True, "Valid"

        elif model_type == 'CGMY':
            if params['C'] <= 0:
                return False, "C must be positive"
            if params['G'] <= 0:
                return False, "G must be positive"
            if params['M'] <= 0:
                return False, "M must be positive"
            if params['Y'] >= 2:
                return False, "Y must be < 2"
            return True, "Valid"

        elif model_type == 'NIG':
            if params['alpha'] <= 0:
                return False, "alpha must be positive"
            if params['delta'] <= 0:
                return False, "delta must be positive"
            if abs(params['beta']) >= params['alpha']:
                return False, "|beta| must be < alpha"
            return True, "Valid"

        else:
            return False, f"Unknown model type: {model_type}"

    except KeyError as e:
        return False, f"Missing parameter: {e}"


if __name__ == "__main__":
    # Example usage and validation
    print("=" * 60)
    print("Lévy Process Characteristic Functions - Examples")
    print("=" * 60)

    # Variance Gamma
    print("\n1. Variance Gamma Model")
    print("-" * 60)
    u = np.linspace(-10, 10, 5)
    cf_vg = variance_gamma_char_func_enhanced(u, T=1.0, sigma=0.2, nu=0.3, theta=-0.1)
    print(f"Frequencies: {u}")
    print(f"Char func:   {np.abs(cf_vg)}")
    print(f"At u=0:      {np.abs(cf_vg[2]):.6f} (should be 1.0)")

    moments = compute_vg_moments(T=1.0, sigma=0.2, nu=0.3, theta=-0.1)
    print(f"\nMoments:")
    for key, val in moments.items():
        print(f"  {key:20s}: {val:10.6f}")

    # CGMY
    print("\n2. CGMY Model")
    print("-" * 60)
    cf_cgmy = cgmy_char_func_enhanced(u, T=1.0, C=0.1, G=5.0, M=5.0, Y=1.5)
    print(f"Char func:   {np.abs(cf_cgmy)}")
    print(f"At u=0:      {np.abs(cf_cgmy[2]):.6f} (should be 1.0)")

    # NIG
    print("\n3. Normal Inverse Gaussian Model")
    print("-" * 60)
    cf_nig = nig_char_func(u, T=1.0, alpha=15.0, beta=-3.0, delta=0.5, mu=0.0)
    print(f"Char func:   {np.abs(cf_nig)}")
    print(f"At u=0:      {np.abs(cf_nig[2]):.6f} (should be 1.0)")

    # Parameter validation
    print("\n4. Parameter Validation")
    print("-" * 60)
    test_params_vg = {'sigma': 0.2, 'nu': 0.3, 'theta': -0.1}
    is_valid, msg = validate_parameters('VG', test_params_vg)
    print(f"VG params valid: {is_valid} - {msg}")

    print("\n" + "=" * 60)
