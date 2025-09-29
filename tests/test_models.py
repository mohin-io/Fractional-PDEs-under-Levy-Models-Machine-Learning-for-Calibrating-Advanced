import pytest
import numpy as np
from models.pricing_engine.levy_models import variance_gamma_char_func, cgmy_char_func
from models.pricing_engine.fourier_pricer import carr_madan_pricer, price_surface

def test_variance_gamma_char_func_returns_complex():
    """Test that VG characteristic function returns complex numbers."""
    u = 0.5
    t = 1.0
    r = 0.05
    sigma = 0.2
    nu = 0.5
    theta = -0.1
    result = variance_gamma_char_func(u, t, r, sigma, nu, theta)
    assert isinstance(result, complex) or isinstance(result, np.complex128)

def test_cgmy_char_func_returns_complex():
    """Test that CGMY characteristic function returns complex numbers."""
    u = 0.5
    t = 1.0
    r = 0.05
    C = 0.1
    G = 5.0
    M = 5.0
    Y = 0.8
    result = cgmy_char_func(u, t, r, C, G, M, Y)
    assert isinstance(result, complex) or isinstance(result, np.complex128)

def test_carr_madan_pricer_returns_positive_price():
    """Test that Carr-Madan pricer returns a positive option price."""
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    nu = 0.5
    theta = -0.1
    char_func = lambda u, t, r: variance_gamma_char_func(u, t, r, sigma, nu, theta)
    price = carr_madan_pricer(S0, K, T, r, char_func)
    assert price > 0

def test_price_surface_returns_2d_array():
    """Test that price_surface returns a 2D numpy array."""
    params = {'sigma': 0.2, 'nu': 0.5, 'theta': -0.1}
    model_name = 'VarianceGamma'
    s0 = 100.0
    grid_strikes = np.array([90, 100, 110])
    grid_maturities = np.array([0.5, 1.0])
    r = 0.05
    surface = price_surface(params, model_name, s0, grid_strikes, grid_maturities, r)
    assert isinstance(surface, np.ndarray)
    assert surface.ndim == 2
    assert surface.shape == (len(grid_strikes), len(grid_maturities))
