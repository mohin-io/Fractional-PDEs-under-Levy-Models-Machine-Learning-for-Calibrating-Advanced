"""
Fractional PDE Solver for Lévy Models

This module implements advanced numerical methods for pricing options
under Lévy processes using Partial Integro-Differential Equations (PIDEs).

Components:
- levy_processes: Enhanced characteristic functions for VG, CGMY, NIG
- spectral_methods: Fourier-based methods (Carr-Madan FFT, COS method)
- discretization: Finite difference schemes for diffusion and jump terms

Author: Mohin Hasin (mohinhasin999@gmail.com)
Project: Fractional PDEs & Lévy Processes: An ML Approach
Repository: https://github.com/mohin-io/Fractional-PDEs-under-Levy-Models-Machine-Learning-for-Calibrating-Advanced
"""

__version__ = "1.0.0"
__author__ = "Mohin Hasin"
__email__ = "mohinhasin999@gmail.com"

# Lévy Processes
from models.pde_solver.levy_processes import (
    variance_gamma_char_func_enhanced,
    cgmy_char_func_enhanced,
    nig_char_func,
    levy_density_vg,
    levy_density_cgmy,
    compute_vg_moments,
    validate_parameters,
)

# Spectral Methods
from models.pde_solver.spectral_methods import (
    carr_madan_fft,
    cos_method,
    compute_implied_volatility_surface,
)

# Finite Difference Methods
from models.pde_solver.discretization import (
    PIDESolver,
    convergence_test,
)

__all__ = [
    # Lévy processes
    "variance_gamma_char_func_enhanced",
    "cgmy_char_func_enhanced",
    "nig_char_func",
    "levy_density_vg",
    "levy_density_cgmy",
    "compute_vg_moments",
    "validate_parameters",
    # Spectral methods
    "carr_madan_fft",
    "cos_method",
    "compute_implied_volatility_surface",
    # Finite difference
    "PIDESolver",
    "convergence_test",
]
