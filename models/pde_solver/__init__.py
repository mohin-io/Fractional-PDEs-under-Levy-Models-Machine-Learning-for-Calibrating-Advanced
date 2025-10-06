"""
Fractional PDE Solver for Lévy Models

This module implements advanced numerical methods for pricing options
under Lévy processes using Partial Integro-Differential Equations (PIDEs).

Components:
- levy_processes: Enhanced characteristic functions for VG, CGMY, NIG
- discretization: Finite difference schemes for diffusion and jump terms
- spectral_methods: Fourier-based methods with FFT acceleration
- convergence_tests: Validation and accuracy diagnostics

Author: Mohin Hasin (mohinhasin999@gmail.com)
Project: Fractional PDEs under Lévy Models
"""

__version__ = "1.0.0"
__author__ = "Mohin Hasin"
__email__ = "mohinhasin999@gmail.com"

__all__ = [
    "variance_gamma_char_func_enhanced",
    "cgmy_char_func_enhanced",
    "nig_char_func",
    "levy_density_vg",
    "levy_density_cgmy",
]
