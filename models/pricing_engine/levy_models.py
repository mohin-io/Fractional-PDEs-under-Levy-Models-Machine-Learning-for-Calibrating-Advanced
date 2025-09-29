"""
Module for defining the characteristic functions of various Lévy models.

Each class represents a specific model and provides a method to compute its
characteristic function, which is essential for Fourier-based pricing.
"""
import numpy as np

class VarianceGamma:
    """
    Implements the characteristic function for the Variance Gamma (VG) model.

    The VG model is defined by three parameters:
    - sigma: Volatility of the underlying Brownian motion.
    - nu: Variance rate of the gamma process. Controls kurtosis (fat tails).
           A small nu leads to fatter tails.
    - theta: Drift of the Brownian motion. Controls skewness.
             A negative theta leads to a left-skewed distribution (negative skew).
    """
    def __init__(self, sigma: float, nu: float, theta: float):
        """
        Initializes the Variance Gamma model with its parameters.
        """
        if nu <= 0:
            raise ValueError("Parameter 'nu' must be positive.")
        if sigma <= 0:
            raise ValueError("Parameter 'sigma' must be positive.")

        self.sigma = sigma
        self.nu = nu
        self.theta = theta

    def characteristic_function(self, u: np.ndarray, t: float) -> np.ndarray:
        """
        Calculates the characteristic function of the VG process.

        The formula is given by:
        phi(u, t) = (1 - i*u*theta*nu + 0.5*sigma^2*u^2*nu)^(-t/nu)

        Args:
            u: A numpy array of points in the Fourier domain where the function is evaluated.
            t: The time horizon in years.

        Returns:
            A numpy array of complex numbers representing the characteristic function values.
        """
        # Use 1j for the imaginary unit in Python
        i = 1j

        # The term inside the power
        inner_term = 1 - i * u * self.theta * self.nu + 0.5 * self.sigma**2 * u**2 * self.nu

        # The characteristic function
        phi = inner_term**(-t / self.nu)

        return phi

# Future extensions can be added here as new classes:
# class CGMY:
#     ...
#
# class MertonJumpDiffusion:
#     ...
