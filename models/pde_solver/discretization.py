"""
Finite Difference Discretization for Lévy PIDEs

This module implements finite difference schemes for Partial Integro-Differential
Equations (PIDEs) arising in Lévy option pricing.

The PIDE for option value V(S,t) under a Lévy process is:
    ∂V/∂t + (r-q)S∂V/∂S + 0.5σ²S²∂²V/∂S² + ∫[V(Se^y,t)-V(S,t)]ν(dy) = rV

where ν(dy) is the Lévy measure (jump distribution).

Numerical Methods:
1. Implicit-Explicit (IMEX) time stepping:
   - Implicit for diffusion (stable for large time steps)
   - Explicit for integral term (easier to implement)

2. Spatial discretization:
   - Log-transformation: x = log(S/K) for uniform grid
   - Central differences for derivatives
   - Quadrature for integral term

Author: Mohin Hasin (mohinhasin999@gmail.com)
Reference: Cont & Voltchkova (2005), d'Halluin et al. (2005)
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
from typing import Callable, Tuple, Dict, Optional
import warnings


class PIDESolver:
    """
    Finite difference solver for Lévy PIDEs.

    This solver implements an IMEX (Implicit-Explicit) scheme:
    - Diffusion term: Implicit (Crank-Nicolson)
    - Jump integral: Explicit
    - Time stepping: Backward in time (from T to 0)

    Attributes
    ----------
    S_min : float
        Minimum asset price in grid
    S_max : float
        Maximum asset price in grid
    N_space : int
        Number of spatial grid points
    N_time : int
        Number of time steps
    r : float
        Risk-free rate
    q : float
        Dividend yield
    levy_density : Callable
        Lévy density function ν(x)
    """

    def __init__(
        self,
        S_min: float,
        S_max: float,
        N_space: int,
        T: float,
        N_time: int,
        r: float,
        q: float,
        levy_density: Callable,
        sigma: float = 0.0
    ):
        """
        Initialize PIDE solver.

        Parameters
        ----------
        S_min : float
            Minimum asset price
        S_max : float
            Maximum asset price
        N_space : int
            Number of spatial grid points
        T : float
            Time to maturity
        N_time : int
            Number of time steps
        r : float
            Risk-free rate
        q : float
            Dividend yield
        levy_density : Callable
            Lévy density ν(x) for jump term
        sigma : float, optional
            Diffusion coefficient (default 0 for pure jump)
        """
        self.S_min = S_min
        self.S_max = S_max
        self.N_space = N_space
        self.T = T
        self.N_time = N_time
        self.r = r
        self.q = q
        self.levy_density = levy_density
        self.sigma = sigma

        # Create grids
        self._create_grids()

        # Build discretization matrices
        self._build_matrices()

    def _create_grids(self):
        """Create spatial and temporal grids."""
        # Spatial grid (uniform in log-space)
        self.x = np.linspace(np.log(self.S_min), np.log(self.S_max), self.N_space)
        self.S = np.exp(self.x)
        self.dx = self.x[1] - self.x[0]

        # Temporal grid
        self.t = np.linspace(0, self.T, self.N_time + 1)
        self.dt = self.T / self.N_time

        print(f"Grid created: {self.N_space} spatial × {self.N_time} time points")
        print(f"  dx = {self.dx:.6f}, dt = {self.dt:.6f}")

    def _build_matrices(self):
        """Build finite difference matrices for PIDE."""
        N = self.N_space
        dx = self.dx

        # Diffusion matrix (second derivative in log-space)
        # d²V/dx² with central differences
        main_diag = -2 * np.ones(N) / dx**2
        off_diag = np.ones(N - 1) / dx**2

        self.D2 = sparse.diags(
            [off_diag, main_diag, off_diag],
            offsets=[-1, 0, 1],
            shape=(N, N),
            format='csr'
        )

        # First derivative matrix (drift term)
        # dV/dx with central differences
        upper_diag = np.ones(N - 1) / (2 * dx)
        lower_diag = -np.ones(N - 1) / (2 * dx)

        self.D1 = sparse.diags(
            [lower_diag, upper_diag],
            offsets=[-1, 1],
            shape=(N, N),
            format='csr'
        )

        # Adjust boundary conditions (zero derivative at boundaries)
        # Forward difference at left boundary
        self.D1[0, :3] = [-3/(2*dx), 2/dx, -1/(2*dx)]
        # Backward difference at right boundary
        self.D1[-1, -3:] = [1/(2*dx), -2/dx, 3/(2*dx)]

        # Build jump integral matrix
        self._build_jump_matrix()

    def _build_jump_matrix(self):
        """Build matrix for jump integral term using quadrature."""
        N = self.N_space
        x = self.x

        # Jump integral: ∫[V(x+y) - V(x)] ν(dy)
        # Discretize using trapezoidal rule

        # Define jump size grid
        y_min, y_max = -3.0, 3.0  # Adjust based on Lévy density support
        N_jump = 200  # Number of quadrature points
        y_grid = np.linspace(y_min, y_max, N_jump)
        dy = y_grid[1] - y_grid[0]

        # Evaluate Lévy density
        levy_values = self.levy_density(y_grid)

        # Build interpolation matrix
        # J[i,j] = coefficient for V(x_j) when evaluating integral at x_i
        self.J = sparse.lil_matrix((N, N))

        for i in range(N):
            x_i = x[i]

            for y_k, nu_k in zip(y_grid, levy_values):
                # Find x_i + y_k in grid
                x_shifted = x_i + y_k

                # Linear interpolation
                if x_shifted < x[0] or x_shifted > x[-1]:
                    # Outside domain, use boundary value
                    continue

                # Find indices for interpolation
                j = np.searchsorted(x, x_shifted) - 1
                j = max(0, min(j, N - 2))  # Ensure valid index

                # Interpolation weights
                w = (x_shifted - x[j]) / (x[j + 1] - x[j])

                # Add contribution
                self.J[i, j] += dy * nu_k * (1 - w)
                self.J[i, j + 1] += dy * nu_k * w

            # Subtract diagonal (V(x) term)
            integral_mass = np.sum(levy_values * dy)
            self.J[i, i] -= integral_mass

        self.J = self.J.tocsr()

    def solve(
        self,
        payoff: Callable,
        boundary_conditions: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve PIDE backward in time.

        Parameters
        ----------
        payoff : Callable
            Terminal payoff function V(S, T) = payoff(S)
        boundary_conditions : dict, optional
            Boundary conditions at S_min and S_max
            Default: {'left': 0, 'right': 'linear_extrapolation'}

        Returns
        -------
        V : np.ndarray
            Option values on grid (N_space × N_time+1)
        S : np.ndarray
            Spatial grid (asset prices)

        Notes
        -----
        IMEX time stepping:
            (I - θ*dt*L_diff) V^{n+1} = (I + (1-θ)*dt*L_diff + dt*L_jump) V^n

        where L_diff = 0.5*σ²*D2 + (r-q)*D1 - r*I
              L_jump = J (jump integral operator)
              θ = 0.5 (Crank-Nicolson)
        """
        if boundary_conditions is None:
            boundary_conditions = {'left': 0.0, 'right': 'linear'}

        N = self.N_space
        M = self.N_time

        # Initialize solution array
        V = np.zeros((N, M + 1))

        # Terminal condition
        V[:, -1] = payoff(self.S)

        # Build system matrix for implicit step
        theta = 0.5  # Crank-Nicolson

        # Diffusion operator: L_diff = 0.5*σ²*D2 + (r-q)*D1 - r*I
        L_diff = 0.5 * self.sigma**2 * self.D2 + (self.r - self.q) * self.D1 - self.r * sparse.eye(N)

        # Implicit matrix: A = I - θ*dt*L_diff
        A = sparse.eye(N) - theta * self.dt * L_diff
        A = A.tocsr()

        # Explicit matrix: B = I + (1-θ)*dt*L_diff
        B = sparse.eye(N) + (1 - theta) * self.dt * L_diff
        B = B.tocsr()

        # Jump contribution (explicit)
        J_contrib = self.dt * self.J

        # Time stepping (backward)
        for n in range(M - 1, -1, -1):
            # Right-hand side
            rhs = B @ V[:, n + 1] + J_contrib @ V[:, n + 1]

            # Apply boundary conditions
            # Left boundary (S_min)
            if isinstance(boundary_conditions['left'], (int, float)):
                rhs[0] = boundary_conditions['left']
                A[0, :] = 0
                A[0, 0] = 1

            # Right boundary (S_max)
            if boundary_conditions['right'] == 'linear':
                # dV/dS = constant (linear extrapolation)
                rhs[-1] = 2 * V[-1, n + 1] - V[-2, n + 1]
                A[-1, :] = 0
                A[-1, -1] = 1
                A[-1, -2] = -1
            elif isinstance(boundary_conditions['right'], (int, float)):
                rhs[-1] = boundary_conditions['right']
                A[-1, :] = 0
                A[-1, -1] = 1

            # Solve linear system
            V[:, n] = splinalg.spsolve(A, rhs)

        return V, self.S

    def price_european_call(
        self,
        K: float,
        analytical_solution: Optional[Callable] = None
    ) -> Dict:
        """
        Price European call option using PIDE solver.

        Parameters
        ----------
        K : float
            Strike price
        analytical_solution : Callable, optional
            Analytical pricing function for comparison
            Should have signature: (S, K, T, r, q, params) -> price

        Returns
        -------
        dict
            Results containing:
            - 'price': Option price at S0
            - 'V': Full solution grid
            - 'S': Asset price grid
            - 'error': Error vs analytical (if provided)
        """
        # Define payoff
        def call_payoff(S):
            return np.maximum(S - K, 0.0)

        # Solve PIDE
        V, S = self.solve(call_payoff)

        # Extract price at t=0
        price_grid = V[:, 0]

        # Interpolate to get price at specific S0 (e.g., S0 = K for ATM)
        S0 = K  # Can be parameterized
        price = np.interp(S0, S, price_grid)

        result = {
            'price': price,
            'V': V,
            'S': S,
            'price_grid': price_grid
        }

        # Compare with analytical if provided
        if analytical_solution is not None:
            analytical_price = analytical_solution(S0)
            result['analytical'] = analytical_price
            result['error'] = abs(price - analytical_price)
            result['relative_error'] = abs(price - analytical_price) / analytical_price

        return result


def convergence_test(
    levy_density: Callable,
    analytical_solution: Callable,
    K: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    T: float = 1.0,
    sigma: float = 0.2,
    grid_sizes: list = None
) -> Dict:
    """
    Test convergence of PIDE solver by refining grid.

    Parameters
    ----------
    levy_density : Callable
        Lévy density function
    analytical_solution : Callable
        Analytical pricing function
    K : float
        Strike price
    r : float
        Risk-free rate
    q : float
        Dividend yield
    T : float
        Maturity
    sigma : float
        Diffusion coefficient
    grid_sizes : list, optional
        List of (N_space, N_time) tuples to test

    Returns
    -------
    dict
        Convergence results with errors for each grid size
    """
    if grid_sizes is None:
        grid_sizes = [(50, 25), (100, 50), (200, 100), (400, 200)]

    S_min, S_max = K * 0.2, K * 2.0
    S0 = K

    results = {
        'grid_sizes': [],
        'prices': [],
        'errors': [],
        'relative_errors': [],
        'cpu_times': []
    }

    analytical_price = analytical_solution(S0)

    for N_space, N_time in grid_sizes:
        import time
        start = time.time()

        # Create solver
        solver = PIDESolver(
            S_min=S_min, S_max=S_max, N_space=N_space,
            T=T, N_time=N_time, r=r, q=q,
            levy_density=levy_density, sigma=sigma
        )

        # Solve
        result = solver.price_european_call(K, analytical_solution)

        elapsed = time.time() - start

        # Store results
        results['grid_sizes'].append((N_space, N_time))
        results['prices'].append(result['price'])
        results['errors'].append(result['error'])
        results['relative_errors'].append(result['relative_error'])
        results['cpu_times'].append(elapsed)

        print(f"N_space={N_space:4d}, N_time={N_time:4d}: "
              f"Price={result['price']:.6f}, "
              f"Error={result['error']:.2e}, "
              f"Time={elapsed:.3f}s")

    return results


if __name__ == "__main__":
    # Example: Solve PIDE for Variance Gamma model
    from models.pde_solver.levy_processes import levy_density_vg
    import functools

    print("=" * 70)
    print("PIDE Solver - Variance Gamma Example")
    print("=" * 70)

    # VG parameters
    sigma_vg, nu, theta = 0.2, 0.3, -0.1

    # Create VG Lévy density
    levy_dens = functools.partial(
        levy_density_vg,
        sigma=sigma_vg, nu=nu, theta=theta
    )

    # Market parameters
    K = 100.0
    r, q, T = 0.05, 0.0, 1.0
    S_min, S_max = K * 0.3, K * 2.0
    N_space, N_time = 100, 50

    print(f"\nParameters:")
    print(f"  VG: σ={sigma_vg}, ν={nu}, θ={theta}")
    print(f"  Market: K={K}, r={r}, T={T}")
    print(f"  Grid: {N_space} × {N_time}")

    # Create solver
    solver = PIDESolver(
        S_min=S_min, S_max=S_max, N_space=N_space,
        T=T, N_time=N_time, r=r, q=q,
        levy_density=levy_dens, sigma=sigma_vg
    )

    # Solve for call option
    print(f"\nSolving PIDE...")
    result = solver.price_european_call(K)

    print(f"\nResults:")
    print(f"  Call price (ATM): {result['price']:.6f}")

    print("\n" + "=" * 70)
