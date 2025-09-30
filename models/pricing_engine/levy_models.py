import numpy as np


def variance_gamma_char_func(u, t, r, sigma, nu, theta):
    """
    Characteristic function for the Variance Gamma (VG) process.

    Args:
        u (float or np.ndarray): Argument for the characteristic function.
        t (float): Time to maturity.
        r (float): Risk-free rate.
        sigma (float): Volatility parameter.
        nu (float): Variance of the gamma subordinator (controls kurtosis).
        theta (float): Drift parameter (controls skewness).

    Returns:
        complex or np.ndarray: Value of the characteristic function.
    """
    omega = (1 / nu) * np.log(1 - theta * nu - 0.5 * sigma**2 * nu)
    exponent = 1j * u * (r + omega) * t - (t / nu) * np.log(
        1 - 1j * u * theta * nu + 0.5 * u**2 * sigma**2 * nu
    )
    return np.exp(exponent)


def cgmy_char_func(u, t, r, C, G, M, Y):
    """
    Characteristic function for the CGMY process.

    Args:
        u (float or np.ndarray): Argument for the characteristic function.
        t (float): Time to maturity.
        r (float): Risk-free rate.
        C (float): Parameter controlling the overall activity of jumps.
        G (float): Parameter controlling the right tail of the jump distribution.
        M (float): Parameter controlling the left tail of the jump distribution.
        Y (float): Parameter controlling the fine structure of jumps (Y < 2).

    Returns:
        complex or np.ndarray: Value of the characteristic function.
    """
    if Y >= 2:
        raise ValueError("Parameter Y for CGMY model must be less than 2.")

    omega = C * ((G - 1) ** Y - G**Y + (M + 1) ** Y - M**Y) / Y

    exponent = 1j * u * (r + omega) * t + t * C * (
        (M**Y - (M + 1j * u) ** Y) / Y + (G**Y - (G - 1j * u) ** Y) / Y
    )
    return np.exp(exponent)


if __name__ == "__main__":
    # Example Usage for Variance Gamma
    u_val = np.array([0.5, 1.0])
    t_val = 1.0
    r_val = 0.05
    sigma_val = 0.2
    nu_val = 0.5
    theta_val = -0.1

    vg_cf = variance_gamma_char_func(u_val, t_val, r_val, sigma_val, nu_val, theta_val)
    print(f"Variance Gamma Characteristic Function: {vg_cf}")

    # Example Usage for CGMY
    u_val = np.array([0.5, 1.0])
    t_val = 1.0
    r_val = 0.05
    C_val = 0.1
    G_val = 5.0
    M_val = 5.0
    Y_val = 0.8

    cgmy_cf = cgmy_char_func(u_val, t_val, r_val, C_val, G_val, M_val, Y_val)
    print(f"CGMY Characteristic Function: {cgmy_cf}")
