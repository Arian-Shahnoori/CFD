import numpy as np


def rk2_upwind_stability_map(
    cfl_max=1.5,
    n_cfl=451,
    n_phi=601,
):
    """
    Calculate the RK2 stability map for the second-order
    upwind spatial discretization of the 1D advection equation.
    """

    # Fourier phase and Courant-number values.
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    cfl_values = np.linspace(0.0, cfl_max, n_cfl)

    # Create every combination of Fourier phase and CFL.
    Phi, C = np.meshgrid(phi, cfl_values)

    # Dimensionless eigenvalue: z = lambda * dt.
    z = -0.5 * C * (
        3.0
        - 4.0 * np.exp(-1j * Phi)
        + np.exp(-2j * Phi)
    )

    # Explicit midpoint RK2 amplification factor.
    G = 1.0 + z + 0.5 * z**2

    # Magnitude determines stability.
    abs_G = np.abs(G)

    # Check whether every sampled Fourier mode is stable
    # at each Courant number.
    max_abs_G = np.max(abs_G, axis=1)
    stable_cfl = max_abs_G <= 1.0 + 1e-10

    sampled_limit = (
        float(np.max(cfl_values[stable_cfl]))
        if np.any(stable_cfl)
        else 0.0
    )

    return {
        "phi": phi,
        "cfl_values": cfl_values,
        "Phi": Phi,
        "C": C,
        "abs_G": abs_G,
        "max_abs_G": max_abs_G,
        "sampled_cfl_limit": sampled_limit,
        "theoretical_cfl_limit": 0.5,
    }



def rk4_upwind_stability_map(
    cfl_max=1.2,
    n_cfl=601,
    n_phi=601,
):
    """
    Calculate the Fourier stability map for classical RK4
    combined with second-order upwind spatial discretization.
    """

    # Fourier phase and Courant-number ranges.
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    cfl_values = np.linspace(0.0, cfl_max, n_cfl)

    Phi, C = np.meshgrid(phi, cfl_values)

    # Dimensionless eigenvalue z = lambda * dt.
    z = -0.5 * C * (
        3.0
        - 4.0 * np.exp(-1j * Phi)
        + np.exp(-2j * Phi)
    )

    # Classical RK4 amplification factor.
    G = (
        1.0
        + z
        + z**2 / 2.0
        + z**3 / 6.0
        + z**4 / 24.0
    )

    abs_G = np.abs(G)

    # A CFL value is stable when every sampled Fourier mode
    # satisfies |G| <= 1.
    max_abs_G = np.max(abs_G, axis=1)
    stable_cfl = max_abs_G <= 1.0 + 1e-10

    sampled_limit = (
        float(np.max(cfl_values[stable_cfl]))
        if np.any(stable_cfl)
        else 0.0
    )

    return {
        "phi": phi,
        "cfl_values": cfl_values,
        "Phi": Phi,
        "C": C,
        "abs_G": abs_G,
        "max_abs_G": max_abs_G,
        "sampled_cfl_limit": sampled_limit,
        "theoretical_cfl_limit": 0.6963,
    }