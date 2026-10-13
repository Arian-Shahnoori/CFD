import math
import numpy as np

from .spatial_operators import (
    rhs_second_order_upwind,
    rhs_second_order_central,
)
from .time_integrators import (
    rk2_midpoint_step,
    rk4_classical_step,
)


def solve_advection(
    n_cells: int,
    final_time: float,
    cfl: float = 0.25,
    length: float = 1.0,
    u: float = 2.0,
    n_steps: int = None,
    spatial_scheme: str = "2nd_upwind",
    temporal_scheme: str = "rk2",
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Solve the 1D linear advection equation using selectable
    spatial discretization and time-integration schemes.

    Parameters
    ----------
    n_cells : int
        Number of control volumes.
    final_time : float
        Simulation end time.
    cfl : float
        Target Courant number, used when n_steps is not specified.
    length : float
        Length of the computational domain.
    u : float
        Positive advection velocity.
    n_steps : int or None
        Explicit number of time steps. If None, it is determined
        from the target CFL.
    spatial_scheme : str
        "2nd_upwind" or "2nd_central".
    temporal_scheme : str
        "rk2" or "rk4".

    Returns
    -------
    x : ndarray
        Cell-centre coordinates.
    T : ndarray
        Numerical solution at final_time.
    dt : float
        Actual time-step size.
    n_steps : int
        Number of time steps used.
    """

    # Validate physical and numerical parameters.
    if not isinstance(n_cells, (int, np.integer)) or n_cells < 3:
        raise ValueError("n_cells must be an integer of at least 3.")

    if final_time <= 0.0 or length <= 0.0 or u <= 0.0:
        raise ValueError(
            "final_time, length, and u must be positive."
        )

    # Available spatial discretization schemes.
    spatial_operators = {
        "2nd_upwind": rhs_second_order_upwind,
        "2nd_central": rhs_second_order_central,
    }

    # Available time-integration schemes.
    time_integrators = {
        "rk2": rk2_midpoint_step,
        "rk4": rk4_classical_step,
    }

    spatial_scheme = spatial_scheme.lower()
    temporal_scheme = temporal_scheme.lower()

    if spatial_scheme not in spatial_operators:
        raise ValueError(
            f"Unknown spatial scheme: {spatial_scheme}"
        )

    if temporal_scheme not in time_integrators:
        raise ValueError(
            f"Unknown temporal scheme: {temporal_scheme}"
        )

    # Spatial discretization.
    dx = length / n_cells

    # Determine the number of time steps.
    if n_steps is None:
        if cfl <= 0.0:
            raise ValueError("CFL must be positive.")

        # Choose dt to satisfy the requested CFL.
        dt_max = cfl * dx / u
        n_steps = math.ceil(final_time / dt_max)

    else:
        # Use the prescribed number of time steps.
        if int(n_steps) != n_steps or n_steps < 1:
            raise ValueError(
                "n_steps must be a positive integer."
            )

        n_steps = int(n_steps)

    # Ensure the simulation reaches final_time exactly.
    dt = final_time / n_steps

    # Cell-centre coordinates and initial condition.
    x = (np.arange(n_cells) + 0.5) * dx
    T = np.zeros(n_cells, dtype=float)

    # Select the requested numerical methods.
    spatial_operator = spatial_operators[spatial_scheme]
    time_stepper = time_integrators[temporal_scheme]

    # Provide the spatial operator to the time integrator.
    def rhs(state, time):
        return spatial_operator(state, time, dx, u)

    # Advance the solution through all time steps.
    for step in range(n_steps):
        t_n = step * dt
        T = time_stepper(T, t_n, dt, rhs)

    return x, T, dt, n_steps