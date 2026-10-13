import numpy as np
from collections.abc import Callable

RHSFunction = Callable[[np.ndarray, float], np.ndarray]


def rk2_midpoint_step(
    T: np.ndarray,
    t: float,
    dt: float,
    rhs: RHSFunction,
) -> np.ndarray:
    """
    Advance the solution by one explicit midpoint RK2 step.
    """
    # Stage 1: evaluate the RHS at the beginning of the step.
    k1 = rhs(T, t)

    # Estimate the solution at the temporal midpoint.
    T_half = T + 0.5 * dt * k1

    # Stage 2: evaluate the RHS at the midpoint.
    k2 = rhs(T_half, t + 0.5 * dt)

    # Advance to the next full time level.
    return T + dt * k2



def rk4_classical_step(T, t, dt, rhs):
    """Advance the solution by one classical RK4 time step."""

    k1 = rhs(T, t)

    k2 = rhs(
        T + 0.5 * dt * k1,
        t + 0.5 * dt,
    )

    k3 = rhs(
        T + 0.5 * dt * k2,
        t + 0.5 * dt,
    )

    k4 = rhs(
        T + dt * k3,
        t + dt,
    )

    return T + (dt / 6.0) * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    )