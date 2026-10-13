import numpy as np

from src.boundary_conditions import inlet_temperature


def rhs_second_order_upwind(
    T: np.ndarray,
    t: float,
    dx: float,
    u: float,
) -> np.ndarray:
    """
    Compute dT/dt using second-order upwind spatial discretization.

    Assumes positive transport velocity, u > 0.
    T contains cell-centred values.
    """
    T = np.asarray(T, dtype=float)

    if T.ndim != 1:
        raise ValueError("T must be a one-dimensional array.")
    if T.size < 2:
        raise ValueError("At least two control volumes are required.")
    if dx <= 0.0:
        raise ValueError("dx must be positive.")
    if u <= 0.0:
        raise ValueError("This implementation assumes u > 0.")

    rhs = np.empty_like(T)
    T_wall = inlet_temperature(t)

    # First cell: apply the inlet boundary condition.
    rhs[0] = -(2.0 * u / dx) * (T[0] - T_wall)

    # Second cell: use the inlet ghost-cell relation.
    rhs[1] = -(u / (2.0 * dx)) * (
        3.0 * T[1] - 5.0 * T[0] + 2.0 * T_wall
    )

    # Remaining cells: standard second-order upwind stencil.
    if T.size > 2:
        rhs[2:] = -(u / (2.0 * dx)) * (
            3.0 * T[2:]
            - 4.0 * T[1:-1]
            + T[:-2]
        )

    return rhs



def rhs_second_order_central(T, t, dx, u):
    """
    Compute dT/dt using second-order central differencing.

    The inlet uses the prescribed boundary condition.
    The outlet uses second-order upwind as a closure.
    """
    T = np.asarray(T, dtype=float)

    if T.ndim != 1 or T.size < 3:
        raise ValueError("T must contain at least three cells.")
    if dx <= 0 or u <= 0:
        raise ValueError("dx and u must be positive.")

    rhs = np.empty_like(T)
    T_wall = inlet_temperature(t)

    # First cell: prescribed inlet boundary.
    rhs[0] = -(u / (2.0 * dx)) * (
        T[1] - 2.0 * T_wall + T[0]
    )

    # Interior cells: second-order central scheme.
    rhs[1:-1] = -(u / (2.0 * dx)) * (
        T[2:] - T[:-2]
    )

    # Last cell: second-order upwind outlet closure.
    rhs[-1] = -(u / (2.0 * dx)) * (
        3.0 * T[-1] - 4.0 * T[-2] + T[-3]
    )

    return rhs