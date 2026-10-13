import time
import numpy as np

from src.boundary_condition import apply_boundary_conditions


# =============================================================================
# Fully Implicit Direct Solver
# =============================================================================

def implicit_solver(N):

    if N < 2:
        raise ValueError("N must be at least 2.")

    start_time = time.perf_counter()

    # ============================================================
    # Initialize the coefficient matrix and right-hand-side vector
    # ============================================================

    number_of_unknowns = N**2

    A = np.zeros((number_of_unknowns, number_of_unknowns))
    d = np.zeros(number_of_unknowns)

    # Cell-center coordinates and top-wall temperatures
    x = (np.arange(N) + 0.5) / N
    T_top = np.sin(np.pi * x)

    T_left = 0.0
    T_right = 0.0
    T_bottom = 0.0

    # Map a two-dimensional cell index to the linear system index
    def kp(i, j):
        return j * N + i

    # ============================================================
    # Assemble the coefficient matrix and right-hand-side vector
    # ============================================================

    for j in range(N):
        for i in range(N):

            k = kp(i, j)

            # Standard five-point Laplace stencil
            A[k, k] = -4.0

            # ----------------------------------------------------
            # Left neighbour or left boundary
            # ----------------------------------------------------

            if i > 0:
                A[k, kp(i - 1, j)] = 1.0
            else:
                A[k, k] -= 1.0
                d[k] -= 2.0 * T_left

            # ----------------------------------------------------
            # Right neighbour or right boundary
            # ----------------------------------------------------

            if i < N - 1:
                A[k, kp(i + 1, j)] = 1.0
            else:
                A[k, k] -= 1.0
                d[k] -= 2.0 * T_right

            # ----------------------------------------------------
            # Bottom neighbour or bottom boundary
            # ----------------------------------------------------

            if j > 0:
                A[k, kp(i, j - 1)] = 1.0
            else:
                A[k, k] -= 1.0
                d[k] -= 2.0 * T_bottom

            # ----------------------------------------------------
            # Top neighbour or top boundary
            # ----------------------------------------------------

            if j < N - 1:
                A[k, kp(i, j + 1)] = 1.0
            else:
                A[k, k] -= 1.0
                d[k] -= 2.0 * T_top[i]

    # ============================================================
    # Solve the linear system
    # ============================================================

    solution = np.linalg.solve(A, d)

    # Convert the one-dimensional solution into the 2D field.
    # The transpose preserves the project's T[i, j] convention.
    T = np.zeros((N + 2, N + 2))

    T[1:N + 1, 1:N + 1] = solution.reshape(N, N).T

    # Reconstruct ghost-cell values from the boundary conditions
    apply_boundary_conditions(T, N)

    cpu_time = time.perf_counter() - start_time

    return T, cpu_time