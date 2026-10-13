import time
from src.boundary_condition import apply_boundary_conditions
from src.residual import residual_norm


# =============================================================================
# Gauss-Seidel Solver
# =============================================================================

def gauss_seidel(T, N, tolerance):

    residual_history = []
    time_history = []

    residual = 1.0
    iteration = 0

    start_time = time.perf_counter()

    while residual > tolerance:

        iteration += 1

        for j in range(1, N+1):
            for i in range(1, N+1):

                T[i, j] = 0.25 * (
                    T[i+1, j]
                    + T[i-1, j]
                    + T[i, j+1]
                    + T[i, j-1]
                )

        apply_boundary_conditions(T, N)

        residual = residual_norm(T, N)

        residual_history.append(residual)

        elapsed_time = time.perf_counter() - start_time
        time_history.append(elapsed_time)

    return T, residual_history, time_history, iteration