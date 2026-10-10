import time
from src.boundary_condition import apply_boundary_conditions
from src.residual import residual_norm


# =============================================================================
# Successive Over Relaxxation Solver
# =============================================================================
def sor(T, N, tolerance, omega):

    residual_history = []
    time_history = []

    residual = 1.0
    iteration = 0

    start_time = time.perf_counter()

    while residual > tolerance:

        iteration += 1

        for j in range(1, N + 1):
            for i in range(1, N + 1):

                T_gs = 0.25 * (
                    T[i+1, j]
                    + T[i-1, j]
                    + T[i, j+1]
                    + T[i, j-1]
                )

                T[i, j] = T[i, j] + omega * (T_gs - T[i, j])

        apply_boundary_conditions(T, N)

        residual = residual_norm(T, N)

        residual_history.append(residual)
        time_history.append(time.perf_counter() - start_time)

    return T, residual_history, time_history, iteration