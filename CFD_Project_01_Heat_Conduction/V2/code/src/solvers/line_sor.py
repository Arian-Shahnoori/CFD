import time
import numpy as np


from src.residual import residual_norm


# =============================================================================
# Line Successive Over-Relaxation Solver
# =============================================================================

def line_sor(T, N, tolerance, omega):

    residual_history = []
    time_history = []

    residual = 1.0
    iteration = 0

    start_time = time.perf_counter()

    # ============================================================
    # Construct coefficient matrix
    # ============================================================

    A = np.zeros((N, N))

    for j in range(N):

        if j == 0:

            A[j, j] = 1.25
            A[j, j + 1] = -0.25

        elif j == N - 1:

            A[j, j] = 1.25
            A[j, j - 1] = -0.25

        else:

            A[j, j] = 1.0
            A[j, j + 1] = -0.25
            A[j, j - 1] = -0.25

    # ============================================================
    # Iterative solution
    # ============================================================

    while residual > tolerance:

        iteration += 1

        # ========================================================
        # Solve each vertical line implicitly
        # ========================================================

        for i in range(1, N + 1):

            b1 = T[i + 1, 1:N + 1]
            b2 = T[i - 1, 1:N + 1]
            b3 = T[i, 2:N + 2]
            b4 = T[i, 0:N]
            b5 = T[i, 1:N + 1]

            b = 0.25 * (b1 + b2 + b3 + b4) - b5

            # Solve the line system
            delta_T = np.linalg.solve(A, b)

            # Apply SOR relaxation
            T[i, 1:N + 1] += omega * delta_T

            # ====================================================
            # Update top and bottom ghost cells
            # ====================================================

            x = (i - 0.5) / N

            T[i, N + 1] = (
                2.0 * np.sin(np.pi * x)
                - T[i, N]
            )

            T[i, 0] = -T[i, 1]

        # ========================================================
        # Update left and right ghost cells
        # ========================================================

        T[0, 1:N + 1] = -T[1, 1:N + 1]
        T[N + 1, 1:N + 1] = -T[N, 1:N + 1]

        # ========================================================
        # Calculate residual
        # ========================================================

        residual = residual_norm(T, N)

        residual_history.append(residual)

        time_history.append(
            time.perf_counter() - start_time
        )

    return T, residual_history, time_history, iteration